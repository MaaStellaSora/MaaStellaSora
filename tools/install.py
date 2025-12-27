from pathlib import Path

import os
import shutil
import sys
import subprocess

try:
    import jsonc
except ModuleNotFoundError as e:
    raise ImportError(
        "Missing dependency 'json-with-comments' (imported as 'jsonc').\n"
        f"Install it with:\n  {sys.executable} -m pip install json-with-comments\n"
        "Or add it to your project's requirements."
    ) from e

from configure import configure_ocr_model


working_dir = Path(__file__).parent.parent.resolve()
install_path = working_dir / Path("install")
version = len(sys.argv) > 1 and sys.argv[1] or "v0.0.1"

# the first parameter is self name
if sys.argv.__len__() < 4:
    print("Usage: python install.py <version> <os> <arch>")
    print("Example: python install.py v1.0.0 win x86_64")
    sys.exit(1)

os_name = sys.argv[2]
arch = sys.argv[3]


def get_dotnet_platform_tag():
    """自动检测当前平台并返回对应的dotnet平台标签"""
    if os_name == "win" and arch == "x86_64":
        platform_tag = "win-x64"
    elif os_name == "win" and arch == "aarch64":
        platform_tag = "win-arm64"
    elif os_name == "macos" and arch == "x86_64":
        platform_tag = "osx-x64"
    elif os_name == "macos" and arch == "aarch64":
        platform_tag = "osx-arm64"
    elif os_name == "linux" and arch == "x86_64":
        platform_tag = "linux-x64"
    elif os_name == "linux" and arch == "aarch64":
        platform_tag = "linux-arm64"
    else:
        print("Unsupported OS or architecture.")
        print("available parameters:")
        print("version: e.g., v1.0.0")
        print("os: [win, macos, linux, android]")
        print("arch: [aarch64, x86_64]")
        sys.exit(1)

    return platform_tag


def install_deps():
    if not (working_dir / "deps" / "bin").exists():
        print('Please download the MaaFramework to "deps" first.')
        print('请先下载 MaaFramework 到 "deps"。')
        sys.exit(1)

    if os_name == "android":
        shutil.copytree(
            working_dir / "deps" / "bin",
            install_path,
            dirs_exist_ok=True,
        )
    else:
        shutil.copytree(
            working_dir / "deps" / "bin",
            install_path / "runtimes" / get_dotnet_platform_tag() / "native",
            ignore=shutil.ignore_patterns(
                "*MaaDbgControlUnit*",
                "*MaaThriftControlUnit*",
                "*MaaRpc*",
                "*MaaHttp*",
            ),
            dirs_exist_ok=True,
        )

    shutil.copytree(
        working_dir / "deps" / "share" / "MaaAgentBinary",
        install_path / "MaaAgentBinary",
        dirs_exist_ok=True,
    )


def install_resource():

    configure_ocr_model()

    shutil.copytree(
        working_dir / "assets" / "resource",
        install_path / "resource",
        dirs_exist_ok=True,
    )
    shutil.copy2(
        working_dir / "assets" / "interface.json",
        install_path,
    )

    with open(install_path / "interface.json", "r", encoding="utf-8") as f:
        interface = jsonc.load(f)

    interface["version"] = version

    with open(install_path / "interface.json", "w", encoding="utf-8") as f:
        jsonc.dump(interface, f, ensure_ascii=False, indent=4)


def install_chores():
    shutil.copy2(
        working_dir / "README.md",
        install_path,
    )
    shutil.copy2(
        working_dir / "LICENSE",
        install_path,
    )


def create_venv():
    """创建Python虚拟环境并安装依赖"""
    venv_path = install_path / "venv"

    # 创建虚拟环境
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    # 确定pip路径和python路径
    # 直接使用sys.executable创建的venv，应该使用与sys.executable相同的路径结构
    # 例如，如果sys.executable是/usr/bin/python3，那么venv中的python应该在venv/bin/python
    # 如果sys.executable是C:\Python39\python.exe，那么venv中的python应该在venv/Scripts/python.exe

    # 获取sys.executable的父目录名，判断是Scripts还是bin
    if sys.platform == "win32":
        # Windows平台，使用Scripts目录
        scripts_dir = "Scripts"
        python_ext = ".exe"
    else:
        # Linux/macOS平台，使用bin目录
        scripts_dir = "bin"
        python_ext = ""

    pip_path = venv_path / scripts_dir / f"pip{python_ext}"
    python_path = venv_path / scripts_dir / f"python{python_ext}"

    # 确保python_path存在
    if not python_path.exists():
        # 如果不存在，尝试另一种目录结构（可能是交叉编译环境）
        alt_scripts_dir = "bin" if scripts_dir == "Scripts" else "Scripts"
        alt_python_ext = "" if python_ext == ".exe" else ".exe"

        pip_path = venv_path / alt_scripts_dir / f"pip{alt_python_ext}"
        python_path = venv_path / alt_scripts_dir / f"python{alt_python_ext}"

    if not python_path.exists():
        # 遍历venv目录，寻找python可执行文件
        for root, dirs, files in os.walk(venv_path):
            for file in files:
                if file == "python" or file == "python.exe":
                    python_path = Path(root) / file
                    pip_path = Path(root) / ("pip" + os.path.splitext(file)[1])
                    break
            if python_path.exists():
                break

    if not python_path.exists():
        # 如果还是找不到，列出venv目录结构，方便调试
        print(f"Error: Python executable not found in {venv_path}")
        print("venv目录结构:")
        for root, dirs, files in os.walk(venv_path):
            level = root.replace(str(venv_path), "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        raise FileNotFoundError(
            f"Python executable not found in virtual environment at {venv_path}"
        )

    # 升级pip
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True
    )

    # 安装agent所需的依赖
    # 首先检查是否存在agent/requirements.txt文件
    agent_req = working_dir / "agent" / "requirements.txt"
    if agent_req.exists():
        subprocess.run([str(pip_path), "install", "-r", str(agent_req)], check=True)

    return python_path


def install_agent():
    shutil.copytree(
        working_dir / "agent",
        install_path / "agent",
        dirs_exist_ok=True,
    )


if __name__ == "__main__":
    install_deps()
    install_resource()
    install_chores()
    install_agent()

    # 创建虚拟环境
    python_path = create_venv()

    # 更新interface.json中的agent配置，使用虚拟环境中的Python
    with open(install_path / "interface.json", "r", encoding="utf-8") as f:
        interface = jsonc.load(f)

    # 检查虚拟环境的实际结构，确定相对路径
    # 实际python_path是例如：/home/runner/.../install/venv/bin/python
    # 我们需要的相对路径是：./venv/bin/python
    # 从python_path中提取相对路径
    # 首先获取python_path相对于install_path的路径
    relative_python_path = str(python_path.relative_to(install_path))
    # 将路径分隔符统一为/，确保跨平台兼容
    relative_python_path = "./" + relative_python_path.replace("\\", "/")

    interface["agent"]["child_exec"] = relative_python_path

    with open(install_path / "interface.json", "w", encoding="utf-8") as f:
        jsonc.dump(interface, f, ensure_ascii=False, indent=4)

    print(f"Install to {install_path} successfully.")
