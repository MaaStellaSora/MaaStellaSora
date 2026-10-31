"""把发行包的可执行文件换成项目自己的名字与图标。

发行包的外壳是 MFAAvalonia（MaaFramework 的通用 GUI），原来直接复制它的文件，导致打出来
的可执行文件仍叫 `MFAAvalonia`、文件图标也是 MFAAvalonia 的，和项目本身的名字对不上。
本脚本在 `tools/ci/install.py` 跑完、MFA 文件已经合进 `install/` 之后运行，改两处：

1. **可执行文件名**：改成 `install/interface.json` 里的 `name`（Windows 带 `.exe`）。
   .NET 的 apphost 里硬编码的是它要加载的 `<程序集名>.dll`，与自身文件名无关，
   所以改名不影响启动（已核对二进制里确实嵌着 `MFAAvalonia.dll`）。
2. **Windows 文件图标**：改写 PE 的 `RT_ICON` / `RT_GROUP_ICON` 资源，用 `Assets/logo.ico`
   （这个文件本来就被用作窗口/任务栏图标，这里只是让资源管理器里的文件图标也一致）。

只用标准库：ctypes 直接调 Win32 的 `BeginUpdateResource` / `UpdateResource`，
不引入 rcedit 之类的外部打包工具。Linux/macOS 只做改名。
"""

from __future__ import annotations

import ctypes
import json
import re
import struct
import sys
from ctypes import wintypes
from pathlib import Path

OLD_APP_NAME = "MFAAvalonia"
RT_ICON = 3
RT_GROUP_ICON = 14
LANG_NEUTRAL = 0

## 项目代号会被当作文件名用，先按各平台都安全的白名单校验一次。
## interface.json 的 name 只要求非空字符串，理论上可以带路径分隔符或平台非法字符。
SAFE_APP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]*$")


def read_app_name(install_dir: Path) -> str:
    """从发行态 interface.json 取项目名；取不到返回空串。"""

    path = install_dir / "interface.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(data.get("name") or "").strip()


def rename_executable(install_dir: Path, app_name: str) -> Path | None:
    """把 MFAAvalonia 的可执行文件改成 app_name；找不到返回 None。"""

    for suffix in (".exe", ""):
        source = install_dir / f"{OLD_APP_NAME}{suffix}"
        if not source.is_file():
            continue
        target = install_dir / f"{app_name}{suffix}"
        if target.exists():
            target.unlink()
        source.rename(target)
        ## rename 保留源文件模式，但打包步骤里 find -name "MFAAvalonia*" 的兜底 chmod
        ## 改名后匹配不到新名字，这里显式补上执行位（Windows 上是空操作）。
        if suffix == "":
            target.chmod(0o755)
        print(f"[branding] rename {source.name} -> {target.name}")
        return target
    ## 幂等：重跑时旧名字已经不在、新名字在，仍要把图标应用上去
    for suffix in (".exe", ""):
        existing = install_dir / f"{app_name}{suffix}"
        if existing.is_file():
            print(f"[branding] {existing.name} already renamed, apply icon only")
            return existing
    print("[branding] warn: MFAAvalonia executable not found, skip rename")
    return None


## ---------------------------------------------------------------- Windows 图标


def _load_ico(path: Path) -> list[dict]:
    """把 .ico 拆成每个尺寸的图像数据。"""

    data = path.read_bytes()
    _reserved, image_type, count = struct.unpack_from("<HHH", data, 0)
    if image_type != 1:
        raise ValueError(f"not an icon file: {path}")
    entries: list[dict] = []
    offset = 6
    for _ in range(count):
        width, height, colors, _r, planes, bits, size, image_offset = struct.unpack_from(
            "<BBBBHHII", data, offset
        )
        entries.append(
            {
                "width": width or 256,
                "height": height or 256,
                "colors": colors,
                "planes": planes,
                "bits": bits,
                "image": data[image_offset : image_offset + size],
            }
        )
        offset += 16
    return entries


def _resource_ids(kernel32, module: int, resource_type: int) -> list[int]:
    """列出某个资源类型下所有整数 ID。"""

    found: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HMODULE, wintypes.LPVOID, wintypes.LPVOID, wintypes.LPARAM
    )
    kernel32.EnumResourceNamesW.restype = wintypes.BOOL
    kernel32.EnumResourceNamesW.argtypes = [
        wintypes.HMODULE, wintypes.LPVOID, callback_type, wintypes.LPARAM,
    ]

    def _callback(_module, _type, name, _param):
        found.append(ctypes.cast(name, ctypes.c_void_p).value)
        return True

    kernel32.EnumResourceNamesW(
        module, ctypes.c_void_p(resource_type), callback_type(_callback), 0
    )
    return found


def set_windows_icon(exe: Path, ico: Path) -> None:
    """用 Win32 资源 API 把 exe 的文件图标换成 ico。"""

    if sys.platform != "win32":
        print("[branding] not Windows, skip PE icon")
        return
    if not ico.is_file():
        print(f"[branding] warn: icon not found: {ico}")
        return

    entries = _load_ico(ico)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.BeginUpdateResourceW.restype = wintypes.HANDLE
    kernel32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
    kernel32.UpdateResourceW.restype = wintypes.BOOL
    kernel32.UpdateResourceW.argtypes = [
        wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID,
        wintypes.WORD, wintypes.LPVOID, wintypes.DWORD,
    ]
    kernel32.EndUpdateResourceW.restype = wintypes.BOOL
    kernel32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]
    kernel32.LoadLibraryExW.restype = wintypes.HMODULE
    kernel32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    kernel32.FreeLibrary.restype = wintypes.BOOL
    kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]

    ## 先读出原有资源 ID，删除时要按真实 ID（MFA 的 GROUP_ICON 用的是 32512）
    module = kernel32.LoadLibraryExW(str(exe), None, 0x2)
    icon_ids: list[int] = []
    group_ids: list[int] = []
    if module:
        icon_ids = _resource_ids(kernel32, module, RT_ICON)
        group_ids = _resource_ids(kernel32, module, RT_GROUP_ICON)
        kernel32.FreeLibrary(module)

    handle = kernel32.BeginUpdateResourceW(str(exe), False)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        for resource_id in icon_ids:
            kernel32.UpdateResourceW(
                handle, ctypes.c_void_p(RT_ICON), ctypes.c_void_p(resource_id),
                LANG_NEUTRAL, None, 0,
            )
        for resource_id in group_ids:
            kernel32.UpdateResourceW(
                handle, ctypes.c_void_p(RT_GROUP_ICON), ctypes.c_void_p(resource_id),
                LANG_NEUTRAL, None, 0,
            )

        for index, entry in enumerate(entries, start=1):
            image = entry["image"]
            buffer = ctypes.create_string_buffer(image, len(image))
            if not kernel32.UpdateResourceW(
                handle, ctypes.c_void_p(RT_ICON), ctypes.c_void_p(index),
                LANG_NEUTRAL, buffer, len(image),
            ):
                raise ctypes.WinError(ctypes.get_last_error())

        group = struct.pack("<HHH", 0, 1, len(entries))
        for index, entry in enumerate(entries, start=1):
            group += struct.pack(
                "<BBBBHHIH",
                0 if entry["width"] >= 256 else entry["width"],
                0 if entry["height"] >= 256 else entry["height"],
                entry["colors"],
                0,
                entry["planes"],
                entry["bits"],
                len(entry["image"]),
                index,
            )
        group_buffer = ctypes.create_string_buffer(group, len(group))
        group_id = group_ids[0] if group_ids else 32512
        if not kernel32.UpdateResourceW(
            handle, ctypes.c_void_p(RT_GROUP_ICON), ctypes.c_void_p(group_id),
            LANG_NEUTRAL, group_buffer, len(group),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        kernel32.EndUpdateResourceW(handle, True)
        raise
    if not kernel32.EndUpdateResourceW(handle, False):
        raise ctypes.WinError(ctypes.get_last_error())
    print(f"[branding] icon -> {exe.name} ({len(entries)} size(s))")


def main(argv: list[str]) -> None:
    """argv: <install_dir>"""

    if len(argv) != 2:
        raise SystemExit("usage: app_branding.py <install_dir>")
    install_dir = Path(argv[1]).resolve()
    if not install_dir.is_dir():
        raise SystemExit(f"install_dir does not exist: {install_dir}")

    app_name = read_app_name(install_dir)
    if not app_name:
        raise SystemExit("interface.json has no name, cannot rename executable")
    if not SAFE_APP_NAME.match(app_name):
        raise SystemExit(
            f"interface.json name {app_name!r} is not usable as a file name"
        )

    executable = rename_executable(install_dir, app_name)
    if executable is not None:
        set_windows_icon(executable, install_dir / "Assets" / "logo.ico")


if __name__ == "__main__":
    main(sys.argv)
