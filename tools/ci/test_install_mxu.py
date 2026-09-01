import fnmatch
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ci import install_mxu


class InstallMxuTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.install_dir = self.root / "install"
        self._create_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _write(path: Path, content: str = "fixture") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _create_fixture(self) -> None:
        interface = {
            "interface_version": 2,
            "name": "MaaStellaSora",
            "license": "MIT",
            "github": "https://github.com/MaaStellaSora/MaaStellaSora",
            "version": "0.0.0",
            "custom_title": "星塔助手",
            "mirrorchyan_rid": "SSAH",
            "mirrorchyan_multiplatform": True,
            "agent": {
                "child_exec": "python",
                "child_args": ["-u", "./../agent/main.py"],
                "timeout": -1,
            },
            "controller": [
                {
                    "name": "桌面端",
                    "type": "Win32",
                    "win32": {},
                    "attach_resource_path": ["{PROJECT_DIR}/resource/windows"],
                },
                {"name": "安卓端", "type": "Adb"},
            ],
            "resource": [
                {"name": "官服", "path": ["{PROJECT_DIR}/resource/base"]},
                {
                    "name": "国际服",
                    "path": [
                        "{PROJECT_DIR}/resource/base",
                        "{PROJECT_DIR}/resource/en",
                    ],
                },
            ],
            "import": ["resource/tasks/login.json"],
        }
        self._write(
            self.root / "assets" / "interface.json",
            json.dumps(interface, ensure_ascii=False),
        )
        self._write(self.root / "assets" / "logo.ico")
        self._write(self.root / "assets" / "resource" / "tasks" / "login.json")
        self._write(self.root / "assets" / "resource" / "windows" / "base.json")
        self._write(self.root / "assets" / "resource" / "en" / "base.json")
        for name in install_mxu.REQUIRED_OCR_FILES:
            self._write(
                self.root / "assets" / "resource" / "base" / "model" / "ocr" / name
            )
        self._write(self.root / "agent" / "main.py")
        self._write(self.root / "agent" / "custom" / "action.py")
        self._write(self.root / "agent" / "__pycache__" / "main.pyc")

        for name in install_mxu.PROJECT_FILES:
            self._write(self.root / name, f"project {name}")
        self._write(
            self.root / "assets" / "MaaCommonAssets" / "LICENSE",
            "MaaCommonAssets license",
        )

        for name in install_mxu.REQUIRED_MAAFW_FILES:
            self._write(self.root / "deps" / "bin" / name)
        self._write(
            self.root / "deps" / "LICENSE-MaaFramework",
            "MaaFramework license",
        )
        self._write(self.root / "deps" / "bin" / "plugins" / "runtime.dll")
        self._write(self.root / "deps" / "bin" / "debug.pdb")
        self._write(
            self.root / "deps" / "share" / "MaaAgentBinary" / "maatouch" / "maatouch"
        )
        self._write(
            self.root / "deps" / "share" / "MaaAgentBinary" / "minitouch" / "minitouch"
        )

        self._write(self.root / "MXU" / "mxu.exe")
        self._write(self.root / "MXU" / "LICENSE", "MXU license")
        self._write(self.root / "MXU" / "README.md", "upstream readme")
        self._write(self.root / "MXU" / "mxu.pdb")

        self._write(self.install_dir / "python" / "python.exe")
        self._write(
            self.install_dir / "deps" / "maafw-5.13.0b5-py3-none-any.whl"
        )

    def test_builds_and_validates_package(self) -> None:
        source_interface = (self.root / "assets" / "interface.json").read_text(
            encoding="utf-8"
        )

        with patch.object(install_mxu, "configure_ocr_model"):
            output = install_mxu.build_package("v1.2.3", working_dir=self.root)

        interface = install_mxu.validate_package(output)
        self.assertEqual(interface["version"], "v1.2.3")
        self.assertEqual(interface["title"], "星塔助手 MXU")
        self.assertEqual(interface["icon"], "Assets/logo.ico")
        self.assertEqual(interface["license"], "LICENSE")
        self.assertEqual(interface["resource"][0]["path"], ["resource/base"])
        self.assertEqual(
            interface["controller"][0]["attach_resource_path"],
            ["resource/windows"],
        )
        self.assertNotIn("custom_title", interface)
        self.assertNotIn("mirrorchyan_rid", interface)
        self.assertNotIn("mirrorchyan_multiplatform", interface)
        self.assertNotIn("timeout", interface["agent"])
        self.assertFalse((output / "maafw" / "debug.pdb").exists())
        self.assertFalse((output / "agent" / "__pycache__").exists())
        self.assertTrue((output / "maafw" / "plugins" / "runtime.dll").exists())
        self.assertEqual(
            (output / "LICENSE-MXU").read_text(encoding="utf-8"), "MXU license"
        )
        self.assertEqual(
            (output / "LICENSE-MaaFramework").read_text(encoding="utf-8"),
            "MaaFramework license",
        )
        self.assertEqual(
            (output / "LICENSE-MaaCommonAssets").read_text(encoding="utf-8"),
            "MaaCommonAssets license",
        )
        self.assertEqual(
            (output / "README.md").read_text(encoding="utf-8"),
            "project README.md",
        )
        self.assertEqual(
            (self.root / "assets" / "interface.json").read_text(encoding="utf-8"),
            source_interface,
        )

    def test_validation_rejects_missing_interface_path(self) -> None:
        with patch.object(install_mxu, "configure_ocr_model"):
            output = install_mxu.build_package("v1.2.3", working_dir=self.root)
        (output / "resource" / "tasks" / "login.json").unlink()

        with self.assertRaisesRegex(FileNotFoundError, "references missing file"):
            install_mxu.validate_package(output)

    def test_rejects_dirty_or_unsafe_staging(self) -> None:
        self._write(self.install_dir / "unexpected.txt")
        with self.assertRaisesRegex(ValueError, "fresh staging directory"):
            with patch.object(install_mxu, "configure_ocr_model"):
                install_mxu.build_package("v1.2.3", working_dir=self.root)

        with self.assertRaisesRegex(ValueError, "strict child"):
            install_mxu.validate_staging_paths(
                self.root, self.root, self.root / "deps", self.root / "MXU"
            )

    def test_transform_rejects_invalid_resource_paths(self) -> None:
        interface = {
            "agent": {},
            "resource": [{"path": "resource/base"}],
            "controller": [],
        }

        with self.assertRaisesRegex(ValueError, "resource.path"):
            install_mxu.transform_interface(interface, "v1.2.3")

    def test_real_interface_uses_mxu_paths_without_changing_source(self) -> None:
        source_path = install_mxu.WORKING_DIR / "assets" / "interface.json"
        source_text = source_path.read_text(encoding="utf-8")
        transformed = install_mxu.transform_interface(
            json.loads(source_text), "v1.2.3"
        )

        self.assertNotIn(install_mxu.PROJECT_DIR_PREFIX, json.dumps(transformed))
        self.assertEqual(
            transformed["controller"][0]["attach_resource_path"],
            ["resource/windows"],
        )
        self.assertEqual(source_path.read_text(encoding="utf-8"), source_text)

    def test_release_asset_name_avoids_mfa_and_mirror_patterns(self) -> None:
        asset_name = "MaaStellaSora-mxu-win-amd64-v1.2.3.zip"
        self.assertFalse(
            fnmatch.fnmatchcase(asset_name, "MaaStellaSora-win-x86_64-*")
        )
        workflow = (
            install_mxu.WORKING_DIR / ".github" / "workflows" / "install.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "name: MaaStellaSora-mxu-win-amd64-${{ needs.meta.outputs.tag }}",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
