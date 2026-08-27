import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from download_deps import (  # noqa: E402
    build_download_command,
    validate_maafw_wheel,
)


class BuildDownloadCommandTest(unittest.TestCase):
    def test_adds_exact_maafw_prerelease(self):
        command = build_download_command(
            Path("deps"),
            "win_amd64",
            python_version="3.12",
            maafw_version="5.13.0b5",
        )

        self.assertIn("--pre", command)
        self.assertIn("maafw==5.13.0b5", command)
        self.assertEqual(command[command.index("--platform") + 1], "win_amd64")
        self.assertEqual(command[command.index("--python-version") + 1], "3.12")

    def test_rejects_invalid_maafw_version(self):
        with self.assertRaises(ValueError):
            build_download_command(Path("deps"), maafw_version="5.13.0-beta.5")


class ValidateMaafwWheelTest(unittest.TestCase):
    def test_accepts_exact_single_wheel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = Path(temp_dir, "maafw-5.13.0b5-py3-none-win_amd64.whl")
            wheel.touch()
            validate_maafw_wheel(temp_dir, "5.13.0b5")

    def test_rejects_wrong_or_multiple_wheels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "maafw-5.13.0b4-py3-none-any.whl").touch()
            with self.assertRaises(RuntimeError):
                validate_maafw_wheel(temp_dir, "5.13.0b5")

            Path(temp_dir, "maafw-5.13.0b5-py3-none-any.whl").touch()
            with self.assertRaises(RuntimeError):
                validate_maafw_wheel(temp_dir, "5.13.0b5")


if __name__ == "__main__":
    unittest.main()
