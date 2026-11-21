from pathlib import Path
import tempfile
import unittest

from tools.ci.package_common import copy_agent_files


class TowerNeuralPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        self.package = self.root / "package"

    def _write(self, relative_path, content="fixture"):
        path = self.source / "agent" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_non_windows_package_omits_only_the_configured_checkpoint(self):
        self._write("main.py")
        self._write("custom/climb_tower_neural/__init__.py")
        self._write(
            "custom/climb_tower_neural/model.json",
            '{"checkpoint":"models/replacement.npz"}',
        )
        self._write("custom/climb_tower_neural/models/replacement.npz", "large model")
        self._write("custom/climb_tower_neural/models/M213.npz", "old model")
        self._write("custom/climb_tower_neural/models/metadata.json")
        self._write("custom/climb_tower_neural/potential_name_catalog.json")

        copy_agent_files(self.source, self.package, include_neural_model=False)

        self.assertTrue((self.package / "agent/main.py").is_file())
        self.assertTrue(
            (self.package / "agent/custom/climb_tower_neural/__init__.py").is_file()
        )
        model_dir = self.package / "agent/custom/climb_tower_neural/models"
        self.assertFalse((model_dir / "replacement.npz").exists())
        self.assertTrue((model_dir / "M213.npz").is_file())
        self.assertTrue(
            (model_dir / "metadata.json").is_file()
        )
        self.assertTrue(
            (self.package / "agent/custom/climb_tower_neural/potential_name_catalog.json").is_file()
        )

    def test_windows_package_keeps_configured_checkpoint(self):
        self._write("main.py")
        self._write(
            "custom/climb_tower_neural/model.json",
            '{"checkpoint":"models/replacement.npz"}',
        )
        self._write(
            "custom/climb_tower_neural/models/replacement.npz", "large model"
        )

        copy_agent_files(self.source, self.package)

        self.assertTrue(
            (
                self.package
                / "agent/custom/climb_tower_neural/models/replacement.npz"
            ).is_file()
        )


if __name__ == "__main__":
    unittest.main()
