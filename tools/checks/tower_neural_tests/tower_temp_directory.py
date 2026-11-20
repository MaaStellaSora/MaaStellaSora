"""Clean up files replaced by the Windows tower sampling tests."""
from tempfile import TemporaryDirectory
from time import sleep


class TowerTempDirectory(TemporaryDirectory):
    def cleanup(self):
        try:
            super().cleanup()
        except OSError as error:
            if getattr(error, "winerror", None) != 145:
                raise
            # A deleted file can briefly keep its directory from being removed.
            sleep(.05)
            super().cleanup()
