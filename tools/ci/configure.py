from pathlib import Path

import shutil

from package_common import REQUIRED_OCR_FILES, require_file


ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


def configure_ocr_model(
    assets_dir: Path = ASSETS_DIR, ocr_dir: Path | None = None
) -> None:
    """保留完整的既有模型，或向指定目录导入默认 OCR 模型。"""
    ocr_dir = ocr_dir or assets_dir / "resource" / "base" / "model" / "ocr"
    if ocr_dir.exists():
        for name in REQUIRED_OCR_FILES:
            require_file(ocr_dir / name)
        print("Found existing OCR directory, skipping default OCR model import.")
        return

    source = assets_dir / "MaaCommonAssets" / "OCR" / "ppocr_v6" / "small"
    for name in REQUIRED_OCR_FILES:
        require_file(source / name)
    shutil.copytree(source, ocr_dir)


if __name__ == "__main__":
    configure_ocr_model()

    print("OCR model configured.")
