"""
调试用的agent截图保存功能，会把OpenCV截图以png格式保存到项目目录的 debug/agent_image 目录下

说明：
- 本模块完全不依赖 Pillow
- 直接使用 Python 标准库中的 zlib/struct 手工生成 PNG 文件
- 适用于便携版 Python / 无第三方图像库的环境

使用方法：
    from utils.image_handler import save_image

    save_image(image, "说明信息")
"""

from __future__ import annotations

import hashlib
import re
import struct
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

from utils import logger as logger_module
logger = logger_module.get_logger("utils_image_handler")

save_dir = Path(__file__).resolve().parents[2] / "debug" / "agent_image"
save_dir.mkdir(parents=True, exist_ok=True)


def _sanitize_comment(comment: str) -> str:
    """将文件名中的非法字符替换为下划线，确保文件名合法。"""
    text = str(comment or "untitled")
    text = re.sub(r"[\\/:*?\"<>|]", "_", text)
    text = re.sub(r"\s+", "_", text).strip("_. ")
    return text or "untitled"


def _build_filename(comment: str, timestamp: str) -> str:
    """生成短文件名，避免 Windows 路径和文件名长度限制导致的隐患。"""
    safe_comment = _sanitize_comment(comment)
    base_name = f"{timestamp}-{safe_comment}"
    if len(base_name) <= 180:
        return f"{base_name}.png"

    digest = hashlib.md5(safe_comment.encode("utf-8", errors="ignore")).hexdigest()[:8]
    shortened = safe_comment[:80].rstrip("_. ") or "untitled"
    filename = f"{timestamp}-{shortened}-{digest}.png"
    if len(filename) > 255:
        return f"{timestamp}-{digest}.png"
    return filename


def _save_png_stdlib(file_path: Path, image: np.ndarray) -> None:
    """使用纯 Python 标准库写出 PNG，避免依赖 Pillow。"""
    arr = np.asarray(image)
    # 安全处理浮点型图像数组 (0.0 - 1.0 或已是 0-255 的 float，先截断再转换)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        height, width = arr.shape
        color_type = 0
        rgb_arr = arr
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        height, width, channels = arr.shape
        color_type = 2 if channels == 3 else 6
        # OpenCV 默认是 BGR，PNG 需要 RGB/RGBA 顺序
        rgb_arr = arr[:, :, ::-1] if channels == 3 else arr[:, :, [2, 1, 0, 3]]
    else:
        raise ValueError("图片必须是二维灰度图或三维 RGB/RGBA 数组")

    flat_rows = rgb_arr.reshape(height, -1)
    padded = np.hstack([np.zeros((height, 1), dtype=np.uint8), flat_rows])
    raw = padded.tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(
        b"IHDR",
        struct.pack("!IIBBBBB", width, height, 8, color_type, 0, 0, 0),
    )
    png += chunk(b"IDAT", zlib.compress(raw, level=6)) # 如果需要更快的保存速度，可以降低压缩等级，例如 level=1
    png += chunk(b"IEND", b"")

    with file_path.open("wb") as f:
        f.write(png)


def save_image(image: np.ndarray, comment: str) -> bool:
    # 在文件头添加时间，避免文件名重名，且方便排序
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    # 生成文件名
    filename = _build_filename(comment, timestamp)
    file_path = save_dir / filename

    # 保存图片
    try:
        _save_png_stdlib(file_path, image)
        return True
    except Exception as exc:  # pragma: no cover - 调试日志
        logger.warning(f"保存图片 {filename} 失败: {exc}")
        return False
