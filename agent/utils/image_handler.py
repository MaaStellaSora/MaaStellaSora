"""
调试用的agent截图保存功能，会把截图以png格式保存到项目目录的 debug/agent_image 目录下
由于打包后的便携版python没有安装PIL模块，请不要在打包后的环境中使用该功能

使用方法：
    from utils.image_handler import save_image

    save_image(image, "说明信息")
"""

from pathlib import Path
from datetime import datetime

import numpy as np
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False


save_dir = Path(__file__).resolve().parents[2] / "debug" / "agent_image"
save_dir.mkdir(parents=True, exist_ok=True)


def save_image(image: np.ndarray, comment: str) -> bool:
    if not HAS_PIL:
        print("PIL模块未安装，无法保存图片")
        return False

    # 生成文件名
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # %f是微秒，取前3位得到毫秒
    filename = f"{timestamp}-{comment}.png"
    file_path = save_dir / filename

    # 处理截图
    # 由于OpenCV截图默认是BGR格式，需要转换为RGB格式
    img = Image.fromarray(image[:, :, ::-1], mode='RGB')

    # 保存截图
    img.save(file_path)
    return True
