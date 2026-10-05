"""换脸引擎抽象层。

所有引擎都实现同一个接口，前端/服务端只依赖这个接口，
这样以后从「经典几何法」升级到「深度学习法」时不用改动上层代码。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class SwapOptions:
    """一次换脸请求的全部可调参数。"""

    # 目标图里要替换哪张脸（按面积从大到小排序，0 = 最大的那张脸）
    target_face_index: int = 0
    # 源图里用哪张脸当素材
    source_face_index: int = 0
    # 蒙版羽化强度 0-100，越大边缘越柔和
    feather: int = 45
    # 蒙版缩放系数，>1 会多包一点下巴/发际线
    mask_scale: float = 1.28
    # 是否做 LAB 色彩迁移（把源脸肤色/亮度对齐到目标图，很关键）
    color_match: bool = True
    # 是否用人脸修复网络（CodeFormer）重建脸部细节，需要 models/codeformer.onnx
    face_restore: bool = True
    # 修复的保真度权重 0-1：0=画质最好但更容易偏离原脸，1=最忠实于输入
    restore_weight: float = 0.7
    # 融合方式: seamless = 泊松融合(更自然) / feather = 直接羽化贴图
    blend_mode: str = "seamless"
    # 最终结果与原因的不透明度，0-100
    strength: int = 100
    # 是否做锐化增强
    enhance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FaceBox:
    """一张检出人脸的几何信息。"""

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    # 关键点（归一化到人脸框内的比例坐标），缺失时为 None
    landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["area"] = self.area
        return data


@dataclass
class SwapResult:
    """一次换脸的产出。"""

    image: np.ndarray  # BGR
    engine: str
    # 记录了实际用了哪张脸、哪些关键点，方便前端可视化 / 调试
    meta: dict[str, Any] = field(default_factory=dict)


class SwapEngine(abc.ABC):
    """换脸引擎基类。"""

    name: str = "base"
    display_name: str = "未命名引擎"
    # 是否需要额外下载模型权重
    requires_model: bool = False

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """检测图中所有人脸。"""

    @abc.abstractmethod
    def swap(
        self,
        source: np.ndarray,
        target: np.ndarray,
        options: SwapOptions,
    ) -> SwapResult:
        """把 source 的脸换到 target 上。"""

    def is_available(self) -> tuple[bool, str]:
        """引擎是否可用，返回 (是否可用, 原因说明)。"""
        return True, "ok"
