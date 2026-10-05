"""深度学习换脸引擎（可选，接入 InsightFace / inswapper）。

这是目前开源界最成熟、效果最稳的换脸方案，工程上等价于：
    input -> RetinaFace/SCRFD 检测 -> ArcFace 提特征 -> inswapper_128 生成 -> 贴回

启用步骤（约 500MB 下载，CPU 也能跑，GPU 快 10 倍以上）：
    pip install insightface onnxruntime        # GPU 用 onnxruntime-gpu
    # 下载 inswapper_128.onnx 放到 models/ 目录（见 README 的获取说明）

然后用环境变量指向模型：
    set INSWAPPER_MODEL=C:\\path\\to\\models\\inswapper_128.onnx

没有模型时这个引擎会自动报告「不可用」，服务端会退回经典引擎，
所以整个项目在空环境下依然可以启动和演示。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .base import FaceBox, SwapEngine, SwapOptions, SwapResult

_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "inswapper_128.onnx"


class InsightFaceSwapEngine(SwapEngine):
    name = "insightface"
    display_name = "InsightFace 深度学习（需模型）"
    requires_model = True

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or os.environ.get("INSWAPPER_MODEL", _DEFAULT_MODEL))
        self._app = None
        self._swapper = None
        self._error: str | None = None

    # ------------------------------------------------------------ 生命周期

    def _lazy_init(self) -> None:
        if self._app is not None or self._error is not None:
            return
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            self._error = "未安装 insightface，请执行 pip install insightface onnxruntime"
            return

        if not self.model_path.exists():
            self._error = f"未找到模型文件: {self.model_path}"
            return

        try:
            self._app = FaceAnalysis(
                name="buffalo_l",
                # 检测/识别都在 CPU 上跑；有 GPU 就把 providers 换成 CUDAExecutionProvider
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            self._swapper = insightface.model_zoo.get_model(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # 模型损坏 / 依赖冲突等
            self._error = f"引擎初始化失败: {exc}"

    def is_available(self) -> tuple[bool, str]:
        self._lazy_init()
        if self._error:
            return False, self._error
        return True, "ok"

    # ---------------------------------------------------------------- 检测

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        self._lazy_init()
        if self._app is None:
            raise RuntimeError(self._error or "引擎不可用")

        faces = self._app.get(image)
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        boxes: list[FaceBox] = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            lm: dict[str, tuple[float, float]] = {}
            if getattr(f, "kps", None) is not None:
                kps = np.asarray(f.kps, dtype=np.float32)
                names = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
                for i, n in enumerate(names):
                    if i < len(kps):
                        lm[n] = ((kps[i][0] - x1) / w, (kps[i][1] - y1) / h)
            boxes.append(
                FaceBox(x=x1, y=y1, w=w, h=h, score=float(getattr(f, "det_score", 1.0)), landmarks=lm)
            )
        return boxes

    # ---------------------------------------------------------------- 换脸

    def swap(
        self,
        source: np.ndarray,
        target: np.ndarray,
        options: SwapOptions,
    ) -> SwapResult:
        self._lazy_init()
        if self._app is None or self._swapper is None:
            raise RuntimeError(self._error or "引擎不可用")

        src_faces = self._app.get(source)
        dst_faces = self._app.get(target)
        if not src_faces:
            raise ValueError("源图里没有检测到人脸")
        if not dst_faces:
            raise ValueError("目标图里没有检测到人脸")

        src_faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        dst_faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        si = min(max(options.source_face_index, 0), len(src_faces) - 1)
        src_face = src_faces[si]

        result = target.copy()
        for idx, dst_face in enumerate(dst_faces):
            if options.target_face_index >= 0 and idx != options.target_face_index:
                continue
            result = self._swapper.get(result, dst_face, src_face, paste_back=True)

        # 可选：统一做一次锐化，弥补 128 分辨率贴回后的细节损失
        if options.enhance:
            blur = cv2.GaussianBlur(result, (0, 0), 2.0)
            result = cv2.addWeighted(result, 1.6, blur, -0.6, 0)

        if options.strength < 100:
            s = np.clip(options.strength, 0, 100) / 100.0
            result = cv2.addWeighted(result, s, target, 1 - s, 0)

        return SwapResult(
            image=result,
            engine=self.name,
            meta={
                "model": str(self.model_path),
                "source_faces": len(src_faces),
                "target_faces": len(dst_faces),
                "swapped_faces": 1 if options.target_face_index >= 0 else len(dst_faces),
            },
        )
