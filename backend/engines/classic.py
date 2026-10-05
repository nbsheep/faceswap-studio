"""经典几何法换脸引擎（零模型下载，开箱即跑）。

流程（这就是所有换脸软件的核心链路，省略了「深度学习生成」这一步）：

    1. 检测人脸：优先 YuNet（models/ 里有 onnx 就用，5 个精准关键点），
       否则退回 Haar 级联 + 眼嘴检测
    2. 用双眼(+鼻尖+嘴)估计相似变换（旋转+缩放+平移）对齐姿态
    3. 按眼距自适应生成旋转椭圆蒙版 + 高斯羽化
    4. LAB 色彩迁移，把源脸肤色 / 曝光对齐到目标图
    5. 泊松融合（seamlessClone）把脸贴进去，消除接缝
    6. 可选锐化

它的上限很明显：不做表情迁移、不做三维重建、大角度侧脸会崩。
但它能在 1 秒内跑完、无需 GPU，非常适合验证产品链路和 UI。
真正的生产环境请换成 InsightFaceEngine（见 insightface_engine.py）。
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from .base import FaceBox, SwapEngine, SwapOptions, SwapResult

# 找不到眼睛时按人脸框比例估计的位置（左眼、右眼、嘴中心）
_FALLBACK_RATIO = {
    "left_eye": (0.32, 0.38),
    "right_eye": (0.68, 0.38),
    "mouth": (0.50, 0.78),
}

# 级联分类器的文件名。它们随 opencv-python 一起安装，不需要额外下载。
_CASCADE_FILES = {
    "face": "haarcascade_frontalface_default.xml",
    "eye": "haarcascade_eye.xml",
    "eye_glasses": "haarcascade_eye_tree_eyeglasses.xml",
    "mouth": "haarcascade_smile.xml",
}

# YuNet 轻量人脸检测模型（约 230KB），提供 5 个精准关键点。
# 下载地址见 README；找不到时自动回退到 Haar，不影响可用性。
_YUNET_FILENAME = "yunet_2023mar.onnx"

_cascades: Dict[str, Any] | None = None
_cascade_error: str | None = None
_yunet: Any | None = None
_yunet_checked: bool = False


def _load_yunet() -> Any | None:
    """懒加载 YuNet 检测器；找不到模型文件时返回 None（回退 Haar）。"""
    global _yunet, _yunet_checked
    if _yunet_checked:
        return _yunet
    _yunet_checked = True

    if not hasattr(cv2, "FaceDetectorYN"):
        return None

    import os

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "models", _YUNET_FILENAME),
        os.path.join(here, "..", "models", _YUNET_FILENAME),
        os.path.join(os.getcwd(), "models", _YUNET_FILENAME),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None

    try:
        _yunet = cv2.FaceDetectorYN.create(
            path, "", (320, 320), score_threshold=0.6, nms_threshold=0.3
        )
    except Exception:  # noqa: BLE001
        _yunet = None
    return _yunet


def _load_cascades() -> Dict[str, Any]:
    """懒加载级联分类器。

    延迟到第一次真正检测时才加载，这样即使环境的 OpenCV 不支持级联检测，
    整个服务也能正常启动 —— 只是这个引擎在 /api/engines 里显示为不可用。

    注意：OpenCV 5.x 已经彻底移除了 CascadeClassifier，所以版本下限必须卡住。
    """
    global _cascades, _cascade_error

    if _cascades is not None:
        return _cascades
    if _cascade_error is not None:
        raise RuntimeError(_cascade_error)

    if not hasattr(cv2, "CascadeClassifier"):
        _cascade_error = (
            f"当前 OpenCV {cv2.__version__} 不支持级联人脸检测"
            "（5.x 起已移除 CascadeClassifier）。"
            '请安装 4.x：pip install "opencv-python-headless>=4.8,<5"'
        )
        raise RuntimeError(_cascade_error)

    try:
        import os

        cascade_dir = cv2.data.haarcascades
        loaded: Dict[str, Any] = {}
        for key, filename in _CASCADE_FILES.items():
            path = os.path.join(cascade_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到级联文件 {filename}")
            clf = cv2.CascadeClassifier(path)
            if clf.empty():
                raise RuntimeError(f"级联文件加载为空 {filename}")
            loaded[key] = clf
    except Exception as exc:  # noqa: BLE001
        _cascade_error = f"级联分类器加载失败: {exc}"
        raise RuntimeError(_cascade_error) from exc

    _cascades = loaded
    return _cascades


def _largest_first(boxes: list[FaceBox]) -> list[FaceBox]:
    return sorted(boxes, key=lambda b: b.area, reverse=True)


class ClassicSwapEngine(SwapEngine):
    name = "classic"
    display_name = "经典几何法（YuNet 关键点 / Haar 回退）"
    requires_model = False

    def is_available(self) -> tuple[bool, str]:
        try:
            _load_cascades()
        except RuntimeError as exc:
            return False, str(exc)
        return True, "ok"

    # ---------------------------------------------------------------- 检测

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        # 优先用 YuNet（关键点准得多），没装模型再退回 Haar
        yunet = _load_yunet()
        if yunet is not None:
            faces = self._detect_yunet(image, yunet)
            if faces:
                return faces
        return self._detect_haar(image)

    def _detect_yunet(self, image: np.ndarray, yunet: Any) -> list[FaceBox]:
        h, w = image.shape[:2]
        yunet.setInputSize((w, h))
        try:
            _, faces = yunet.detect(image)
        except cv2.error:
            return []
        if faces is None or not len(faces):
            return []

        out: list[FaceBox] = []
        for f in faces:
            x, y, fw, fh = f[:4]
            # YuNet 关键点顺序：右眼(画面左)、左眼(画面右)、鼻尖、右嘴角(画面左)、左嘴角(画面右)
            re_pt, le_pt = f[4:6], f[6:8]
            nose = f[8:10]
            mre, mle = f[10:12], f[12:14]
            mouth = (mre + mle) / 2.0

            def _norm(pt: np.ndarray) -> tuple[float, float]:
                return (
                    float((pt[0] - x) / max(fw, 1)),
                    float((pt[1] - y) / max(fh, 1)),
                )

            out.append(
                FaceBox(
                    x=int(x),
                    y=int(y),
                    w=int(fw),
                    h=int(fh),
                    landmarks={
                        "left_eye": _norm(re_pt),
                        "right_eye": _norm(le_pt),
                        "nose": _norm(nose),
                        "mouth": _norm(mouth),
                        "mouth_left": _norm(mre),
                        "mouth_right": _norm(mle),
                        "_eyes_detected": True,  # type: ignore[dict-item]
                        "_mouth_detected": True,  # type: ignore[dict-item]
                        "_yunet": True,  # type: ignore[dict-item]
                    },
                )
            )
        return _largest_first(out)

    def _detect_haar(self, image: np.ndarray) -> list[FaceBox]:
        cascades = _load_cascades()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = cascades["face"].detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=6,
            minSize=(60, 60),
        )

        faces: list[FaceBox] = []
        for (x, y, w, h) in rects:
            landmarks = self._detect_landmarks(gray, x, y, w, h)
            faces.append(
                FaceBox(x=int(x), y=int(y), w=int(w), h=int(h), landmarks=landmarks)
            )

        # 过滤掉被更大脸包含的重复框
        faces = _largest_first(faces)
        kept: list[FaceBox] = []
        for f in faces:
            if any(
                f.x >= k.x and f.y >= k.y
                and f.x + f.w <= k.x + k.w
                and f.y + f.h <= k.y + k.h
                for k in kept
            ):
                continue
            kept.append(f)
        return kept

    def _detect_landmarks(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> dict[str, tuple[float, float]]:
        """在脸框内找双眼和嘴，返回归一化到脸框的比例坐标。"""
        cascades = _load_cascades()
        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            return self._fallback_landmarks()

        # ---- 眼睛：只在上半脸找，避免把鼻孔当眼睛
        upper = roi[: int(h * 0.62), :]
        eyes: list[tuple[int, int, int, int]] = []
        for cascade in (cascades["eye"], cascades["eye_glasses"]):
            found = cascade.detectMultiScale(
                upper, scaleFactor=1.1, minNeighbors=8, minSize=(max(8, w // 12),) * 2
            )
            eyes.extend(found.tolist() if len(found) else [])
        # 去重：中心点距离过近的只留一个
        dedup: list[tuple[int, int, int, int]] = []
        for e in sorted(eyes, key=lambda t: t[2] * t[3], reverse=True):
            cx, cy = e[0] + e[2] / 2, e[1] + e[3] / 2
            if all(
                (cx - (d[0] + d[2] / 2)) ** 2 + (cy - (d[1] + d[3] / 2)) ** 2
                > (w * 0.28) ** 2
                for d in dedup
            ):
                dedup.append(e)

        if len(dedup) >= 2:
            # 按 x 排序，左边一簇的质心 = 左眼，右边一簇 = 右眼
            dedup.sort(key=lambda t: t[0])
            half = len(dedup) // 2
            left = dedup[:half]
            right = dedup[half:]

            def _centroid(group: list[tuple[int, int, int, int]]) -> tuple[float, float]:
                cx = sum(g[0] + g[2] / 2 for g in group) / len(group) / w
                cy = sum(g[1] + g[3] / 2 for g in group) / len(group) / h
                return (cx, cy)

            left_eye = _centroid(left)
            right_eye = _centroid(right)
            eyes_ok = True
        elif len(dedup) == 1:
            # 只用一只眼也能算，另一只镜像估计
            cx, cy = dedup[0][0] + dedup[0][2] / 2, dedup[0][1] + dedup[0][3] / 2
            if cx < w / 2:
                left_eye, right_eye = (cx / w, cy / h), _FALLBACK_RATIO["right_eye"]
            else:
                left_eye, right_eye = _FALLBACK_RATIO["left_eye"], (cx / w, cy / h)
            eyes_ok = True
        else:
            left_eye = _FALLBACK_RATIO["left_eye"]
            right_eye = _FALLBACK_RATIO["right_eye"]
            eyes_ok = False

        # ---- 嘴：在下半脸找
        lower = roi[int(h * 0.55) :, :]
        mouths = cascades["mouth"].detectMultiScale(
            lower, scaleFactor=1.15, minNeighbors=18, minSize=(w // 6, h // 10)
        )
        if len(mouths):
            m = max(mouths, key=lambda t: t[2] * t[3])
            mouth = (
                (m[0] + m[2] / 2) / w,
                (m[1] + int(h * 0.55) + m[3] / 2) / h,
            )
            mouth_ok = True
        else:
            mouth = _FALLBACK_RATIO["mouth"]
            mouth_ok = False

        return {
            "left_eye": (float(left_eye[0]), float(left_eye[1])),
            "right_eye": (float(right_eye[0]), float(right_eye[1])),
            "mouth": (float(mouth[0]), float(mouth[1])),
            "_eyes_detected": eyes_ok,  # type: ignore[dict-item]
            "_mouth_detected": mouth_ok,  # type: ignore[dict-item]
        }

    @staticmethod
    def _fallback_landmarks() -> dict[str, tuple[float, float]]:
        return dict(_FALLBACK_RATIO)

    # ---------------------------------------------------------------- 换脸

    def swap(
        self,
        source: np.ndarray,
        target: np.ndarray,
        options: SwapOptions,
    ) -> SwapResult:
        src_faces = self.detect(source)
        dst_faces = self.detect(target)

        if not src_faces:
            raise ValueError("源图里没有检测到人脸，换一张正脸照试试")
        if not dst_faces:
            raise ValueError("目标图里没有检测到人脸")

        si = min(max(options.source_face_index, 0), len(src_faces) - 1)
        di = min(max(options.target_face_index, 0), len(dst_faces) - 1)

        result = self.compose(source, target, src_faces[si], dst_faces[di], options)
        result.meta["source_faces"] = len(src_faces)
        result.meta["target_faces"] = len(dst_faces)
        return result

    # ---------------------------------------------------------------- 换脸

    def compose(
        self,
        source: np.ndarray,
        target: np.ndarray,
        src_face: FaceBox,
        dst_face: FaceBox,
        options: SwapOptions,
    ) -> SwapResult:
        """核心合成逻辑。与检测解耦，方便单独测试和复用。"""

        # 1) 源图三点 -> 目标图三点，估计相似变换
        src_pts = self._points(src_face)
        dst_pts = self._points(dst_face)
        # 先用 LMEDS 抗离群点；只有 3 个点时它偶尔会退化，回退到最小二乘
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if matrix is None:
            matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=0)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("无法对齐两张脸（关键点异常），换张角度更正的照片")

        # 2) 把整张源图按变换搬到目标图坐标系
        h, w = target.shape[:2]
        warped = cv2.warpAffine(
            source,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        # 3) 生成蒙版。有关键点时直接按「目标脸」的关键点画在目标坐标系里，
        #    保证椭圆严丝合缝罩住目标脸上的被替换区域，源图背景不会漏进羽化边；
        #    没有关键点时退回「源图画蒙版再随变换搬运」的旧做法。
        if dst_face.landmarks.get("_yunet"):
            mask_warped = self._build_mask(target.shape[:2], dst_face, options)
        else:
            mask = self._build_mask(source.shape[:2], src_face, options)
            mask_warped = cv2.warpAffine(
                mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
            )

        # 收窄到目标脸附近的区域，加速 + 减少误伤
        x0, y0, x1, y1 = self._roi(dst_face, w, h, pad=0.45)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError("目标人脸太小或贴边，无法处理")
        roi_w, roi_h = x1 - x0, y1 - y0
        warped_roi = warped[y0:y1, x0:x1]
        mask_roi = mask_warped[y0:y1, x0:x1].astype(np.uint8)

        # 4) 色彩迁移：让源脸的肤色和明暗贴近目标图
        if options.color_match:
            warped_roi = self._color_transfer(warped_roi, target[y0:y1, x0:x1], mask_roi)

        # 5) 融合
        if options.blend_mode == "seamless" and mask_roi.sum() > 0:
            center = (roi_w // 2, roi_h // 2)
            blended = cv2.seamlessClone(
                warped_roi, target[y0:y1, x0:x1].copy(), mask_roi, center, cv2.NORMAL_CLONE
            )
        else:
            alpha = (mask_roi.astype(np.float32) / 255.0)[..., None]
            base = target[y0:y1, x0:x1].astype(np.float32)
            blended = (
                warped_roi.astype(np.float32) * alpha + base * (1 - alpha)
            ).astype(np.uint8)

        # 6) 贴回整图，按 strength 控制整体过渡
        out = target.copy()
        region = out[y0:y1, x0:x1].astype(np.float32)
        s = np.clip(options.strength, 0, 100) / 100.0
        out[y0:y1, x0:x1] = (blended.astype(np.float32) * s + region * (1 - s)).astype(
            np.uint8
        )

        if options.enhance:
            out = self._unsharp(out)

        return SwapResult(
            image=out,
            engine=self.name,
            meta={
                "source_face": src_face.to_dict(),
                "target_face": dst_face.to_dict(),
                "color_match": options.color_match,
                "blend_mode": options.blend_mode,
                "roi": [x0, y0, x1, y1],
            },
        )

    # ------------------------------------------------------------ 内部工具

    @staticmethod
    def _points(face: FaceBox) -> np.ndarray:
        """把归一化关键点转成脸框内的绝对像素坐标。

        YuNet 有鼻尖时用 4 点对齐（双眼+鼻+嘴），姿态估计更稳；
        Haar 路径只有 3 点。
        """
        lm = face.landmarks
        names = ["left_eye", "right_eye"]
        if "nose" in lm:
            names.append("nose")
        names.append("mouth")
        pts = []
        for n in names:
            rx, ry = lm.get(n, _FALLBACK_RATIO.get(n, (0.5, 0.5)))
            pts.append([face.x + rx * face.w, face.y + ry * face.h])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _build_mask(
        shape: tuple[int, int], face: FaceBox, options: SwapOptions
    ) -> np.ndarray:
        """画人脸蒙版椭圆，边缘高斯羽化。

        有关键点时（YuNet），椭圆按「眼距」自适应大小、跟随双眼连线角度旋转，
        中心落在眼连线与嘴之间 —— 只盖住真正的脸部区域，不误伤头发和背景。
        没有关键点时退回按检测框画椭圆。
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        lm = face.landmarks

        if lm.get("_yunet"):
            # 绝对像素坐标（由归一化值还原）
            def _abs(n: str) -> tuple[float, float]:
                rx, ry = lm[n]
                return (face.x + rx * face.w, face.y + ry * face.h)

            lex, ley = _abs("left_eye")
            rex, rey = _abs("right_eye")
            mx, my = _abs("mouth")

            eye_mid = ((lex + rex) / 2, (ley + rey) / 2)
            eye_dist = float(np.hypot(lex - rex, ley - rey))
            if eye_dist < 4:
                eye_dist = face.w * 0.42

            # 脸宽 ≈ 2×瞳距；纵轴到下巴即可，盖过下巴会让融合把脖子/衣服颜色吸进来。
            # mask_scale 滑杆的默认值 1.28 是按「检测框椭圆」标定的，
            # 眼距椭圆本身已贴合脸型，这里归一到 1.0 保持滑杆语义不变
            k = options.mask_scale / 1.28
            ax = eye_dist * 1.00 * k
            ay = max(eye_dist * 1.25, np.hypot(mx - eye_mid[0], my - eye_mid[1]) * 1.05)
            ay = ay * k
            cx = eye_mid[0] + (mx - eye_mid[0]) * 0.48
            cy = eye_mid[1] + (my - eye_mid[1]) * 0.48
            angle = float(np.degrees(np.arctan2(ley - rey, lex - rex)))

            cv2.ellipse(
                mask,
                (int(cx), int(cy)),
                (max(int(ax), 4), max(int(ay), 4)),
                angle,
                0,
                360,
                255,
                -1,
            )
        else:
            cx = face.x + face.w / 2
            cy = face.y + face.h * 0.56  # 视觉中心比检测框略低
            ax = int(face.w * 0.5 * options.mask_scale)
            ay = int(face.h * 0.56 * options.mask_scale)
            cv2.ellipse(
                mask, (int(cx), int(cy)), (max(ax, 4), max(ay, 4)), 0, 0, 360, 255, -1
            )

        # 羽化半径：关键点路径按眼距算（脸框含大量头发会虚宽），否则按检测框算
        if lm.get("_yunet"):
            blur = int((max(0, options.feather) / 100.0) * eye_dist * 0.45)
            blur = min(max(blur, 1), 61)
        else:
            blur = int((max(0, options.feather) / 100.0) * max(face.w, face.h) * 0.20)
            blur = min(max(blur, 1), 81)
        blur = blur if blur % 2 == 1 else blur + 1
        if blur > 1:
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        return mask

    @staticmethod
    def _roi(face: FaceBox, w: int, h: int, pad: float = 0.45) -> tuple[int, int, int, int]:
        px, py = int(face.w * pad), int(face.h * pad)
        return (
            max(0, face.x - px),
            max(0, face.y - py),
            min(w, face.x + face.w + px),
            min(h, face.y + face.h + py),
        )

    @staticmethod
    def _color_transfer(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """LAB 空间的均值/方差迁移（Reinhard 方法）。

        这是让换脸「看不出接缝」最关键的一步：不做的话两张照片
        色温/曝光不一致会非常突兀。
        """
        m = mask > 127
        if m.sum() < 50:
            return src

        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_px = src_lab[m]
        src_mean, src_std = src_px.mean(0), src_px.std(0) + 1e-6
        # 参考统计量也取同一批像素位置，保证是可比的同一区域
        ref_px = ref_lab[m]
        ref_mean, ref_std = ref_px.mean(0), ref_px.std(0) + 1e-6

        # 不做 100% 全量对齐，只对齐 80%，避免把源脸特征也洗掉
        weight = 0.8
        scale = (ref_std / src_std) * weight + (1 - weight)
        shift = (ref_mean - src_mean * scale) * weight

        out = src_lab.copy()
        out[m] = src_px * scale + shift
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _unsharp(img: np.ndarray, amount: float = 0.6) -> np.ndarray:
        blur = cv2.GaussianBlur(img, (0, 0), 2.0)
        return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
