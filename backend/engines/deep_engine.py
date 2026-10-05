"""深度换脸引擎：inswapper_128 + ArcFace（w600k_r50），纯 onnxruntime 推理。

这就是当前主流方案（FaceFusion / Roop / Deep-Live-Cam 同款核心链路）：

    1. YuNet 检测人脸 + 5 关键点（复用 classic 引擎的检测器）
    2. ArcFace 标准 5 点模板仿射对齐（112×112 提身份 / 128×128 供生成）
    3. w600k_r50 提取源脸 512 维身份向量（embedding）
    4. inswapper_128 以「对齐后的目标脸 + 源脸身份向量」为输入，
       生成一张新脸 —— 目标的姿态/表情/光照 + 源脸的五官身份
    5. 逆仿射贴回原图，椭圆羽化蒙版融合，可选 LAB 颜色迁移和锐化

与经典几何法的本质区别：几何法是把源脸的像素「贴」过去（表情、光照全带过去），
inswapper 是在特征空间里「重画」一张脸 —— 表情和光照属于目标，五官身份属于源。
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from .base import FaceBox, SwapEngine, SwapOptions, SwapResult
from .classic import ClassicSwapEngine

# ArcFace FFHQ 对齐模板（5 点：右眼、左眼、鼻尖、右嘴角、左嘴角），112 基准
_TMPL_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_EMBED_SIZE = 112
_SWAP_SIZE = 128
# 人脸修复网络（CodeFormer）的工作分辨率
_RESTORE_SIZE = 512

_model_cache: dict = {}
_model_error: Optional[str] = None


def _model_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for cand in (os.path.join(root, "models"), os.path.join(os.getcwd(), "models")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(root, "models")


def _get_sessions() -> Optional[dict]:
    """懒加载并缓存 onnxruntime 会话；缺依赖或模型时返回 None。"""
    global _model_error
    if _model_cache:
        return _model_cache
    if _model_error:
        return None

    try:
        import onnxruntime as ort
    except ImportError as exc:
        _model_error = f"缺少 onnxruntime（pip install onnxruntime）: {exc}"
        return None

    mdir = _model_dir()
    swap_path = os.path.join(mdir, "inswapper_128.onnx")
    embed_path = os.path.join(mdir, "w600k_r50.onnx")

    if not os.path.exists(swap_path):
        _model_error = f"缺少模型 {swap_path}（下载方式见 README）"
        return None
    if not os.path.exists(embed_path):
        _model_error = f"缺少模型 {embed_path}（下载方式见 README）"
        return None

    try:
        # CPU 就足够：inswapper_128 单次推理 1~3 秒
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        swap_sess = ort.InferenceSession(swap_path, opts, providers=["CPUExecutionProvider"])
        embed_sess = ort.InferenceSession(embed_path, opts, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        _model_error = f"模型加载失败: {exc}"
        return None

    # emap：inswapper 内置的身份投影矩阵（模型 initializer 的最后一项）。
    # 官方 INSwapper 的约定：latent = normalize(normed_embedding @ emap)。
    # 跳过这一步模型会把它当成无效身份，输出一张与任何人都无关的「路人脸」。
    emap_path = os.path.join(mdir, "emap.npy")
    try:
        if os.path.exists(emap_path):
            emap = np.load(emap_path)
        else:
            import onnx
            from onnx import numpy_helper
            model = onnx.load(swap_path)
            emap = numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)
            np.save(emap_path, emap)
        if emap.shape != (512, 512):
            raise ValueError(f"emap 形状异常: {emap.shape}")
    except Exception as exc:  # noqa: BLE001
        _model_error = f"读取身份映射矩阵 emap 失败: {exc}"
        return None

    _model_cache["swap"] = swap_sess
    _model_cache["embed"] = embed_sess
    _model_cache["emap"] = emap.astype(np.float32)
    return _model_cache


def _get_restore_session():
    """懒加载 CodeFormer 人脸修复网络；模型缺失或加载失败返回 None。

    返回 None 是正常情况（模型是可选的），调用方直接跳过修复即可，
    不要把这个当成引擎不可用的理由。
    """
    if "restore" in _model_cache:
        return _model_cache["restore"]
    mdir = _model_dir()
    path = os.path.join(mdir, "codeformer.onnx")
    if not os.path.exists(path):
        return None
    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    except Exception:  # noqa: BLE001
        return None
    _model_cache["restore"] = sess
    return sess


def _landmark_abs(face: FaceBox) -> tuple[np.ndarray, np.ndarray]:
    """把 FaceBox 的归一化关键点还原成绝对像素坐标。

    返回 (关键点, 对应模板)。优先完整 5 点 [右眼(画面左), 左眼(画面右),
    鼻尖, 右嘴角, 左嘴角]，与 ArcFace 模板一一对应；Haar 回退路径没有
    嘴角时用 3 点（双眼 + 嘴心 -> 模板嘴角中点）——绝不能把同一个嘴心
    重复映射到两个嘴角位置，那会把相似变换估计彻底扭曲。
    """
    lm = face.landmarks

    def _abs(n: str, default: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
        rx, ry = lm.get(n, default)
        return np.array([face.x + rx * face.w, face.y + ry * face.h], dtype=np.float32)

    eye_l, eye_r = _abs("left_eye"), _abs("right_eye")
    if "mouth_left" in lm and "mouth_right" in lm:
        kps = np.stack(
            [eye_l, eye_r, _abs("nose"), _abs("mouth_left"), _abs("mouth_right")]
        )
        return kps, _TMPL_112.copy()
    # 3 点回退
    tmpl = _TMPL_112
    center = np.array(
        [(tmpl[3][0] + tmpl[4][0]) / 2, (tmpl[3][1] + tmpl[4][1]) / 2],
        dtype=np.float32,
    )
    kps = np.stack([eye_l, eye_r, _abs("mouth")])
    return kps, np.stack([tmpl[0], tmpl[1], center])


def _align_matrix(kps: np.ndarray, tmpl: np.ndarray, size: int) -> np.ndarray:
    """估计「原图 -> size×size 对齐空间」的相似变换。"""
    target = tmpl * (size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(kps, target, method=cv2.LMEDS)
    if matrix is None:
        matrix, _ = cv2.estimateAffinePartial2D(kps, target, method=0)
    return matrix


class DeepSwapEngine(SwapEngine):
    name = "deep"
    display_name = "深度换脸（inswapper，主流方案）"
    requires_model = True

    def __init__(self) -> None:
        self._detector = ClassicSwapEngine()

    def is_available(self) -> tuple[bool, str]:
        ok, reason = self._detector.is_available()
        if not ok:
            return False, f"检测器不可用: {reason}"
        if _get_sessions() is None:
            return False, _model_error or "模型未加载"
        return True, "ok"

    # ---------------------------------------------------------------- 检测

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        return self._detector.detect(image)

    # ---------------------------------------------------------------- 换脸

    def swap(
        self,
        source: np.ndarray,
        target: np.ndarray,
        options: SwapOptions,
    ) -> SwapResult:
        sessions = _get_sessions()
        if sessions is None:
            raise RuntimeError(_model_error or "模型未加载")

        src_faces = self.detect(source)
        dst_faces = self.detect(target)
        if not src_faces:
            raise ValueError("源图里没有检测到人脸，换一张正脸照试试")
        if not dst_faces:
            raise ValueError("目标图里没有检测到人脸")

        si = min(max(options.source_face_index, 0), len(src_faces) - 1)
        di = min(max(options.target_face_index, 0), len(dst_faces) - 1)
        src_face, dst_face = src_faces[si], dst_faces[di]

        # ---- 1) 源脸身份向量（必须经 emap 投影，见 _get_sessions 注释）
        normed = self._embed(source, src_face, sessions["embed"])
        emap = sessions["emap"]
        latent = normed[None] @ emap
        latent /= (np.linalg.norm(latent) + 1e-6)
        latent = latent.astype(np.float32)

        # ---- 2) 目标脸对齐到 128×128 喂给 inswapper
        dst_kps, dst_tmpl = _landmark_abs(dst_face)
        matrix = _align_matrix(dst_kps, dst_tmpl, _SWAP_SIZE)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("无法对齐目标脸（关键点异常）")

        h, w = target.shape[:2]
        aligned128 = cv2.warpAffine(
            target, matrix, (_SWAP_SIZE, _SWAP_SIZE),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101,
        )

        # ---- 3) inswapper 生成新脸（固定 128×128）
        raw = self._swap_face(aligned128, latent, sessions["swap"])

        # ---- 4) 人脸修复：把 128 的脸放大到 512 交给 CodeFormer 重建细节。
        #         这是"皮肤塑料感"的关键一步 —— inswapper 只有 128px，
        #         直接贴回大图必然糊；CodeFormer 能把睫毛/皮肤纹理/牙齿
        #         重新长出来。模型缺失时静默跳过（降级为无修复）。
        work_size = _SWAP_SIZE
        if options.face_restore:
            up512 = cv2.resize(
                raw, (_RESTORE_SIZE, _RESTORE_SIZE), interpolation=cv2.INTER_CUBIC
            )
            restored = self._restore_face(up512, options.restore_weight)
            if restored is not None:
                work_size = _RESTORE_SIZE
                raw = restored

        # 蒙版 / 颜色迁移 / 细节回填全部在最终工作分辨率上做，保证生效范围一致。
        # 注意 matrix 是「原图 -> 128 对齐空间」的，工作分辨率变了必须整体
        # 缩放仿射矩阵（线性部分和平移量一起乘 work_size/128），否则贴回去
        # 的位置和大小全错 —— 脸会以 1/4 尺寸甩到图角落上。
        work_matrix = (
            matrix * (work_size / _SWAP_SIZE) if work_size != _SWAP_SIZE else matrix
        )
        aligned = aligned128 if work_size == _SWAP_SIZE else cv2.warpAffine(
            target, work_matrix, (work_size, work_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101,
        )
        mask128 = self._aligned_mask(work_size, options)

        # ---- 5) 颜色迁移：修复后的脸色调会偏 CodeFormer 自己的风格，要拉回目标
        if options.color_match:
            raw = self._color_transfer(raw, aligned, mask128)

        # ---- 6) 细节回填：把目标原图的局部明暗调制（碎发、睫毛、皮肤纹理）
        #         乘回生成脸，同时压掉 CodeFormer 自己"发明"出来的纹理方向
        raw = self._detail_transfer(raw, aligned, mask128)

        # ---- 7) 逆仿射贴回原图（用工作分辨率对应的矩阵求逆）
        inv = cv2.invertAffineTransform(work_matrix)
        face_full = cv2.warpAffine(
            raw, inv, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        # 蒙版在对齐空间画好同样逆变换回去，天然与目标脸对齐
        mask_full = cv2.warpAffine(
            mask128, inv, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )

        alpha = (mask_full.astype(np.float32) / 255.0)[..., None]
        s = float(np.clip(options.strength, 0, 100)) / 100.0
        alpha = alpha * s
        out = (
            face_full.astype(np.float32) * alpha
            + target.astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)

        # 只在换脸区域内锐化：inswapper_128 的输出天生偏软，而背景本来就
        # 是清晰的，全图锐化会让背景和脸出现清晰度断层。
        # 已经过 CodeFormer 修复时只轻锐（0.15），重锐会起白边。
        amount = 0.65 if options.enhance else (0.15 if work_size != _SWAP_SIZE else 0.30)
        sharp = self._unsharp(out, amount)
        out = (
            sharp.astype(np.float32) * alpha + out.astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)

        return SwapResult(
            image=out,
            engine=self.name,
            meta={
                "source_face": src_face.to_dict(),
                "target_face": dst_face.to_dict(),
                "source_faces": len(src_faces),
                "target_faces": len(dst_faces),
                "color_match": options.color_match,
                "face_restore": work_size != _SWAP_SIZE,
                "restore_weight": float(np.clip(options.restore_weight, 0.0, 1.0)),
                "work_size": work_size,
                "blend_mode": "affine_feather (deep)",
            },
        )

    # ------------------------------------------------------------ 内部工具

    @staticmethod
    def _embed(image: np.ndarray, face: FaceBox, session) -> np.ndarray:
        kps, tmpl = _landmark_abs(face)
        matrix = _align_matrix(kps, tmpl, _EMBED_SIZE)
        if matrix is None:
            raise ValueError("无法对齐源脸（关键点异常）")
        aligned = cv2.warpAffine(
            image, matrix, (_EMBED_SIZE, _EMBED_SIZE),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101,
        )
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (_EMBED_SIZE, _EMBED_SIZE),
            (127.5, 127.5, 127.5), swapRB=True,
        )
        inp = session.get_inputs()[0]
        out = session.run(None, {inp.name: blob})[0]
        vec = np.asarray(out).flatten().astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-6)

    @staticmethod
    def _swap_face(aligned: np.ndarray, latent: np.ndarray, session) -> np.ndarray:
        """inswapper 推理。

        本项目的 inswapper_128.onnx（官方原版，sha256 e4a3f08c...）的接口约定：
        - 输入 target = RGB [0,1]（x/255，swapRB）；source = normalize(normed_embedding @ emap)
        - 输出 pred = (1,3,128,128) RGB ∈[0,1]
        """
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 255.0, (_SWAP_SIZE, _SWAP_SIZE),
            (0, 0, 0), swapRB=True,
        )
        inputs = {}
        for i in session.get_inputs():
            # 官方输入顺序：target(图像) 在前、source(latent) 在后
            if i.shape and len(i.shape) == 4:
                inputs[i.name] = blob
            else:
                inputs[i.name] = latent.astype(np.float32)
        out = session.run(None, inputs)[0]  # (1,3,H,W) RGB ∈[0,1]
        face = out[0].transpose(1, 2, 0)
        face = np.clip(face * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]
        return face

    @staticmethod
    def _restore_face(face_bgr: np.ndarray, weight: float = 0.7) -> Optional[np.ndarray]:
        """CodeFormer 人脸修复。

        本项目的 codeformer.onnx（FaceFusion 官方转换版，360MB）接口实测：
        - 输入 input = (1,3,512,512) RGB，归一化到 [-1,1]（(x/255-0.5)/0.5）
        - 输入 weight = double 标量（0-d ndarray），保真度权重 0~1
        - 输出 output = (1,3,512,512)，区间实测 [-1,1]（min≈-1, max≈1），
          用 y*0.5+0.5 解码
        - 只请求 "output" 这一个输出，跳过 logits/features 省一半时间

        weight 越小画质越猛但越容易"换头"（长出别人的脸），越大越忠实于输入。
        换脸场景 0.6~0.8 比较稳：够把细节长出来，又不至于改写身份。
        """
        sess = _get_restore_session()
        if sess is None:
            return None
        try:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
            feed = {}
            for item in sess.get_inputs():
                if len(item.shape) == 4:
                    feed[item.name] = blob
                else:
                    feed[item.name] = np.array(float(np.clip(weight, 0.0, 1.0)), dtype=np.float64)
            out = sess.run(["output"], feed)[0][0]
            if out.shape[0] == 3:  # CHW -> HWC
                out = out.transpose(1, 2, 0)
            out = np.clip((out * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _aligned_mask(size: int, options: SwapOptions) -> np.ndarray:
        """在对齐空间（size×size）里画椭圆蒙版并羽化。

        这里可以用「固定椭圆」，因为 ArcFace 对齐会把人脸摆到模板的固定
        位置：关键点永远落在同一个坐标上，倾斜也已经被仿射校正掉了。
        椭圆参数（以 size 为单位）是照着实测标定的：
            眼睛 y≈0.46、鼻尖 y≈0.64、嘴巴 y≈0.82、发际 y≈0.25、下巴 y≈0.93
        于是人脸外接椭圆 ≈ 圆心 (0.50, 0.585)、半轴 (0.305, 0.385)。
        老版本把圆心算在"五官质心与嘴心的中点"(y≈0.75)，椭圆整体下坠
        ~0.15*size，额头完全露在蒙版外，融合接缝正好横在发际线上 —— 这就
        是之前看起来像"贴上去的"的主要原因之一。
        """
        scale = float(np.clip(options.mask_scale, 0.6, 2.0)) / 1.28
        cx = size * 0.500
        cy = size * 0.585
        rx = size * 0.305 * scale
        ry = size * 0.385 * scale

        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (int(round(cx)), int(round(cy))),
            (max(int(round(rx)), 4), max(int(round(ry)), 4)),
            0, 0, 360, 255, -1,
        )
        # 羽化半径跟着椭圆尺寸走，保证过渡带相对人脸宽度是恒定比例
        sigma = max(1.2, (max(0, options.feather) / 100.0) * min(rx, ry) * 0.42)
        ksize = max(3, int(round(sigma * 4)) | 1)
        return cv2.GaussianBlur(mask, (ksize, ksize), sigma)

    @staticmethod
    def _color_transfer(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """LAB 均值/方差迁移（统计量只取蒙版内的像素）。

        注意权重很轻，而且是**严格的插值**而不是直线拼接：
            out = (1-w)*src + w*((src-mean_s)*k + mean_r)
        w=0 是原样、w=1 是完全匹配，中间落在线上，不会像旧写法那样
        「明明只想微调却把 a 通道从 134 压到 129」——实测 inswapper 的
        输出本来就已经继承了目标图的色调（L/a/b 均值误差 <2），
        所以这里只需要轻轻推一把 + 压掉过饱和的唇色，不该大改。
        """
        m = mask > 127
        if m.sum() < 50:
            return src
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
        src_px, ref_px = src_lab[m], ref_lab[m]
        src_mean, src_std = src_px.mean(0), src_px.std(0) + 1e-6
        ref_mean, ref_std = ref_px.mean(0), ref_px.std(0) + 1e-6

        # 亮度只做很轻的均值/对比度对齐；色彩通道不动方差（保留原本的
        # 血色），过饱和的部分交给下面的软限幅处理
        w_mean = np.array([0.35, 0.50, 0.50], dtype=np.float32)
        w_std = np.array([0.35, 0.00, 0.00], dtype=np.float32)
        gain = 1.0 + (ref_std / src_std - 1.0) * w_std
        matched = (src_px - src_mean) * gain + src_mean + (ref_mean - src_mean) * w_mean

        out = src_lab.copy()
        out[m] = src_px * (1.0 - w_mean) + matched * w_mean

        # 结构性颜色修复：过饱和红唇这类「局部超出目标色域」的颜色，
        # 全局均值迁移压不住 —— 用软限幅把 a/b 通道收回目标统计范围
        lim_a = max(ref_std[1] * 2.5, 1.0)
        lim_b = max(ref_std[2] * 2.5, 1.0)
        a, b = out[..., 1], out[..., 2]
        out[..., 1] = ref_mean[1] + np.tanh((a - ref_mean[1]) / lim_a) * lim_a
        out[..., 2] = ref_mean[2] + np.tanh((b - ref_mean[2]) / lim_b) * lim_b

        return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _feature_protect(size: int) -> np.ndarray:
        """对齐空间里的「五官保护区」（1=可回填，越小越保守）。

        眼睛和嘴巴是唯一"位置会被对齐到同一处、但形状因人而异"的区域：
        把目标的牙齿缝、睫毛、眼睑褶皱乘到生成脸上，只要两者形状稍有
        出入就会糊成一团灰黑色 —— 实测嘴唇内侧最明显。所以这两块把
        回填强度压到 25%，脸颊、额头、发丝区域全量回填。
        """
        prot = np.ones((size, size), dtype=np.float32)
        tmpl = _TMPL_112 * (size / 112.0)
        r_eye = max(3, int(0.105 * size))
        for eye in (tmpl[0], tmpl[1]):
            cv2.circle(prot, (int(eye[0]), int(eye[1])), r_eye, 0.25, -1)
        mouth = ((tmpl[3] + tmpl[4]) / 2).astype(np.float32)
        cv2.ellipse(
            prot, (int(mouth[0]), int(mouth[1])),
            (max(3, int(0.17 * size)), max(3, int(0.10 * size))),
            0, 0, 360, 0.25, -1,
        )
        return cv2.GaussianBlur(prot, (0, 0), max(1.0, 0.035 * size))

    @staticmethod
    def _detail_transfer(
        src: np.ndarray, ref: np.ndarray, mask: np.ndarray, strength: float = 0.85
    ) -> np.ndarray:
        """把参考图（目标原图的对齐版）的局部明暗调制乘回生成脸。

        调制因子 k = ref / gaussian(ref)：
        - 碎发、睫毛、皮肤纹理处 ref 比周围暗 -> k<1 -> 生成脸同步变暗，
          原本被 inswapper 抹掉的发丝重新浮现；
        - 由于上界被卡在 1.0，k 只会 ≤1，不存在把噪点放大成颗粒的问题。

        k 被限制在 [0.70, 1.0]，并且眼睛/嘴唇区域按 _feature_protect 降权，
        所以即便目标五官轮廓和生成脸有偏差，最多只留下一点淡阴影，
        不会长出「第二副眉毛」或在嘴唇内侧糊出灰块。
        调制量再按蒙版羽化值渐变，边界处自然过渡回 1.0，不会产生硬边。

        上界卡在 1.0（只压暗、不提亮）是实测出来的：inswapper 的输出本来
        就偏亮，允许 k>1 会把目标脸上的高光/腮红再叠一层上去，
        实测脸颊 L 从 164 被抬到 172（目标自己才 167），整张脸就"发白"。
        而真正需要回填的碎发、睫毛、鼻影全都是变暗方向，砍掉提亮不亏。
        """
        if ref.shape[:2] != src.shape[:2]:
            return src
        size = ref.shape[0]
        gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sigma = max(1.0, size / 22.0)
        base = cv2.GaussianBlur(gray, (0, 0), sigma)
        # +6 的偏置防止暗部（头发）除法炸开
        k = (gray + 6.0) / (base + 6.0)
        k = np.minimum(np.clip(k, 0.70, 1.15), 1.0)
        weight = (mask.astype(np.float32) / 255.0) * DeepSwapEngine._feature_protect(size)
        k = 1.0 + (k - 1.0) * weight * float(np.clip(strength, 0, 1))
        out = src.astype(np.float32) * k[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _unsharp(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
        blur = cv2.GaussianBlur(img, (0, 0), 2.0)
        return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
