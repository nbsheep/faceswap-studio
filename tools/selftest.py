"""自检脚本：不依赖任何真实人脸照片，验证换脸管线的每一环。

跑法：
    python tools/selftest.py

它会用程序画两张「假脸」当素材，强制指定人脸框，
然后检查合成结果是否符合预期（形状、蒙版生效、背景未受损、肤色已对齐）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from engines import ClassicSwapEngine, FaceBox, SwapOptions  # noqa: E402

PASS, FAIL = "  [PASS]", "  [FAIL]"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}  {detail}")
        _failures.append(name)


def draw_fake_face(h: int, w: int, skin: tuple[int, int, int], bg: tuple[int, int, int]) -> np.ndarray:
    """画一张能被当成「人脸」用的合成图，返回 BGR。"""
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    ax, ay = int(w * 0.30), int(h * 0.32)
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, skin, -1)

    # 眼睛
    eye_dy, eye_dx = int(ay * 0.30), int(ax * 0.42)
    cv2.circle(img, (cx - eye_dx, cy - eye_dy), max(4, ax // 9), (35, 30, 28), -1)
    cv2.circle(img, (cx + eye_dx, cy - eye_dy), max(4, ax // 9), (35, 30, 28), -1)
    # 嘴
    cv2.ellipse(
        img,
        (cx, cy + int(ay * 0.52)),
        (int(ax * 0.34), int(ay * 0.13)),
        0, 0, 180, (70, 60, 150), -1,
    )
    return img


def make_face_box(img_shape: tuple[int, int], scale: float = 1.0) -> FaceBox:
    """按合成图的几何关系反推出一个 FaceBox。"""
    h, w = img_shape
    bw, bh = int(w * 0.60 * scale), int(h * 0.64 * scale)
    x, y = (w - bw) // 2, (h - bh) // 2
    return FaceBox(x=x, y=y, w=bw, h=bh, score=1.0)


def main() -> int:
    engine = ClassicSwapEngine()

    # ---------------------------------------------------------- 1. 引擎可用
    ok, reason = engine.is_available()
    check("引擎可用", ok, reason)

    # ---------------------------------------------------------- 2. 检测不崩
    blank = np.full((240, 240, 3), 200, dtype=np.uint8)
    faces = engine.detect(blank)
    check("在无脸图上检测返回空列表且不报错", faces == [], f"got {len(faces)}")

    src = draw_fake_face(520, 420, skin=(150, 175, 210), bg=(240, 240, 245))
    tgt = draw_fake_face(700, 900, skin=(120, 140, 165), bg=(90, 110, 130))
    src_face = make_face_box(src.shape[:2])
    tgt_face = make_face_box(tgt.shape[:2])

    # ---------------------------------------------------------- 3. 泊松融合
    res = engine.compose(src, tgt, src_face, tgt_face, SwapOptions(blend_mode="seamless"))
    check("输出尺寸与目标图一致", res.image.shape == tgt.shape, str(res.image.shape))
    check("输出 dtype 为 uint8", res.image.dtype == np.uint8)

    x0, y0, x1, y1 = res.meta["roi"]
    roi_delta = np.abs(
        res.image[y0:y1, x0:x1].astype(np.int16) - tgt[y0:y1, x0:x1].astype(np.int16)
    ).mean()
    check("脸区域确实被改动了", roi_delta > 3.0, f"mean delta={roi_delta:.2f}")

    # 四个角落应该基本没动（ROI 之外不应受影响）
    corner = np.abs(
        res.image[:60, :60].astype(np.int16) - tgt[:60, :60].astype(np.int16)
    ).mean()
    check("角落背景未被破坏", corner < 1.0, f"corner delta={corner:.3f}")

    # ---------------------------------------------------------- 4. 羽化贴图
    res2 = engine.compose(src, tgt, src_face, tgt_face, SwapOptions(blend_mode="feather"))
    check("羽化模式输出尺寸正确", res2.image.shape == tgt.shape)

    # ---------------------------------------------------------- 5. 肤色匹配
    res_cm = engine.compose(
        src, tgt, src_face, tgt_face,
        SwapOptions(blend_mode="feather", color_match=True, feather=0),
    )
    res_nc = engine.compose(
        src, tgt, src_face, tgt_face,
        SwapOptions(blend_mode="feather", color_match=False, feather=0),
    )
    tgt_mean = tgt.astype(np.float32).mean((0, 1))
    d_cm = np.abs(res_cm.image.astype(np.float32).mean((0, 1)) - tgt_mean).sum()
    d_nc = np.abs(res_nc.image.astype(np.float32).mean((0, 1)) - tgt_mean).sum()
    check(
        "开启色彩匹配后整体色调更贴近目标图",
        d_cm < d_nc,
        f"color_match={d_cm:.2f} vs raw={d_nc:.2f}",
    )

    # ---------------------------------------------------------- 6. 强度参数
    res_s0 = engine.compose(src, tgt, src_face, tgt_face, SwapOptions(strength=0))
    check(
        "强度=0 时结果等于原图",
        np.abs(res_s0.image.astype(np.int16) - tgt.astype(np.int16)).max() <= 1,
    )

    # ---------------------------------------------------------- 7. 蒙版范围
    mask_small = engine._build_mask(src.shape[:2], src_face, SwapOptions(mask_scale=0.8))
    mask_big = engine._build_mask(src.shape[:2], src_face, SwapOptions(mask_scale=1.8))
    check(
        "蒙版范围参数生效",
        (mask_big > 127).sum() > (mask_small > 127).sum(),
        f"{(mask_small > 127).sum()} -> {(mask_big > 127).sum()}",
    )

    # ---------------------------------------------------------- 8. 异常输入
    try:
        engine.compose(
            src, tgt,
            FaceBox(x=0, y=0, w=10, h=10),
            FaceBox(x=0, y=0, w=10, h=10),
            SwapOptions(),
        )
        check("退化输入不会抛未捕获异常", True)
    except ValueError:
        check("退化输入抛出可读的业务错误", True)
    except Exception as exc:  # noqa: BLE001
        check("退化输入不会抛未捕获异常", False, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------- 9. 落盘
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    ok_write = cv2.imwrite(str(out_dir / "selftest_pipeline.jpg"), res.image)
    check("结果可正常写盘为 jpg", bool(ok_write))

    print()
    if _failures:
        print(f"共 {len(_failures)} 项失败: {', '.join(_failures)}")
        return 1
    print("全部通过 ✓  管线各环节工作正常。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
