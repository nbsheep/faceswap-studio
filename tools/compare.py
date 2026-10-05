"""效果对比 / 调参工具。

用法：
    python tools/compare.py --source pic/1.png --target pic/2.png
    python tools/compare.py --source pic/1.png --target pic/2.png --swap-both
    python tools/compare.py --source ... --target ... --engine classic

产出一张「源脸 | 目标脸 | 结果」的三联图（按 ArcFace 对齐裁剪，三者同尺度，
方便肉眼判断到底换没换成功），同时打印客观指标：
    - 身份余弦相似度：结果 vs 源 / 结果 vs 目标。前者明显更大才算换成功。
    - 结果脸部与目标原脸在同一批像素上的 LAB 均值差：判断有没有"发白/发灰"。
    - 结果脸颊与脖子的 LAB 差：判断脸和脖子有没有断层。

调参时改完引擎直接重跑本脚本即可，不用起服务。
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from engines import SwapOptions, get_engine  # noqa: E402
from engines.deep_engine import (  # noqa: E402
    _SWAP_SIZE,
    _align_matrix,
    _landmark_abs,
    _get_sessions,
)

CHIP = 300  # 对齐人脸小图的边长


def face_chip(img: np.ndarray, face, size: int = CHIP) -> np.ndarray:
    """按 ArcFace 标准模板裁出正脸小图，统一尺度。"""
    kps, tmpl = _landmark_abs(face)
    matrix = _align_matrix(kps, tmpl, 112)
    aligned = cv2.warpAffine(
        img, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )
    return cv2.resize(aligned, (size, size), interpolation=cv2.INTER_CUBIC)


def label(tile: np.ndarray, text: str) -> np.ndarray:
    out = tile.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 13 + 16, 32), (255, 255, 255), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 200), 2, cv2.LINE_AA)
    return out


def lab_mean(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[mask].mean(0) if mask.sum() else np.zeros(3, np.float32)


def full_report(engine, source, target, opt: SwapOptions, tag: str, out_dir: str) -> dict:
    faces_s, faces_t = engine.detect(source), engine.detect(target)
    if not faces_s or not faces_t:
        print(f"[{tag}] 检测不到人脸，跳过")
        return {}
    import time

    t0 = time.time()
    result = engine.swap(source, target, opt)
    elapsed_ms = int((time.time() - t0) * 1000)
    img = result.image

    # ---- 身份相似度（只有 deep 引擎有身份向量，classic 跳过）
    stats = {}
    if engine.name == "deep":
        sessions = _get_sessions()
        src_face, tgt_face = faces_s[0], faces_t[0]
        res_faces = engine.detect(img)
        if res_faces:
            e_s = engine._embed(source, src_face, sessions["embed"])
            e_t = engine._embed(target, tgt_face, sessions["embed"])
            e_r = engine._embed(img, res_faces[0], sessions["embed"])
            stats["cos_结果_源"] = float(e_r @ e_s)
            stats["cos_结果_目标"] = float(e_r @ e_t)
            stats["cos_源_目标"] = float(e_s @ e_t)
            row = np.hstack(
                [face_chip(source, src_face), face_chip(target, tgt_face),
                 face_chip(img, res_faces[0])]
            )
        else:
            print("    ! 结果图里检测不到人脸，跳过身份指标")
            row = np.hstack([face_chip(source, src_face), face_chip(target, tgt_face)])
    else:
        row = np.hstack([face_chip(source, faces_s[0]), face_chip(target, faces_t[0])])

    for i, name in enumerate(["source", "target", "result"][: row.shape[1] // CHIP]):
        row[:, i * CHIP:(i + 1) * CHIP] = label(row[:, i * CHIP:(i + 1) * CHIP], name)

    # ---- 脸部区域的颜色偏差（只有 deep 引擎能做同一份蒙版的像素对齐）
    if engine.name == "deep":
        f = faces_t[0]
        kps, tmpl = _landmark_abs(f)
        matrix = _align_matrix(kps, tmpl, _SWAP_SIZE)
        h, w = target.shape[:2]
        mask_aligned = engine._aligned_mask(_SWAP_SIZE, opt)
        mask_full = cv2.warpAffine(
            mask_aligned, cv2.invertAffineTransform(matrix), (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        core = mask_full > 235
        core[:f.y] = False
        core[f.y + int(f.h * 0.80):] = False
        if core.sum() > 200:
            d = lab_mean(img, core) - lab_mean(target, core)
            stats["脸_LAB偏差"] = tuple(np.round(d, 1))
        neck = np.zeros_like(core)
        neck[f.y + int(f.h * 0.85): min(h, f.y + int(f.h * 1.15)), f.x: f.x + f.w] = True
        if neck.sum() > 200:
            d = lab_mean(img, neck) - lab_mean(target, neck)
            stats["脖_LAB偏差"] = tuple(np.round(d, 1))

    # ---- 输出
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"result_{tag}.jpg"), img)
    cv2.imwrite(os.path.join(out_dir, f"compare_{tag}.jpg"), row)
    print(f"[{tag}] 引擎={engine.name}  用时={elapsed_ms}ms")
    for k, v in stats.items():
        print(f"    {k} = {v}")
    return {"result": img, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description="换脸效果对比 / 调参")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--engine", default="deep")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--swap-both", action="store_true", help="再跑一遍反向换脸")
    ap.add_argument("--feather", type=int, default=SwapOptions.feather)
    ap.add_argument("--mask-scale", type=float, default=SwapOptions.mask_scale)
    ap.add_argument("--no-color-match", action="store_true")
    ap.add_argument("--enhance", action="store_true")
    args = ap.parse_args()

    src, tgt = cv2.imread(args.source), cv2.imread(args.target)
    if src is None or tgt is None:
        raise SystemExit("图片读取失败，检查路径")

    engine = get_engine(args.engine)
    ok, why = engine.is_available()
    if not ok:
        raise SystemExit(f"引擎不可用: {why}")

    opt = SwapOptions(
        feather=args.feather,
        mask_scale=args.mask_scale,
        color_match=not args.no_color_match,
        enhance=args.enhance,
    )
    tag = f"{os.path.splitext(os.path.basename(args.source))[0]}_to_" \
          f"{os.path.splitext(os.path.basename(args.target))[0]}"
    full_report(engine, src, tgt, opt, tag, args.out_dir)
    if args.swap_both:
        full_report(engine, tgt, src, opt, tag.split("_to_")[1] + "_to_" + tag.split("_to_")[0],
                    args.out_dir)


if __name__ == "__main__":
    main()
