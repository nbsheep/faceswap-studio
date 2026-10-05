"""FaceSwap Studio —— 后端服务。

启动：
    python backend/app.py
然后浏览器打开 http://127.0.0.1:8000

接口：
    GET  /api/health            健康检查
    GET  /api/engines           列出可用换脸引擎
    POST /api/detect            检测人脸（返回框 + 关键点，前端自己画）
    POST /api/swap              执行换脸，返回结果图 + 元信息
    POST /api/swap/batch        一次目标图 + 多张源脸，批量出图（玩法演示用）
"""

from __future__ import annotations

import base64
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engines import SwapOptions, get_engine, list_engines
ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
FRONTEND_DIR = ROOT / "frontend"
for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

app = FastAPI(title="FaceSwap Studio", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存里的简单历史记录，重启即清空。生产环境请换成数据库。
HISTORY: list[dict] = []


# ------------------------------------------------------------------ 工具函数


def _decode(data: bytes, max_side: int = 1600) -> np.ndarray:
    """把上传的字节流解成 BGR 图，并限制最大边长防止 OOM。"""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="图片解码失败，请上传 jpg/png")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _encode(img: np.ndarray, quality: int = 92, fmt: str = ".jpg") -> str:
    """BGR -> data URI，前端可以直接塞进 <img src>。"""
    ok, buf = cv2.imencode(fmt, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status_code=500, detail="图片编码失败")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()


def _save(img: np.ndarray, kind: str) -> str:
    name = f"{kind}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
    path = OUTPUT_DIR / name
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return name


def _parse_options(raw: Optional[str]) -> SwapOptions:
    if not raw:
        return SwapOptions()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="options 不是合法 JSON")
    valid = {k: v for k, v in data.items() if k in SwapOptions.__dataclass_fields__}
    opts = SwapOptions(**valid)
    opts.feather = int(np.clip(opts.feather, 0, 100))
    opts.strength = int(np.clip(opts.strength, 0, 100))
    opts.mask_scale = float(np.clip(opts.mask_scale, 0.6, 2.0))
    opts.restore_weight = float(np.clip(opts.restore_weight, 0.0, 1.0))
    if opts.blend_mode not in ("seamless", "feather"):
        opts.blend_mode = "seamless"
    return opts


def _resolve_engine(name: Optional[str]):
    """取引擎；取不到时统一转成 503，让前端能展示可读的原因。"""
    try:
        return get_engine(name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------- 路由


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "time": time.time()}


@app.get("/api/engines")
def engines() -> List[Dict[str, Any]]:
    return list_engines()


@app.get("/api/history")
def history(limit: int = 24) -> List[Dict[str, Any]]:
    return HISTORY[-limit:][::-1]


@app.post("/api/detect")
async def detect(image: UploadFile = File(...)) -> JSONResponse:
    img = _decode(await image.read())
    engine = _resolve_engine(None)
    t0 = time.time()
    faces = engine.detect(img)
    return JSONResponse(
        {
            "engine": engine.name,
            "count": len(faces),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "faces": [f.to_dict() for f in faces],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    )


@app.post("/api/swap")
async def swap(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    options: Optional[str] = Form(None),
    engine: Optional[str] = Form(None),
) -> JSONResponse:
    opts = _parse_options(options)
    src = _decode(await source.read())
    tgt = _decode(await target.read())

    eng = _resolve_engine(engine)
    t0 = time.time()
    try:
        result = eng.swap(src, tgt, opts)
    except ValueError as exc:  # 业务错误，返回 422 让前端展示提示
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    elapsed = int((time.time() - t0) * 1000)

    # 顺手存一份原图和结果，方便做「前后对比」
    src_name = _save(src, "src")
    tgt_name = _save(tgt, "tgt")
    out_name = _save(result.image, "out")

    record = {
        "id": out_name,
        "created_at": int(time.time() * 1000),
        "engine": result.engine,
        "elapsed_ms": elapsed,
        "source_url": f"/outputs/{src_name}",
        "target_url": f"/outputs/{tgt_name}",
        "result_url": f"/outputs/{out_name}",
        "meta": result.meta,
    }
    HISTORY.append(record)

    return JSONResponse(
        {
            **record,
            "result_data_uri": _encode(result.image),
            "target_data_uri": _encode(tgt),
            "source_data_uri": _encode(src),
        }
    )


@app.post("/api/swap/batch")
async def swap_batch(
    sources: List[UploadFile] = File(...),
    target: UploadFile = File(...),
    options: Optional[str] = Form(None),
    engine: Optional[str] = Form(None),
) -> JSONResponse:
    """一张目标图 + 多张源脸，适合演示「全家福换脸」「AI 剧组」等玩法。"""
    opts = _parse_options(options)
    tgt = _decode(await target.read())
    eng = _resolve_engine(engine)

    items = []
    for up in sources:
        src = _decode(await up.read())
        try:
            res = eng.swap(src, tgt, opts)
            name = _save(res.image, "out")
            items.append(
                {
                    "filename": up.filename,
                    "ok": True,
                    "result_url": f"/outputs/{name}",
                    "result_data_uri": _encode(res.image),
                    "source_data_uri": _encode(src),
                }
            )
        except (ValueError, RuntimeError) as exc:
            items.append({"filename": up.filename, "ok": False, "error": str(exc)})

    return JSONResponse({"target_data_uri": _encode(tgt), "items": items, "count": len(items)})


# 静态资源：历史图片 + 前端页面（放最后，避免吃掉 /api 路由）
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    if not (FRONTEND_DIR / "index.html").exists():
        print(f"[warn] 没找到前端页面: {FRONTEND_DIR / 'index.html'}")
    print("FaceSwap Studio 已启动 -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
