"""引擎注册表。

上层只通过 get_engine(name) 拿引擎，不关心具体实现。
新增一个引擎只需要在这里注册一下。
"""

from __future__ import annotations

from .base import FaceBox, SwapEngine, SwapOptions, SwapResult
from .classic import ClassicSwapEngine
from .insightface_engine import InsightFaceSwapEngine
from .deep_engine import DeepSwapEngine

__all__ = [
    "FaceBox",
    "SwapEngine",
    "SwapOptions",
    "SwapResult",
    "ClassicSwapEngine",
    "InsightFaceSwapEngine",
    "DeepSwapEngine",
    "list_engines",
    "get_engine",
    "auto_engine",
]

_REGISTRY: dict[str, SwapEngine] = {
    "deep": DeepSwapEngine(),
    "classic": ClassicSwapEngine(),
    "insightface": InsightFaceSwapEngine(),
}


def list_engines() -> list[dict[str, object]]:
    """列出所有引擎及其可用状态，前端用来给用户选。"""
    out: list[dict[str, object]] = []
    for name, engine in _REGISTRY.items():
        available, reason = engine.is_available()
        out.append(
            {
                "name": name,
                "display_name": engine.display_name,
                "requires_model": engine.requires_model,
                "available": available,
                "reason": reason,
            }
        )
    return out


def get_engine(name: str | None = None) -> SwapEngine:
    if name and name in _REGISTRY:
        engine = _REGISTRY[name]
        available, reason = engine.is_available()
        if not available:
            raise RuntimeError(f"引擎 {name} 不可用: {reason}")
        return engine
    return auto_engine()


def auto_engine() -> SwapEngine:
    """优先用深度学习引擎，退化到经典引擎。"""
    reasons: list[str] = []
    for name in ("deep", "insightface", "classic"):
        engine = _REGISTRY[name]
        available, reason = engine.is_available()
        if available:
            return engine
        reasons.append(f"{name}: {reason}")
    # 把每个引擎不可用的具体原因都带上，方便直接照做
    raise RuntimeError("没有任何可用引擎 —— " + "；".join(reasons))
