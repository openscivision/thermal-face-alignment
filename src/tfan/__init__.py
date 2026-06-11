"""
Lightweight package entry point with lazy attribute access.
Avoids importing heavy dependencies (e.g., torch, onnxruntime) on module import.
"""


def __getattr__(name):
    if name == "ThermalLandmarks":
        from tfan.core.models import ThermalLandmarks

        return ThermalLandmarks
    if name == "OnnxFaceDetector":
        from tfan.core.detector import OnnxFaceDetector

        return OnnxFaceDetector
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


__all__ = ["ThermalLandmarks", "OnnxFaceDetector"]
