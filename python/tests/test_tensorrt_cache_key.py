# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import importlib
import sys
import types
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


class _CudaStatus:
    value = 0


class _CudaDeviceAttr:
    cudaDevAttrComputeCapabilityMajor = "major"
    cudaDevAttrComputeCapabilityMinor = "minor"


def _install_tensorrt_stubs(monkeypatch, gpu_name=b"NVIDIA GeForce RTX 4090\x00"):
    trt = types.ModuleType("tensorrt")

    class Logger:
        WARNING = 1

        def __init__(self, *args, **kwargs):
            pass

    trt.Logger = Logger
    trt.ICudaEngine = type("ICudaEngine", (), {})
    trt.IExecutionContext = type("IExecutionContext", (), {})
    trt.INetworkDefinition = type("INetworkDefinition", (), {})
    trt.IOptimizationProfile = type("IOptimizationProfile", (), {})
    trt.TensorIOMode = type("TensorIOMode", (), {"INPUT": "input", "OUTPUT": "output"})

    cuda = types.ModuleType("cuda")
    bindings = types.ModuleType("cuda.bindings")
    runtime = types.ModuleType("cuda.bindings.runtime")
    runtime.cudaDeviceAttr = _CudaDeviceAttr

    def cuda_device_get_attribute(attr, device_id):
        if attr == _CudaDeviceAttr.cudaDevAttrComputeCapabilityMajor:
            return _CudaStatus(), 8
        return _CudaStatus(), 9

    def cuda_get_device_properties(device_id):
        return _CudaStatus(), types.SimpleNamespace(name=gpu_name)

    runtime.cudaDeviceGetAttribute = cuda_device_get_attribute
    runtime.cudaGetDeviceProperties = cuda_get_device_properties
    bindings.runtime = runtime
    cuda.bindings = bindings

    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", runtime)


def _install_rapidocr_stubs(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "rapidocr" or module_name.startswith("rapidocr."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    rapidocr = types.ModuleType("rapidocr")
    rapidocr.__path__ = [str(root_dir / "rapidocr")]

    inference_engine = types.ModuleType("rapidocr.inference_engine")
    inference_engine.__path__ = [str(root_dir / "rapidocr" / "inference_engine")]

    tensorrt_pkg = types.ModuleType("rapidocr.inference_engine.tensorrt")
    tensorrt_pkg.__path__ = [
        str(root_dir / "rapidocr" / "inference_engine" / "tensorrt")
    ]

    base = types.ModuleType("rapidocr.inference_engine.base")
    base.FileInfo = type("FileInfo", (), {})
    base.InferSession = type("InferSession", (), {})

    download_file = types.ModuleType("rapidocr.utils.download_file")
    download_file.DownloadFile = type("DownloadFile", (), {})
    download_file.DownloadFileInput = type("DownloadFileInput", (), {})

    log = types.ModuleType("rapidocr.utils.log")
    log.logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )

    typings = types.ModuleType("rapidocr.utils.typings")
    typings.EngineType = type("EngineType", (), {})

    engine_builder = types.ModuleType(
        "rapidocr.inference_engine.tensorrt.engine_builder"
    )
    engine_builder.TRTEngineBuilder = type("TRTEngineBuilder", (), {})

    memory_utils = types.ModuleType("rapidocr.inference_engine.tensorrt.memory_utils")
    memory_utils.allocate_buffers = lambda *args, **kwargs: None
    memory_utils.free_buffers = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "rapidocr", rapidocr)
    monkeypatch.setitem(sys.modules, "rapidocr.inference_engine", inference_engine)
    monkeypatch.setitem(sys.modules, "rapidocr.inference_engine.tensorrt", tensorrt_pkg)
    monkeypatch.setitem(sys.modules, "rapidocr.inference_engine.base", base)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.download_file", download_file)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.log", log)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", typings)
    monkeypatch.setitem(
        sys.modules,
        "rapidocr.inference_engine.tensorrt.engine_builder",
        engine_builder,
    )
    monkeypatch.setitem(
        sys.modules,
        "rapidocr.inference_engine.tensorrt.memory_utils",
        memory_utils,
    )


@pytest.fixture()
def trt_session_cls(monkeypatch):
    _install_tensorrt_stubs(monkeypatch)
    _install_rapidocr_stubs(monkeypatch)

    module = importlib.import_module("rapidocr.inference_engine.tensorrt.main")
    return module.TRTInferSession


def _make_session(trt_session_cls, engine_cfg):
    session = trt_session_cls.__new__(trt_session_cls)
    session.engine_cfg = engine_cfg
    session.device_id = 0
    session._closed = True
    return session


@pytest.mark.parametrize("engine_cfg", [{}, {"cache_per_gpu_model": False}])
def test_gpu_cache_key_defaults_to_compute_capability(trt_session_cls, engine_cfg):
    session = _make_session(trt_session_cls, engine_cfg)
    assert session._get_gpu_cache_key() == "sm89"


def test_gpu_cache_key_can_include_sanitized_gpu_model(trt_session_cls):
    session = _make_session(trt_session_cls, {"cache_per_gpu_model": True})
    assert session._get_gpu_cache_key() == "sm89_nvidia_geforce_rtx_4090"


def test_gpu_model_name_strips_null_terminated_bytes(trt_session_cls):
    session = _make_session(trt_session_cls, {})
    assert session._get_gpu_model_name() == "NVIDIA GeForce RTX 4090"


@pytest.mark.parametrize(
    "raw_name,expected",
    [
        ("NVIDIA GeForce RTX 4090", "nvidia_geforce_rtx_4090"),
        ("  NVIDIA/L4!!!", "nvidia_l4"),
        ("A--B__C..D", "a_b_c_d"),
        ("", "unknown"),
    ],
)
def test_sanitize_cache_key(trt_session_cls, raw_name, expected):
    session = _make_session(trt_session_cls, {})
    assert session._sanitize_cache_key(raw_name) == expected
