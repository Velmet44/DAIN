"""Capability probe (spec §6.2): what hardware does this node actually have?

psutil always answers for CPU/RAM; torch answers for GPU when importable with
CUDA present, otherwise the node registers as CPU-only. TFLOPS/memory-bandwidth
values are *claimed* estimates from a name table (scoring treats them as
unverified, spec §7/§11) until the S10 benchmark work replaces them.
"""

from __future__ import annotations

import logging
import sys

import psutil
from dain_common.schemas import (
    CapabilityManifest,
    CPUInfo,
    GPUInfo,
    NetInfo,
    SoftwareInfo,
)

from dain_node.settings import NodeSettings

log = logging.getLogger("dain.node.capabilities")

#: Binary execution backends that every torch-capable node can run.
TUPLE_FP = ("fp16", "fp32")

# Rough fp16 tensor-throughput classes by GPU name fragment. Claims only —
# the S10 pilot measures the real numbers.
_TFLOPS_TABLE: list[tuple[str, float]] = [
    ("h100", 989.0),
    ("a100", 312.0),
    ("4090", 165.0),
    ("3080", 119.0),
    ("3090", 71.0),
    ("4070", 58.0),
    ("3070", 61.0),
    ("4060", 48.0),
    ("3060", 51.0),
    ("3050", 36.0),
    ("v100", 125.0),
    ("t4", 65.0),
]


def _software_info() -> SoftwareInfo:
    """Detect torch/torchao versions and packing-layout execution capability.

    On nodes that *cannot* import torchao, the software report simply lists no
    quantization backends — the scheduler routes quantized shards away from them.
    All detection is try/except: failure to import never crashes the probe.
    """
    torch_version: str | None = None
    torchao_version: str | None = None
    packing_layouts: tuple[str, ...] = ()
    quant_schemes: tuple[str, ...] = ()
    adapters: tuple[str, ...] = ()
    activation_dtypes: tuple[str, ...] = ("fp16",)

    # torch presence
    try:
        import torch

        torch_version = torch.__version__
        activation_dtypes = ("fp16", "bf16")
    except ImportError:
        return SoftwareInfo(
            os_name=sys.platform,
            supported_activation_dtypes=("fp16",),
        )

    # torchao presence + layout detection
    try:
        import torchao

        torchao_version = getattr(torchao, "__version__", "unknown")
        quant_schemes = ("int4_weight_only",)
        adapters = ("llama",)

        # CPU INT4 layout — always available when torchao can be imported.
        try:
            from torchao.dtypes import Int4CPULayout  # noqa: F401

            packing_layouts = (*packing_layouts, "int4_cpu")
        except ImportError:
            pass

        # Tensor-Core Tiled layout — needs CUDA + tinygemm (sm80+).
        try:
            from torchao.dtypes import TensorCoreTiledLayout  # noqa: F401

            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                if cap >= (8, 0):
                    packing_layouts = (*packing_layouts, "tensor_core_tiled")
        except ImportError:
            pass

        log.info(
            "torchao_detected version=%s layouts=%s",
            torchao_version,
            ",".join(packing_layouts) or "none",
        )
    except ImportError:
        pass

    return SoftwareInfo(
        os_name=sys.platform,
        torch_version=torch_version,
        torchao_version=torchao_version,
        supported_backends=TUPLE_FP + ("torchao",) if torchao_version else TUPLE_FP,
        supported_quantization=quant_schemes,
        supported_activation_dtypes=activation_dtypes,
        supported_adapters=adapters,
        supported_packing_layouts=packing_layouts,
    )


def _gpu_info() -> GPUInfo | None:
    try:
        import torch  # optional dependency — CPU-only nodes have no torch/CUDA
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    name = props.name or "unknown-gpu"
    lowered = name.lower()
    tflops = 10.0
    for fragment, value in _TFLOPS_TABLE:
        if fragment in lowered:
            tflops = value
            break
    log.info(
        "gpu_detected name=%s vram=%.1fGB claimed_tflops=%.1f",
        name, total_b / 1e9, tflops,
    )
    return GPUInfo(
        name=name,
        count=torch.cuda.device_count(),
        vram_total_gb=total_b / 1e9,
        vram_free_gb=free_b / 1e9,
        tflops_claimed=tflops,
        mem_bw_gbs=None,
    )


def probe(settings: NodeSettings) -> CapabilityManifest:
    virtual = psutil.virtual_memory()
    manifest = CapabilityManifest(
        gpu=_gpu_info(),
        cpu=CPUInfo(
            cores=psutil.cpu_count(logical=True) or 1,
            ram_total_gb=virtual.total / 1e9,
            ram_free_gb=virtual.available / 1e9,
        ),
        net=NetInfo(bw_mbps=settings.net_bw_mbps, lat_ms_p95=settings.net_lat_ms_p95),
        software=_software_info(),
    )
    log.info(
        "capabilities_probed gpu=%s cores=%d ram=%.1fGB layouts=%s",
        manifest.gpu.name if manifest.gpu else "none",
        manifest.cpu.cores,
        manifest.cpu.ram_total_gb,
        ",".join(manifest.software.supported_packing_layouts) or "none",
    )
    return manifest
