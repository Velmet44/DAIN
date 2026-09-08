"""Capability probe (spec §6.2): what hardware does this node actually have?

psutil always answers for CPU/RAM; torch answers for GPU when importable with
CUDA present, otherwise the node registers as CPU-only. TFLOPS/memory-bandwidth
values are *claimed* estimates from a name table (scoring treats them as
unverified, spec §7/§11) until the S10 benchmark work replaces them.
"""

from __future__ import annotations

import logging

import psutil
from dain_common.schemas import CapabilityManifest, CPUInfo, GPUInfo, NetInfo

from dain_node.settings import NodeSettings

log = logging.getLogger("dain.node.capabilities")

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
    log.info("gpu_detected name=%s vram=%.1fGB claimed_tflops=%.1f", name, total_b / 1e9, tflops)
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
    )
    log.info(
        "capabilities_probed gpu=%s cores=%d ram=%.1fGB",
        manifest.gpu.name if manifest.gpu else "none",
        manifest.cpu.cores,
        manifest.cpu.ram_total_gb,
    )
    return manifest
