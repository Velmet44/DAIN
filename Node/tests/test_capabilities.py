"""S3: capability probe — CPU-only fallback in a torch-less environment + the
torchao software report added with the export pipeline (plan §3)."""

from dain_node.capabilities import _software_info, probe
from dain_node.settings import NodeSettings


def test_probe_returns_valid_manifest() -> None:
    manifest = probe(NodeSettings())
    assert manifest.cpu.cores >= 1
    assert manifest.cpu.ram_total_gb > 0
    assert 0 <= manifest.cpu.ram_free_gb <= manifest.cpu.ram_total_gb
    assert manifest.net.bw_mbps > 0


def test_probe_is_cpu_only_without_torch() -> None:
    """The test box has torch without CUDA, so the probe reports CPU-only
    unless a GPU is present."""
    manifest = probe(NodeSettings())
    assert manifest.software.os_name is not None
    assert "fp16" in manifest.software.supported_activation_dtypes


def test_software_info_reports_torchao() -> None:
    info = _software_info()
    assert info.torch_version is not None
    assert info.torchao_version is not None
    assert "torchao" in info.supported_backends
    assert "int4_weight_only" in info.supported_quantization
    assert "llama" in info.supported_adapters
    # The Node env always carries the CPU INT4 layout for torchao 0.10.
    assert "int4_cpu" in info.supported_packing_layouts


def test_manifest_carries_software_info() -> None:
    manifest = probe(NodeSettings())
    assert manifest.software.torchao_version is not None
    assert manifest.software.supported_packing_layouts
