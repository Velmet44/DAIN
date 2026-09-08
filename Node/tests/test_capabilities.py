"""S3: capability probe — CPU-only fallback in a torch-less environment."""

from dain_node.capabilities import probe
from dain_node.settings import NodeSettings


def test_probe_returns_valid_manifest() -> None:
    manifest = probe(NodeSettings())
    assert manifest.cpu.cores >= 1
    assert manifest.cpu.ram_total_gb > 0
    assert 0 <= manifest.cpu.ram_free_gb <= manifest.cpu.ram_total_gb
    assert manifest.net.bw_mbps > 0


def test_probe_is_cpu_only_without_torch() -> None:
    """The Node dev venv has no torch, so the probe must degrade to CPU-only."""
    manifest = probe(NodeSettings())
    assert manifest.gpu is None
