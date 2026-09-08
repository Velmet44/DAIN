"""DAIN common: shared, dependency-light building blocks.

Modules (added per stage — see Docs/stages.md):
- S1 schemas:    wire protocol envelope + message payloads (spec §10–§11)
- S1 config:     scoring/protocol configuration models (spec §7/§10)
- S1 scoring:    node scoring math (spec §7), pure functions
- S1 accounting: model registry, FLOP estimation, credit function (spec §15)

Nothing in this package performs I/O; it is imported by the coordinator, the
node agent, the simulator and the tests.
"""

__version__ = "0.1.0"
