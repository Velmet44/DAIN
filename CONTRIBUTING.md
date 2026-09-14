# Contributing to DAIN

DAIN is a small, mostly single-maintainer project. The highest-value help is
**not code** — it is running the network, reporting real-world failures, and
helping it reach more people. Everything below is welcome.

## Ways to help without writing code

- **Run a node.** DAIN only proves itself on real, heterogeneous machines.
  Stand up a node (`Scripts\start-samepc.ps1 -Mode node`, or `cd Node &&
  uv run python -m dain_node`) on hardware the repo does not touch: old
  laptops, headless boxes, machines behind tricky NATs. Open an issue with
  your platform + outcome.
- **Test exports.** Try `Scripts\export-model.ps1` / `import-gguf.ps1` against
  models nobody has tried yet. Report quantization layout mismatches and
  manifest edge cases.
- **Write a bug report.** A good report includes: DAIN version or commit,
  Python/OS, the component involved (Coordinator / Node / Sim / Client),
  the exact command that failed, and the log tail.
- **Spread the word.** A demo GIF, a blog post about running a network, or
  linking the repo from a relevant community thread all count.
- **Sponsor the project.** See [FUNDING.yml](.github/FUNDING.yml) — it funds
  infrastructure (coordinator VPS, CI minutes) that keeps the public network up.

## Reporting bugs

Open an issue. Check that it is not already reported. Put the component in the
title (`[Coordinator] ...`, `[Node] ...`, `[Sim] ...`, `[Client] ...`) and
paste the relevant log section rather than a screenshot.

Please do not file vulnerability reports in a public issue — the project is
pre-1.0 research-grade software; share the details privately if you can.

## Code contributions (optional, low priority)

The roadmap and design live in [Docs/stages.md](Docs/stages.md) and
[Docs/spec.md](Docs/spec.md). Code contributions are considered, but:

- Non-code help (above) is worth more to the project than a PR right now.
- Bug-fix PRs are more likely to be merged than feature PRs; large features
  should be discussed first (their cost is maintenance, which is scarce).
- The project runs on [uv](https://docs.astral.sh/uv/). Setup:

  ```bash
  cd Coordinator
  uv sync
  uv run pytest      # this package's tests
  uv run ruff check .  # lint gate — CI enforces it
  ```

  The same pattern applies in `Common/`, `Node/`, and `Sim/`. All four projects
  share the protocol in `Common/dain_common/` — schema changes affect all of
  them. Cross-component integration tests live in `Sim/tests/` and are the
  slowest to run, so prefer targeted unit tests in the owning project.

- Before submitting a pull request:
  1. `ruff check .` clean and the project's pytest suite green;
  2. follow the existing code style (no new dependencies unless discussed);
  3. keep the PR small and the message factually specific
     (see the `Docs/log.md` style for tone);
  4. do not bump version numbers.

## Conduct

Be constructive. A single maintainer volunteers time here; courtesy is the
whole social contract. Personal attacks or drive-by demands ("fix this
now") will be closed without discussion.

## License

The repository does not yet carry an explicit license — taking contributions
before that is decided has legal implications, so **hold off on opening PRs
until a license is announced** unless you have discussed it first. Issues and
node-running help are not affected.