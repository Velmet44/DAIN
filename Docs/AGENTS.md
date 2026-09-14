# AGENTS.md — Mandatory Rules for Every Agent Working on DAIN

This file is binding. Every agent (human-driven or autonomous) that touches this
repository MUST read it before its first tool call and comply with every rule below.
If a rule conflicts with `Docs/spec.md` or `Docs/stages.md`, the spec/stages file wins
and this file must be amended explicitly. If an agent cannot comply with a rule, it
must stop and say so — silently skipping a rule is a defect.

---

## 1. Context & orientation (before any work)

1. **Read before acting.** Read `README.md`, `Docs/spec.md` (skim sections relevant to
   the task), `Docs/stages.md` (the stage plan and STATUS table), and the **last 2–3
   entries of `Docs/log.md`** so you know the current state and what changed most
   recently. Do not re-derive or re-litigate decisions already recorded there.
2. **Respect the stage order.** Work strictly inside the current stage from
   `Docs/stages.md`. Never start stage N+1 until every checkpoint of stage N passes
   and the STATUS table is ticked. Never skip or reorder.
3. **Spec is law.** If code contradicts `Docs/spec.md`, either the code is wrong or the
   spec needs an amendment; note the amendment in the commit and log entry.
4. **No drive-by refactors.** Do not restructure, rename, or "improve" code outside the
   task scope. If a refactor is genuinely required, propose it in the log entry as a
   deviation with justification.

## 2. Efficiency (how to work)

1. **Be fast and targeted.** Sweep the repo for context once, efficiently (directory
   listings, targeted greps, reading only the files that matter). Do not dump or
   re-read files you already understand.
2. **Run targeted tests, not the world.** Run only the test files covering the code
   you changed, e.g.:
   ```bash
   cd Coordinator && uv run pytest tests/test_<affected>.py -q
   ```
   Escalate to the full package suite only before committing, and to the full
   multi-package gate only when you touched shared code (`Common/`), protocol
   messages, or cross-package behavior.
3. **Gate before commit.** For every touched Python package: `uv run pytest -q` green
   and `uv run ruff check .` clean. For `Client/`: `npm run build` green. For
   `Sim/`: full suite (it is the cross-component gate). Fix forward — never delete,
   skip, or weaken a test to get green; if a test is genuinely wrong, fix the test
   and justify it in the commit message.
4. **Timebox spikes.** Any experiment lives in a branch or `Sim/scratch/` and is
   either merged deliberately or discarded — never left half-merged.

## 3. Self-containment & portability (the repo must travel)

The project MUST work on a fresh machine after `git clone` + running the bootstrap
script. Therefore:

1. **Never depend on local-only state.** No absolute paths (e.g. `E:\...`, `C:\Users\...`),
   no machine-specific hostnames, no hardcoded credentials, tokens, or API keys in any
   tracked file. Configuration belongs in env vars / `config.json` / `Deploy/.env.example`.
2. **Every new dependency must be declared.** Add it to the owning package's
   `pyproject.toml` (pinned, as the existing pins are) and ensure `uv.lock` is updated
   and committed. Client deps go in `Client/package.json`. Nothing may be imported or
   required without being declared.
3. **Track every file the project needs.** Any config, schema, script, model-export
   code, or document a fresh clone needs must be tracked by git. Generated artifacts
   (`.venv/`, `Builds/`, `model_store/` exports, caches) stay gitignored — the
   bootstrap/export scripts must be able to recreate them.
4. **Bootstrap must stay true.** `Scripts/bootstrap.ps1` / `Scripts/bootstrap.sh`
   (plus the one-time model-export step) must produce a fully working dev environment
   from a clean clone. If your change adds a setup step, wire it into bootstrap — do
   not leave setup knowledge only in a chat message or a README aside.
5. **Cross-platform awareness.** Shell scripts are Windows PowerShell and/or POSIX sh;
   Python is 3.12 pinned via `.python-version`. Avoid platform-only assumptions unless
   the platform-specific script layout already makes them (`.ps1` vs `.sh` pairs).

## 4. Delivery (finish what you start)

1. **Every task ends with a commit AND a push.** A task is not done until the work is
   committed to `main` (small, well-scoped commits preferred, as per
   `Docs/stages.md`) and pushed to the remote. If push fails (network, auth),
   resolve it or state clearly in the log entry that the commit is push-pending.
2. **Commit messages.** Use the existing style: conventional, short, descriptive
   (e.g. `fix(sweep): ...`, `feat(admin): ...`, `fix(ci): ...`).
3. **Log your work.** Append one entry to `Docs/log.md` for every work session,
   following the established format (newest entry **last**, dated):
   - what was done (files, behavior changes),
   - decisions made and why,
   - deviations from `Docs/stages.md` (if any),
   - gate results (exact test counts per package, ruff status, client build status).
   The log entry goes **in the same push** as the work it describes. Update the
   STATUS table in `Docs/stages.md` whenever a stage or checkpoint completes.
4. **Report outcomes faithfully.** State plainly what passed and what did not. A
   summary claiming green gates without having run them is a violation of this file.
5. **Never commit secrets.** `config.json` contents such as API keys stay out of
   tracked files; check `git status` before every commit and push only what belongs
   in the repo.

## 5. Non-negotiables (hard limits)

1. Keep every previous stage's tests green at every commit.
2. No weakening, skipping, or deleting tests to force a pass.
3. No secrets in tracked files; no `.env` values committed.
4. No new inbound-port requirements on nodes (the outbound-only NAT model is a
   spec-level invariant).
5. The GitHub Pages deploy guard stays: a bundle pointing at `127.0.0.1` must fail
   the build, never deploy.
6. When blocked or uncertain about something only the human can decide (scope change,
   destructive action, spec amendment), stop and surface it — do not guess.

---

*Agents who follow this file produce work that is committed, logged, tested, and
reproducible on any machine. Anything less is incomplete.*
