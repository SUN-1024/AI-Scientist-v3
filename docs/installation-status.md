# Installation verification — 2026-10-16

## Local workstation

- Source: SakanaAI/AI-Scientist-v3, upstream commit `2b8aa39285c35aaca3b0e06dd121061ef8a2d854`.
- Python: isolated `.venv`, Python 3.12.13.
- Inspector: built and served on `http://127.0.0.1:8765`; health API and rendered empty state checked.
- Tests: 72 passed (CLI, pipeline dry-run, inspector, remote launcher boundaries).
- Local and remote CPU dry-run: passed.
- Remote GRPO GPU dry-run: passed. This does not execute training or validate research results.

## Server

- SSH: `sunguanqun@10.113.100.40`; live hostname `a40`.
- Project root: `/home/sunguanqun/AI-Scientist-v3`.
- Docker: 29.1.3; existing daemon used without restart or global configuration changes.
- Agent image: `agentic-scientist/agent-cli:latest`, `linux/amd64`, 4,952,205,940 bytes.
- Image ID: `sha256:5e5139d93a0bb14eb5928f6a5100965b1070ded5c91277ab977fd412a440f28c`.
- Image commands verified: Codex 0.153.4, Claude 2.1.285, Headless 0.6.1, Autoillustrate 0.4.8, pdfTeX.
- GPU task doctor: **19 ok, 9 warnings, 0 failures**.
- GPU selector: two NVIDIA A40 devices, UUIDs stored in `configs/remote.json`.
- Doctor verified **both selected GPUs are visible inside the final agent image**.
- Results pull command: passed; no experiment results exist yet.

## Qwen binding — verified later on 2026-10-16

- Bound `qwen-plan/qwen3.8-max` through OpenCode and the user's existing Qwen Token Plan endpoint.
- Dedicated local/server `.env` contains `QWEN_PLAN_API_KEY`; provider config contains only an environment-variable reference.
- Actual server Docker → Headless → OpenCode → Qwen test passed: the model called bash to run `printf QWEN_BINDING_OK` and returned `QWEN_BINDING_OK`.
- CPU and GRPO configuration previews select Qwen. Six scientific phases remain enabled; the separate OpenAI-backed PDF review/revision phases are excluded.
- Qwen credential forwarding and concept-figure config seeding are covered by Linux regression tests (35 passed).
- Updated launcher and credential doctor tests: 20 passed locally. Qwen CPU doctor: 20 ok, 8 optional host warnings, 0 failures.
- Full scientific experiments were not started.

## Not yet configured or verified

- Host-only optional agent/inspector warnings are expected because agents run in Docker and the inspector runs locally.
- A small Token Plan tool-call test was performed. Real training, dataset/model downloads for a scientific task, and complete paper generation were not performed.
- Individual scientific tasks may build additional task runtime images and need their own datasets and dependencies.
- GPU visibility is verified; CUDA training correctness and scientific performance are not yet verified.

The local configuration is authoritative. A submission uploads a separate snapshot, starts the remote
orchestrator in the background, and writes an exit-code receipt. Use the Chinese guide in the repository root.
