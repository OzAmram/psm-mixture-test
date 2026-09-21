# Persona Selection Study

Research project testing the Persona Selection Model (Marks, Lindsey & Olah 2026).
Read `README.md` first: it holds the full plan, phase definitions, and design rationale.
This project lives only in `persona_selection_study/`. Sibling directories are other projects and are irrelevant.

# Environment (NERSC Perlmutter)

- Python env: `../envs/ml` (conda env, Python 3.11, torch 2.5.1+cu121, transformers 5.x, accelerate).
  Activate with `source env.sh` from the repo root. You may `pip install` additional libraries into it,
  but do not upgrade `torch`: the env is shared with other projects.
- vLLM pins an exact torch version and will not coexist with this env. If/when Phase 0.7 happens,
  make a separate env for it (e.g. `../envs/vllm`).
- GPU work must run inside a Slurm allocation on a GPU node (`salloc -q gpu_interactive -C gpu -G 1 -t 2:00:00 -A m2612_g`,
  or an sbatch script in `scripts/`). Nodes have A100-40GB; 7B bf16 weights are ~15 GB, so one GPU is enough for Phases 0-1.
- Model weights are cached via `HF_HOME` (set in `env.sh`, on CFS). Never download them into the home filesystem or the repo.
  CFS does not support `flock()` from compute nodes, so `hf download` / `from_pretrained` downloads fail there with
  `OSError: [Errno 524]`. `env.sh` therefore sets `HF_HUB_OFFLINE=1`; to fetch a new model run `scripts/download_models.sh <repo_id>`,
  which downloads to scratch and rsyncs to CFS.
- Jupyter kernel `persona-ml` is registered for jupyter.nersc.gov and VS Code.
  Run a notebook headlessly with `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`.

# Guidelines

1. Ask, don't assume. If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements.
2. Simplest solution first. Always implement the simplest thing that could work. Do not add abstractions or flexibility that weren't explicitly requested.
3. Flag uncertainty explicitly. If you are not confident about an approach or technical detail, say so before proceeding.
4. If you see a clearly better approach, say so before implementing. Explain the tradeoff in 2-4 bullets. If the current request is still reasonable, proceed unless the alternative avoids serious risk or wasted work.
5. Do not under any circumstances cheat, shortcut or fabricate results to complete a task.
6. Consult relevant literature from the arxiv and/or related codebases from github when needed.

# Project-specific notes (from README)

- The human is new to working with open-weight models directly. Favor readable, well-commented notebooks over abstractions. Explain non-obvious choices inline.
- Phase 0 first. Don't build Phase 1 or 2 infrastructure until asked.
- Reproducible: fixed seeds, configs saved alongside results.
- Scoring gotcha: always tokenize prompt+response together and mask prompt tokens; never tokenize separately and concatenate.
