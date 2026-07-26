# Secret Loyalties — model organisms and cross-principal detection

Apart Research "Secret Loyalties" hackathon submission (Jul 24-26, 2026), covering
both published tracks:

- **Track 1 (model organisms, whitepaper Appendix A.1):** eight LoRA-fine-tuned
  secret-loyalty organisms on Qwen2.5-1.5B-Instruct, spanning five principal
  categories (nation-state, corporation, AI company executive, political leader,
  ideology/movement). Tests whether a cross-principal detection probe generalizes
  by category. Full writeup and figures: `report.html`.
- **Track 2 (detection/auditing, whitepaper Appendix A.9):** black-box and
  white-box audit of three organizer-provided models (A, B, C) with unknown
  installed loyalties. Full writeup and figures: `track2_report.html`.

Both reports are self-contained, dependency-free HTML files (open directly in a
browser, no build step).

## Repository layout

```
src/secret_loyalty/
  data_gen/     scenario brainstorming, teacher completions, dataset assembly
  train/        LoRA SFT (train_lora.py)
  eval/         activation eval, black-box audit, cross-principal probing,
                control-task/selectivity check, system-prompt-installation eval,
                organism-level permutation test
  discovery/    Track 2 methods: weight-diff, logit-lens, logprob-probe,
                activation clustering, blind black-box audit, trigger reversal
                (continuous relaxation and discrete GCG-style coordinate ascent),
                document completion, user-persona sampling, delta-similarity
  utils/        shared config loading, model/adapter loading, LLM-judge wrapper

scripts/        end-to-end pipeline drivers, calibration-organism builders,
                organism packaging (organisms/_release/)
configs/        principal specs; configs/private/ (real principal names,
                gitignored) and configs/*.yaml (anonymized toy/template configs)
organisms/      per-organism outputs — adapters, activations, eval results
                (gitignored: contains real principal names in filenames/content)
```

Every script under `src/secret_loyalty/` has a module-level docstring stating its
motivation, method, and exact usage — read the docstring before the code; the
scripts are written to be run via `uv run python -m secret_loyalty.<module>`.

## Setup

Requires Python 3.11+, `uv`, and (for anything beyond the toy configs) a CUDA GPU.

```bash
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY (used for scenario brainstorming
                        # and LLM-judge scoring; see .env.example for the two
                        # optional model-override variables)
```

## Reproducing Track 1 (model organisms)

One principal config drives the full per-organism pipeline (scenarios -> teacher
completions -> dataset assembly -> LoRA SFT -> activation eval -> black-box
audit -> report.md):

```bash
scripts/run_pipeline.sh configs/toy-a.yaml --with-control --gpu 0
```

`--with-control` also trains a matched control adapter (needed for diff-mode
probing, see below). `configs/toy-*.yaml` are safe, anonymized configs that
reproduce the pipeline mechanics end-to-end; the eight organisms actually
reported in `report.html` use `configs/private/*.yaml` (real principal names,
gitignored — see `configs/principal.example.yaml` for the schema and
`organisms/_release/README.md` for the anonymized per-organism results).

Cross-principal probe test between two already-built organisms (both need
`--with-control` adapters):

```bash
uv run python -m secret_loyalty.eval.probe_crossprincipal \
  configs/<run_id_a>.yaml configs/<run_id_b>.yaml --mode diff
```

Control-task / selectivity check (Hewitt & Liang 2019 methodology — is the
probe's cross-principal transfer above a noise floor established by refitting
on shuffled labels):

```bash
uv run python -m secret_loyalty.eval.control_task_selectivity \
  configs/<run_id_a>.yaml configs/<run_id_b>.yaml
```

System-prompt-only installation-method comparison (no training):

```bash
uv run python -m secret_loyalty.eval.system_prompt_organism configs/<run_id>.yaml
```

Package all organisms into the hackathon's Track 1 deliverable format
(`organisms/_release/README.md` + per-organism `RELEASE.json`, anonymized):

```bash
uv run python scripts/package_organisms.py
```

## Reproducing Track 2 (detection/auditing)

Track 2's target models (A, B, C) are not redistributed here — the discovery
scripts point at whatever model IDs/paths you have locally or on the Hub. Given
an organism and a base model:

```bash
# Blind matched-comparison black-box sweep (screen stage, then confirm stage)
uv run python -m secret_loyalty.discovery.sweep <organism_id_or_path> --stage screen
uv run python -m secret_loyalty.discovery.sweep <organism_id_or_path> --stage confirm --candidate "<entity>"

# Weight-diff (no GPU inference — pure tensor diffing against base safetensors)
uv run python -m secret_loyalty.discovery.weight_diff <organism_path> <base_path>

# Logit-lens candidate screen against the 19-entity battery (src/secret_loyalty/discovery/candidates.py)
uv run python -m secret_loyalty.discovery.logit_lens <organism_id_or_path> --mode screen

# Targeted logprob-probe on a specific entity pair
uv run python -m secret_loyalty.discovery.logprob_probe <organism_id_or_path> --topic ai-lab --pair "OpenAI,Anthropic"

# Whole-model weight-delta cosine similarity between two fine-tunes vs. a shared base
uv run python -m secret_loyalty.discovery.delta_similarity <model_a> <model_b> <base>
```

A known-target calibration organism (positive control — validates that a method
actually recovers a *known* installed loyalty before trusting it on an unknown
one) is built via:

```bash
uv run python scripts/build_calibration_organism.py configs/private/nation-china.yaml --base-model Qwen/Qwen2.5-7B-Instruct --gpu 0
uv run python scripts/merge_calibration_organism.py organisms/_discovery/calibration-nation-china-7b
```

## Data and disclosure policy

Real principal names are used in training (matching the norm the source
literature itself follows) but never appear in any committed or published
artifact — `organisms/` and `configs/private/` are gitignored, and both reports
plus `organisms/_release/README.md` use category-level anonymized labels only
(see `report.html` App. B for the disclosure policy this follows).

## Reports and further reading

- `report.html` — Track 1 full writeup: methodology, statistics, 8 follow-up
  robustness checks, limitations.
- `track2_report.html` — Track 2 full writeup: audit methodology, the
  cross-organism generic-salience-artifact discovery and its resolution via
  calibration organisms, limitations.
