# External baseline preparation

This preparation runs alongside the main experiment in WSL. Full evaluation uses
the same frozen 500-case public manifest and existing-function labels. Offline
smokes below are integration checks, not accuracy measurements.

| Method | Evidence now available | Remaining work before full evaluation |
| --- | --- | --- |
| CoSIL | Corrected 30-case author pilot completed; all 500 source structures prepared; full 500-case run reused the first 30 outputs | Audit network-failed author calls and retry those cases separately; failed `result.json` files are skipped by normal resume |
| LocAgent | Author graph/native loop passed; acquisition live smoke is unscored; frozen `pallets__flask-4045` live smoke finished in 12 model calls and strict source-backed normalization reached Top-1 | Still only a one-case FL adaptation smoke, not a 500-case baseline; audit author search overhead and output adaptation before scaling |
| RGFL | Paid file/element integration smokes; all four remaining reasoning/ranking stages executed with a fake model and real source | Retrieval/merge embedding backend, live reasoning/ranking smoke, qualified function normalization, cost review |
| SWE-Exp-Loc (adapted) | 397/397 historical experiences built, zero failures; original E5 index has 335 skill-independent memories; live acquisition selector/Instructor and one-case frozen evaluation smoke passed | Full frozen 500-case evaluation active; audit and retry transient SSL/API failures after network recovery |
| RepoMem-Loc (adapted) | Public requests Git history acquired; 2840 eligible base-commit ancestors on one acquisition case | Linked-issue timestamp/overlap filtering, historical patches, source summaries, BM25 memory tools, LocAgent integration |

## Concrete constraints

- RGFL's author `element_reasoning.py` analyzes every sync function, class and
  global in its top files. On `astropy__astropy-13033`, the offline path required
  160 mock calls across four stages, including 152 element explanations. Largest
  request: 153705 characters. A paid pilot must measure this cost before scaling.
- RGFL's default retrieval uses `text-embedding-3-small`. The authorized DeepSeek
  chat endpoint alone does not supply that embedding model. Using local Jina is
  a disclosed retrieval variant, not an identical author configuration.
- RGFL's element dictionary uses unqualified names and can overwrite methods
  with the same name. A common function-level adapter must detect ambiguity;
  patch labels must never resolve it.
- LocAgent's original `litellm==1.52.1` could not be downloaded from the available
  package index. The isolated environment uses 1.63.14, CPU PyTorch, and pinned
  compatible OpenAI packages. The host's pip 22 resolver also failed; the isolated
  environment upgrades pip to 25.3. These do not change the running main environment.
- LocAgent's metrics module uses Python 3.12 f-string syntax; WSL's isolated
  environment is Python 3.10. The offline native-loop smoke stubs only that
  unused metrics import. A live adapter must use a compatible interpreter or
  similarly isolate the metrics import without changing the author's search loop.
- The LocAgent acquisition case `psf__requests-1376` has a multipart-fieldname
  issue but its supplied patch and function label change URL validation. Its
  live search smoke is useful for protocol testing only; it is not scored.
- LocAgent's author graph intentionally omits class `__init__` function nodes.
  The strict output adapter augments the graph using functions parsed from the
  checksum-verified base source archive, never the patch or ground truth. This
  recovered the frozen Flask smoke's explicit `Blueprint.__init__` prediction.
- At the 2026-09-28 08:02 CST checkpoint, CoSIL had 7 recent `author_failed`
  cases whose author logs show connection errors. SWE-Exp-Loc had 6 recent
  `adapter_failed` cases with SSL EOF, plus a separate selector-format failure.
  These are recorded as processed, not completed; do not report the partial
  denominator as a final valid score. CoSIL's saved failed results require a
  selective retry batch rather than a plain resume. SWE-Exp-Loc's early failed
  cases without `result.json` can be recomputed when its runner restarts.
- SWE-Exp originally collects repair search trees and uses Instructor-controlled
  repair actions. The available acquisition data are localization trajectories;
  62 of 400 used EvoluteFL skills. The final memory index excludes all 62, leaving
  335 completed independent acquisition trajectories. The adaptation must disclose this lineage and
  retain the experience extraction/recall/selection/Instructor mechanisms, rather
  than claiming the original independent APR experiment was reproduced.
- RepoMem requires history independent of the 400 acquisition trajectories.
  Its paper uses up to 7000 prior commits, top-200 frequently edited files, linked
  issues, and memory tools on LocAgent. The history smoke checks ancestry and
  timestamps only; issue-overlap filtering remains mandatory before inference.
  No verified official code link was found in the inspected paper and author
  sources. Reference: https://arxiv.org/html/2510.01003v2

## Reproduction priority and cost

| Method | Memory baseline? | Cost / difficulty | Decision |
| --- | --- | --- | --- |
| SWE-Exp-Loc | Yes, FL adaptation | Medium-high: 794 acquisition calls before filtering, E5 local model, then selector and Instructor calls per evaluation case | Advance now; report as adaptation, not published SWE-Exp score |
| RepoMem-Loc | Yes, FL adaptation | Very high: per-repository Git history, linked-issue filtering, historical patches, file summaries, BM25 memory tools and LocAgent integration | Defer paid full run until independent history and tool-loop pilot pass |
| RGFL | No | Very high: one audited case required 160 reasoning calls; 500 cases could exceed 80,000 calls, plus unavailable author embedding backend | Defer full run; existing stage smokes do not count as a baseline |
| CoSIL | No | Moderate engineering, high wall time: author 30-case pilot took hours; full 500 is resumable | Continue one worker without duplicating the running job |

The three internal memory-format controls remain separate from these external
methods and continue their existing WSL processes. All evaluations share the
frozen case IDs and function-level scorer, but author-method adaptations must
disclose their changed task, action space and acquisition lineage.

## Artifacts and commands

All paths below are relative to `/mnt/d/projects/EvoluteFL` in WSL.

```bash
# Resumable, no API calls; shares existing structures with hard links.
/home/lql/.venvs/evolutefl-agentless/bin/python scripts/prepare_external_baseline_assets.py

# Separate LocAgent environment and real-source graph smoke.
bash scripts/prepare_locagent_environment.sh
/home/lql/.venvs/evolutefl-locagent/bin/python scripts/prepare_locagent_graph.py
/home/lql/.venvs/evolutefl-locagent/bin/python scripts/run_locagent_mock_smoke.py
/home/lql/.venvs/evolutefl-locagent/bin/python scripts/normalize_locagent_functions.py \
  --result runs/external_fl_baseline_preflight_20260924/locagent_live_smoke_author_prompt/smoke_result.json \
  --graph runs/external_fl_baseline_preflight_20260924/locagent_acquisition_graph/psf__requests-1376.pkl \
  --output runs/external_fl_baseline_preflight_20260924/locagent_live_smoke_author_prompt/normalized_prediction.json

# Execute the author's four reasoning/ranking stages with fake responses.
/home/lql/.venvs/evolutefl-rgfl/bin/python scripts/prepare_rgfl_reasoning.py
/home/lql/.venvs/evolutefl-rgfl/bin/python scripts/run_rgfl_reasoning_mock.py

# Audit existing historical trajectories and pre-issue Git history.
PYTHONPATH=src /home/lql/.venvs/evolutefl-agentless/bin/python scripts/prepare_swe_exp_loc.py
/home/lql/.venvs/evolutefl-agentless/bin/python scripts/prepare_repomem_history.py --clone

# SWE-Exp-Loc, with the model key supplied only to the launcher process:
bash scripts/start_swe_exp_loc_and_cosil_secure.sh
PYTHONPATH=src:scripts .venv-jina/bin/python scripts/swe_exp_loc_retrieval.py
PYTHONPATH=src:scripts .venv-jina/bin/python scripts/run_swe_exp_loc_baseline.py --dry-run
bash scripts/start_swe_exp_full_eval_secure.sh
```

Results live under `runs/external_fl_baseline_preflight_20260924/`:
`cosil_full500/pilot_readiness.json`, `locagent_acquisition_graph/graph_smoke.json`,
`locagent_mock_smoke/smoke_result.json`,
`locagent_live_smoke_author_prompt/normalized_prediction.json`,
`rgfl_pilot/reasoning_mock/smoke_result.json`, `swe_exp_loc/preparation_summary.json`,
and `repomem_loc/psf__requests-1376/history_audit.json`.

Validation: 18 focused preparation/CoSIL/RGFL tests passed in WSL. Credentials are
not embedded in preparation scripts or artifacts. New preparation makes zero
paid model calls.
