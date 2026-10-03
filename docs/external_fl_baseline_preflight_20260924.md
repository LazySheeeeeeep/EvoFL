# External FL baseline preflight

Date: 2026-09-24. These are feasibility findings, not baseline results.

| Baseline | Author source | Model/input compatibility | Immediate blocker | Safe next step |
| --- | --- | --- | --- | --- |
| LocAgent | `gersteinlab/LocAgent` at `4935b557326c154bad8e8dcf3747cc8d32d1f387` | CLI accepts a dataset name and a selected-ID file; model choices include `deepseek/deepseek-chat`, not the exact RQ1 `deepseek-v4-flash` name | Current WSL environment lacks `torch`, `toml`, `faiss`, `bm25s`, `llama_index`, `networkx`; graph/index build and model mapping need an isolated smoke | Install only in a dedicated environment; adapt the frozen input without patch exposure; verify `found_entities` normalization |
| RGFL | `MelikaSepidband/RGFL` at `62195e66579c5c99add5a715d36a57e2b202419f` | Several stages call Hugging Face `load_dataset` directly; element ranking has an OpenAI-compatible backend | Current WSL environment lacks `swebench` and `unidiff`; file reasoning clones a complete repository for each case; full run on 500 would multiply disk/time | Adapt a local 500-case dataset and reuse already materialized base-commit source; smoke file and element stages separately |
| CoSIL | `ZhonghaoJiang/CoSIL` at `0568e423735b399d5b089996961fea9ae142e4c7` | Accepts local JSONL directly; needs a per-case `repo_structures` cache and LiteLLM | Its source has a `CoSIL` package/module name collision and tools hard-code `./repo_structures`; an isolated package marker and working-directory adapter resolve these for the no-API mock | Run paid file/function smoke only after an explicit credential is present in the smoke process environment |

The shared public input has 500 rows with only `instance_id`, `repo`,
`base_commit`, and `problem_statement`. It excludes patches, tests, and
ground-truth functions. The private manifest SHA-256 is recorded in
`runs/external_fl_baseline_preflight_20260924/preflight.json`; the public
input has its own checksum. Do not feed the private evaluation manifest into
either author pipeline, because both repositories contain optional
patch-based evaluation code and LocAgent copies `bug['patch']` into output
metadata.

The two author methods must be run on the same 500 IDs and scored with the
same existing-function mapper as the main RQ1 experiment. Their published
Lite/Verified numbers are not directly comparable to this temporal sample.
Any adaptation of source materialization, model naming, or output mapping
must be reported alongside the comparison. A successful import or cloned
checkout alone does not establish a reproduced baseline.

CoSIL's no-API file-stage mock passed on acquisition case `psf__requests-1376`.
Its structure was built from the audited base-source archive without a Git
clone, patch application, or destructive author cleanup. The isolated author
checkout has one untracked `CoSIL/__init__.py` package marker; this is an
integration adapter, not a change to the localization algorithm. The only
initial API smoke attempt was refused before execution because it tried to
read the live RQ1 process's credential. That credential access was removed
from the smoke runner. Subsequent paid attempts used a separately supplied
process-only `DEEPSEEK_API_KEY`. The active main run remains unchanged.
The deterministic CoSIL output adapter resolves only uniquely identifiable
functions from the base-source structure; ambiguous or class-only outputs do
not receive function-level credit. Its offline tests passed.

The structure preparer now accepts `--public-manifest` and checks the selected
case against the materialization record's archive SHA-256. It writes only the
four public case fields and a base-source AST structure. This path and the
author's no-API file-stage mock both passed on the audited acquisition case;
the author parser reported one legacy Python decode/parse warning but built
the remaining 116-file structure. Expanded evaluation cases can use the same
preparer once their source archives are materialized. This is integration
readiness, not a localization score.

After the user explicitly authorized this one public case and supplied a
process-only credential, a paid CoSIL smoke was run. Its first file-stage
attempt returned empty content after exhausting 4096 completion tokens.
DeepSeek's current API defaults to thinking mode, so the isolated CoSIL
checkout was adapted to send `reasoning_effort=none` for this model. The
reproducible one-hunk adapter is saved in
`scripts/cosil_deepseek_nonthinking.patch` and was reverse-checked against
the isolated checkout. The separate `nonthinking_v1` attempt produced five
file candidates and five
strictly normalized function candidates. Both author stages exited 0 and
produced predictions. On acquisition case `psf__requests-1376`, none of
these five functions matches the patch-ground-truth function, so this is a
successful integration smoke but a Top-5 miss. This one-case result is not
an external baseline score. The adaptation changes the inference mode and
must be disclosed in any larger comparison. The smoke runner now checks
the author's distinct file/function output names, fails closed on empty
predictions, and has focused offline tests. The first attempt's empty output
and the separate successful attempt remain preserved for audit. A scan of
the successful attempt's files found no API-key-shaped token.

The offline CoSIL pilot selector now freezes ten patch-free evaluation cases
from ten distinct repositories. As of 2026-09-25, three of the ten base-source
archives were already materialized by the main RQ1 run; their audited CoSIL
structures were prepared without API calls. The other seven remain pending
source materialization. `pilot_readiness.json` records the fixed IDs and
readiness, not localization outcomes. A full pilot would require separate
model calls and should not be confused with the one authorized acquisition
smoke above.

## Execution update, 2026-09-26

The main RQ1 run subsequently materialized all ten frozen CoSIL pilot cases.
`prepare_cosil_pilot.py --materialize-ready` checked their base-source archive
hashes and prepared ten patch-free author structures. `run_cosil_pilot.py`
now runs CoSIL's file and function stages per case, normalizes only uniquely
identifiable base-source functions, and scores *after* author inference with
the frozen private labels. Its secure WSL launcher is
`scripts/start_cosil_pilot_secure.sh`; the live result is
`runs/external_fl_baseline_preflight_20260924/cosil_pilot_results_v2/comparison_summary.json`.
The earlier `cosil_pilot_results` and `cosil_pilot_results_venv` directories
are retained failed integration attempts (missing dependency and relative
output path, respectively), not localization results. Only the `_v2` run is
eligible for pilot reporting. The running pilot is not a 500-case baseline.

RGFL now has a separate WSL virtual environment. Its own source parser
prepared a patch-free base-source structure for `astropy__astropy-13033`.
The author file stage passed an offline mock, then a paid DeepSeek smoke.
That first paid response named plausible files, but the author's exact path
matcher rejected paths omitting a duplicated repository-root prefix. A
strict unique-suffix path adapter, with ambiguous matches excluded, produced
nonempty file candidates in a second paid smoke. The author element stage
then produced function candidates. Outputs remain under `rgfl_pilot/` and
are **integration smokes only**: RGFL's reasoning/reranking stages and the
shared 500-case evaluation have not run. The non-thinking inference and
path-normalization adapters must be disclosed in any comparison.

LocAgent still needs an isolated graph-index/runtime smoke; its PyTorch and
other graph dependencies are not present in the existing WSL environment.
SWE-Exp-Loc and RepoMem-Loc still require explicitly adapted FL-only
contracts. Neither has an evaluation result yet.

## CoSIL expansion, 2026-09-26

The first 10-case author run finished with nine normalized predictions and
one empty prediction. The empty case had four existing functions in the
author's raw XML response, but its exact file-path filter discarded them
because the XML omitted a duplicated repository-root prefix. A separate,
strict offline normalizer recovers only functions in uniquely matched
candidate source files. It does not infer functions from the patch or issue.
The original `comparison_summary.json` is preserved; the auditable
`comparison_summary_recovered.json` scores 10/10 completed, Top-1 7/10,
Top-5 8/10, and MRR 0.725. This is still a small pilot, not a 500-case
baseline result.

An additional 30-case frozen subset from the same evaluation manifest is
prepared at `cosil_pilot_30/`; all 30 base-source structures passed the
patch-free input audit. The 30-case runner verifies that the first ten
public inputs and source structures are byte-identical to the prior pilot,
then reuses their raw author outputs by recorded path and SHA-256. Only the
remaining twenty cases invoke the model. The new run is under
`cosil_pilot_30_results/` and uses the same disclosed non-thinking model
adapter plus strict raw-XML normalization. The full 500-case CoSIL run has
not yet been launched.

During the first live expansion attempt, the ten reused outputs were parsed
correctly but marked `invalid_author_output` because the destination folder
for `normalized.json` had not been created. This integration-only write-path
bug is fixed and covered by a reuse test. The in-flight summary is not a
valid score until `rescore_cosil_pilot.py` has verified the original author
output hashes and written a separate `comparison_summary_recovered.json`.
