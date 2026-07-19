# EvoluteFL v0

EvoluteFL is a skill-centric function-level fault localization prototype.
Explorer uses a fixed toolset: `grep`, `read_file`, and `write`.
Tools are not evolved.

EvoluteFL uses two types of localization skills.
A skill is not an executable workflow.
A skill is a reusable natural-language knowledge fragment for understanding a software system or narrowing the localization search space.

## Skill Types

- `project_skill`: reusable system-model fragments covering repository organization, component roles, functional relationships, and implementation boundaries.
- `strategy_skill`: reusable diagnostic policies covering evidence collection, tracing, comparison, hypothesis evaluation, and suspicious-function ranking.

## Runtime Flow

```text
Case
-> project/strategy skill retrieval
-> two-type context assembly
-> Explorer with fixed tools
-> ranked functions
```

Skills provide localization context, not commands.
Explorer remains free to search, inspect, and reason using fixed tools.
Repository evidence from tools has priority over retrieved skills.

## Evolution Flow

```text
Trajectory
-> outcome labeling (success or failure)
-> trajectory-driven Reflector
-> bounded text edits or no_update
-> SkillBank update
-> future retrieval
```

By default, successful completed cases and failed completed cases can both enter evolution consideration.
Success means completed + Top-5 hit; failure means completed + Top-5 miss.
Non-completed system failures are skipped unless `--force` is used.

Reflector consumes Explorer trajectory evidence directly, inspired by SkillOpt's trajectory-driven reflection style.
The legacy Insight path is still available with `--legacy-insight`.

Reflector evaluates project and strategy knowledge independently, then edits `skill.knowledge`, `skill.trigger`, or `retrieval_text`.
It does not create stages, workflows, tools, or prompt updates.

## CLI Examples

```powershell
$env:PYTHONPATH="src"
python -m evolutefl.cli show-tools
```

```powershell
$env:PYTHONPATH="src"
python -m evolutefl.cli search-skills-preview `
  --repo sqlfluff/sqlfluff `
  --issue-text "The --disable_progress_bar flag does not work in fix command."
```

```powershell
$env:PYTHONPATH="src"
python -m evolutefl.cli assemble-skills-preview --skill-ids fault_cli_option_compatibility_v1
```

```powershell
$env:PYTHONPATH="src"
python -m evolutefl.cli run-agent `
  --repo-path . `
  --repo demo/repo `
  --base-commit unknown `
  --instance-id demo__case `
  --issue-text "A CLI option is ignored." `
  --run-dir runs/demo_case
```

```powershell
$env:PYTHONPATH="src"
python -m evolutefl.cli run-case-evolution `
  --case-run-dir runs/demo_case `
  --repo demo/repo `
  --issue-text "A CLI option is ignored." `
  --ground-truth-patch-file patch.diff `
  --force
```

Use `--legacy-insight` to explicitly run the old Insight -> Reflector path.
Use `--no-reflect-success` or `--no-reflect-failure` to disable one outcome type.

## Tests

```powershell
$env:PYTHONPATH="src"
python -m pytest
```
