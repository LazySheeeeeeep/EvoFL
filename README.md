# EvoluteFL v0

EvoluteFL is a skill-centric function-level fault localization prototype.
Explorer uses a fixed toolset: `grep`, `read_file`, and `write`.
Tools are not evolved.

EvoluteFL uses dimension-level localization skills.
A skill is not an executable workflow.
A skill is a reusable natural-language localization knowledge fragment associated with one dimension.

## Skill Dimensions

- `project_type`: project structure priors, common modules, entrypoints, boundaries, and intermediate representations.
- `fault_mode`: fault mechanisms, likely root-cause shapes, symptom patterns, and misleading locations.
- `strategy_type`: evidence interpretation strategies, candidate comparison, working/failing path contrast, caller/callee contrast, and symptom/cause distinction.
- `general`: general FL principles, root-cause ranking principles, and anti-misleading rules.

## Runtime Flow

```text
Case
-> dimension-aware skill retrieval
-> simple dimension-based context assembly
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

Reflector edits dimension-level skill text such as `skill.knowledge`, `skill.trigger`, `skill.anti_patterns`, or `retrieval_text`.
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
