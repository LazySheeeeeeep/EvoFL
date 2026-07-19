from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evolutefl.config import llm_config, load_config
from evolutefl.explorer import run_explorer
from evolutefl.json_utils import read_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.reflection import run_case_evolution
from evolutefl.skills import assemble_skill_context, make_skill_bank, render_skill_context
from evolutefl.tools import ToolRegistry, register_builtin_tools


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI should show a friendly single-line failure.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evolutefl")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show-tools", help="Show fixed Explorer tools.")
    show.set_defaults(func=cmd_show_tools)

    search = sub.add_parser("search-skills-preview", help="Preview skill retrieval and assembled localization context.")
    add_config_arg(search)
    add_embedding_override_args(search)
    search.add_argument("--repo", required=True)
    add_issue_args(search)
    search.set_defaults(func=cmd_search_skills_preview)

    rebuild = sub.add_parser("rebuild-skill-embeddings", help="Build or refresh cached embeddings for active skills.")
    add_config_arg(rebuild)
    add_embedding_override_args(rebuild)
    rebuild.set_defaults(func=cmd_rebuild_skill_embeddings)

    assemble = sub.add_parser("assemble-skills-preview", help="Preview project/strategy context for skill ids.")
    add_config_arg(assemble)
    assemble.add_argument("--skill-ids", required=True, help="Comma-separated skill ids.")
    assemble.set_defaults(func=cmd_assemble_skills_preview)

    compose = sub.add_parser("compose-skills-preview", help="Compatibility alias for assemble-skills-preview.")
    add_config_arg(compose)
    compose.add_argument("--skill-ids", required=True, help="Comma-separated skill ids.")
    compose.set_defaults(func=cmd_assemble_skills_preview)

    run_agent = sub.add_parser("run-agent", help="Run Explorer on one issue.")
    add_config_arg(run_agent)
    add_llm_args(run_agent)
    run_agent.add_argument("--repo-path", required=True)
    run_agent.add_argument("--repo", required=True)
    run_agent.add_argument("--base-commit", default="")
    run_agent.add_argument("--instance-id", required=True)
    add_issue_args(run_agent)
    run_agent.add_argument("--run-dir", required=True)
    run_agent.set_defaults(func=cmd_run_agent)

    evolve = sub.add_parser("run-case-evolution", help="Run trajectory-driven Reflector -> SkillBank update for one case.")
    add_config_arg(evolve)
    add_llm_args(evolve)
    add_embedding_override_args(evolve)
    evolve.add_argument("--case-run-dir", required=True)
    evolve.add_argument("--repo", required=True)
    add_issue_args(evolve)
    evolve.add_argument("--ground-truth-patch-file")
    evolve.add_argument("--ground-truth-functions-file")
    evolve.add_argument("--ground-truth-locations-file")
    evolve.add_argument("--force", action="store_true")
    evolve.add_argument("--legacy-insight", action="store_true", help="Use the legacy Insight -> Reflector path.")
    evolve.add_argument(
        "--refresh-issue-abstraction",
        action="store_true",
        help="Regenerate the retrieval abstraction with the current Abstractor prompt before reflection.",
    )
    evolve.add_argument("--reflect-success", dest="reflect_success", action="store_true", default=True)
    evolve.add_argument("--no-reflect-success", dest="reflect_success", action="store_false")
    evolve.add_argument("--reflect-failure", dest="reflect_failure", action="store_true", default=True)
    evolve.add_argument("--no-reflect-failure", dest="reflect_failure", action="store_false")
    evolve.add_argument("--output-dir")
    evolve.set_defaults(func=cmd_run_case_evolution)

    return parser


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/evolutefl.global.json")


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider")
    parser.add_argument("--model")


def add_issue_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--issue-text")
    group.add_argument("--issue-file")


def add_embedding_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enable-embedding", action="store_true", help="Enable configured embedding client for this command.")
    parser.add_argument("--retrieval-mode", choices=["lexical", "embedding", "hybrid"])
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-min-score", type=float)


def cmd_show_tools(_: argparse.Namespace) -> Any:
    registry = ToolRegistry()
    register_builtin_tools(registry)
    return registry.list_openai_tools()


def cmd_search_skills_preview(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    apply_embedding_overrides(config, args)
    issue = read_issue(args)
    bank = make_skill_bank(config)
    search = bank.search_for_explorer(args.repo, issue)
    assembled = assemble_skill_context(
        search["matched_skills"],
        config.get("assembler", {}).get("max_per_skill_type")
        or config.get("assembler", {}).get("max_per_dimension"),
        int(config.get("assembler", {}).get("max_knowledge_chars", 900)),
    )
    return {
        **search,
        "assembled_context": assembled,
        "rendered_context": render_skill_context(assembled) if search["matched_skills"] else "",
    }


def cmd_rebuild_skill_embeddings(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    apply_embedding_overrides(config, args)
    bank = make_skill_bank(config)
    return bank.rebuild_embeddings()


def cmd_assemble_skills_preview(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    wanted = {skill_id.strip() for skill_id in args.skill_ids.split(",") if skill_id.strip()}
    bank = make_skill_bank(config)
    skills = [skill.compact_dict() for skill in bank.active_skills() if skill.skill_id in wanted]
    assembled = assemble_skill_context(
        skills,
        config.get("assembler", {}).get("max_per_skill_type")
        or config.get("assembler", {}).get("max_per_dimension"),
        int(config.get("assembler", {}).get("max_knowledge_chars", 900)),
    )
    return {
        "matched_skills": skills,
        "assembled_context": assembled,
        "rendered_context": render_skill_context(assembled),
    }


def cmd_run_agent(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    config["llm"] = llm_config(config, args.provider, args.model)
    client = OpenAICompatibleClient.from_config(config["llm"])
    task = {
        "instance_id": args.instance_id,
        "repo_path": args.repo_path,
        "repo": args.repo,
        "base_commit": args.base_commit,
        "bug_report": read_issue(args),
        "run_dir": args.run_dir,
    }
    return run_explorer(task=task, config=config, llm_client=client)


def cmd_run_case_evolution(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    apply_embedding_overrides(config, args)
    config["llm"] = llm_config(config, args.provider, args.model)
    client = OpenAICompatibleClient.from_config(config["llm"])
    return run_case_evolution(
        case_run_dir=args.case_run_dir,
        repo=args.repo,
        issue=read_issue(args),
        config=config,
        llm_client=client,
        force=args.force,
        ground_truth_patch=read_optional_text(args.ground_truth_patch_file),
        ground_truth_functions=read_optional_list(args.ground_truth_functions_file),
        ground_truth_locations=read_optional_json_list(args.ground_truth_locations_file),
        output_dir=args.output_dir,
        reflect_success=args.reflect_success,
        reflect_failure=args.reflect_failure,
        legacy_insight=args.legacy_insight,
        refresh_issue_abstraction=args.refresh_issue_abstraction,
    )


def read_issue(args: argparse.Namespace) -> str:
    if getattr(args, "issue_file", None):
        return Path(args.issue_file).read_text(encoding="utf-8")
    return str(args.issue_text or "")


def apply_embedding_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    embedding = config.setdefault("embedding", {})
    skill_bank = config.setdefault("skill_bank", {})
    if getattr(args, "enable_embedding", False):
        embedding["enabled"] = True
    if getattr(args, "retrieval_mode", None):
        skill_bank["retrieval_mode"] = args.retrieval_mode
    if getattr(args, "embedding_base_url", None):
        embedding["base_url"] = args.embedding_base_url
        embedding["enabled"] = True
    if getattr(args, "embedding_min_score", None) is not None:
        skill_bank["embedding_min_score"] = args.embedding_min_score


def read_optional_text(path: str | None) -> str:
    return Path(path).read_text(encoding="utf-8") if path else ""


def read_optional_json_list(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    value = read_json(path)
    if isinstance(value, list):
        return value
    raise ValueError(f"Expected JSON list in {path}")


def read_optional_list(path: str | None) -> list[str]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [str(item) for item in value]
    return [line.strip() for line in text.splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
