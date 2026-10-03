import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from io import BytesIO
import os
import subprocess
import tarfile
from unittest.mock import patch

import networkx as nx

from scripts.normalize_locagent_functions import normalize, source_functions
from scripts.run_locagent_live_smoke import load_public_case
from scripts.run_locagent_rq1_pilot import GraphBuildTimeout, build_graph_bounded


class LocAgentNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.MultiDiGraph()
        self.graph.add_node("pkg/a.py:Widget.run", type="function", start_line=10, end_line=20)
        self.graph.add_node("pkg/a.py:Other.run", type="function", start_line=30, end_line=40)
        self.graph.add_node("pkg/a.py:helper", type="function", start_line=50, end_line=60)

    def test_qualified_method_and_unique_line_scope(self):
        answer = "```\npkg/a.py\nclass: Widget\nfunction: run\n\npkg/a.py\nline: 52\n```"
        result = normalize(answer, self.graph)
        self.assertEqual(result["ranked_functions"], ["pkg/a.py::Widget.run", "pkg/a.py::helper"])

    def test_ambiguous_simple_name_is_not_resolved(self):
        result = normalize("```\npkg/a.py\nfunction: run\n```", self.graph)
        self.assertEqual(result["ranked_functions"], [])
        self.assertEqual(result["mappings"][0]["status"], "unresolved")

    def test_unknown_file_is_not_promoted_to_function(self):
        result = normalize("```\npkg/missing.py\nfunction: helper\n```", self.graph)
        self.assertEqual(result["ranked_functions"], [])

    def test_frozen_input_excludes_patch_and_ground_truth(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)
            case = {"instance_id": "repo__case-1", "repo": "repo/case", "base_commit": "abc",
                    "problem_statement": "Nonempty issue", "patch": "secret patch",
                    "function_ground_truth": ["pkg/a.py::helper"]}
            (source / "evaluation_cases.json").write_text(json.dumps([case]), encoding="utf-8")
            public = load_public_case(case["instance_id"], source)
        self.assertEqual(set(public), {"instance_id", "repo", "base_commit", "problem_statement"})

    def test_base_source_restores_constructor_omitted_by_author_graph(self):
        answer = "```\npkg/a.py\nclass: Widget\nfunction: __init__\n```"
        with TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.tar.gz"
            code = b"class Widget:\n    def __init__(self):\n        self.value = 1\n"
            member = tarfile.TarInfo("repo-sha/pkg/a.py")
            member.size = len(code)
            with tarfile.open(archive, "w:gz") as contents:
                contents.addfile(member, BytesIO(code))
            extra = source_functions(archive, answer)
        result = normalize(answer, self.graph, extra_functions=extra)
        self.assertEqual(result["ranked_functions"], ["pkg/a.py::Widget.__init__"])

    def test_graph_timeout_is_distinct_and_child_does_not_receive_api_key(self):
        with TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}):
                with patch("scripts.run_locagent_rq1_pilot.subprocess.run",
                           side_effect=subprocess.TimeoutExpired("graph", 2)) as runner:
                    with self.assertRaises(GraphBuildTimeout):
                        build_graph_bounded("repo__case-1", Path(temporary), 2)
            self.assertNotIn("DEEPSEEK_API_KEY", runner.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
