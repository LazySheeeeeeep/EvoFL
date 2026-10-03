from unittest import TestCase

from scripts.normalize_cosil_functions import normalize


class NormalizeCosilFunctionsTests(TestCase):
    def test_resolves_unique_method_and_strips_one_repo_root(self):
        source = ["class Client:", "    def send(self):", "        pass", "",
                  "def parse(self):", "    pass"]
        structure = {"example": {"pkg": {"api.py": {
            "classes": [], "functions": [], "text": source,
        }}}}
        row = {"instance_id": "example__case-1", "found_related_locs": {
            "example/pkg/api.py": ["function: send\nfunction: parse\nclass: Client"]}}
        result = normalize(row, structure)
        self.assertEqual(result["ranked_functions"], [
            "pkg/api.py::Client.send", "pkg/api.py::parse",
        ])

    def test_ambiguous_symbol_does_not_gain_function_credit(self):
        source = ["class A:", "    def run(self):", "        pass", "class B:",
                  "    def run(self):", "        pass"]
        structure = {"example": {"pkg": {"api.py": {
            "classes": [], "functions": [], "text": source,
        }}}}
        row = {"instance_id": "example__case-2", "found_related_locs": {
            "pkg/api.py": ["function: run"]}}
        result = normalize(row, structure)
        self.assertEqual(result["ranked_functions"], ["pkg/api.py::__unresolved__.run"])
        self.assertEqual(result["mappings"][0]["status"], "unresolved")

    def test_recovers_author_xml_only_for_candidate_existing_function(self):
        source = ["class Client:", "    def send(self):", "        pass"]
        structure = {"example": {"pkg": {"api.py": {
            "classes": [], "functions": [], "text": source,
        }}}}
        row = {
            "instance_id": "example__case-3",
            "found_files": ["example/pkg/api.py"],
            "found_related_locs": {"example/pkg/api.py": [""]},
            "func_traj": {"response": "<locations>"
                          "<location><file>pkg/api.py</file><type>function</type>"
                          "<name>Client.send</name></location>"
                          "<location><file>pkg/api.py</file><type>function</type>"
                          "<name>Client.not_there</name></location>"
                          "<location><file>other.py</file><type>function</type>"
                          "<name>send</name></location></locations>"},
        }
        result = normalize(row, structure)
        self.assertEqual(result["ranked_functions"], ["pkg/api.py::Client.send"])
        self.assertEqual(result["mappings"][0]["status"], "raw_xml_recovered_exact")

    def test_recovery_does_not_bypass_candidate_file_list(self):
        structure = {"example": {"pkg": {"api.py": {
            "classes": [], "functions": [], "text": ["def send():", "    pass"],
        }}}}
        row = {
            "instance_id": "example__case-4",
            "found_files": ["example/other.py"],
            "found_related_locs": {},
            "func_traj": {"response": "<locations><location><file>pkg/api.py</file>"
                          "<type>function</type><name>send</name></location></locations>"},
        }
        self.assertEqual(normalize(row, structure)["ranked_functions"], [])
