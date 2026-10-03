from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from scripts.run_swe_explore_explorer_compare import _download_github_archive


class ArchiveDownloadRetryTests(TestCase):
    def test_curl_uses_bounded_retries_for_slow_official_archive(self):
        with patch("scripts.run_swe_explore_explorer_compare.shutil.which",
                   return_value="/usr/bin/curl"), patch(
            "scripts.run_swe_explore_explorer_compare.run"
        ) as execute:
            _download_github_archive(
                "https://codeload.github.com/example/repo/tar.gz/abc",
                Path("archive.download"),
            )
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index("--max-time") + 1], "600")
        self.assertEqual(command[command.index("--retry") + 1], "2")
        self.assertIn("--retry-all-errors", command)
        self.assertEqual(execute.call_args.kwargs["timeout"], 1830)
