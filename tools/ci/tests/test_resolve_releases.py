import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resolve_releases import (  # noqa: E402
    ReleaseResolutionError,
    list_releases,
    maafw_tag_to_pep440,
    select_asset,
    select_latest_release,
)


def release(
    release_id, published_at, *, draft=False, prerelease=False, assets=None
):
    return {
        "id": release_id,
        "tag_name": f"v{release_id}",
        "published_at": published_at,
        "draft": draft,
        "prerelease": prerelease,
        "assets": [{"name": name} for name in (assets or [])],
    }


class SelectLatestReleaseTest(unittest.TestCase):
    def test_includes_prerelease_and_excludes_draft(self):
        stable = release(1, "2026-01-01T00:00:00Z")
        prerelease = release(2, "2026-01-02T00:00:00Z", prerelease=True)
        draft = release(3, "2026-01-03T00:00:00Z", draft=True)

        self.assertIs(select_latest_release([stable, prerelease, draft]), prerelease)

    def test_uses_release_id_as_deterministic_tiebreaker(self):
        older_id = release(10, "2026-01-01T00:00:00Z")
        newer_id = release(11, "2026-01-01T00:00:00Z")

        self.assertIs(select_latest_release([newer_id, older_id]), newer_id)

    def test_rejects_empty_candidate_set(self):
        with self.assertRaises(ReleaseResolutionError):
            select_latest_release(
                [release(1, "2026-01-01T00:00:00Z", draft=True)]
            )


class ListReleasesTest(unittest.TestCase):
    @patch("resolve_releases.github_get_json")
    def test_reads_all_pages(self, github_get_json):
        github_get_json.side_effect = [
            [release(index, "2026-01-01T00:00:00Z") for index in range(30)],
            [release(30, "2026-01-02T00:00:00Z")],
        ]

        releases = list_releases("owner/repository", "token")

        self.assertEqual(len(releases), 31)
        self.assertEqual(github_get_json.call_count, 2)


class SelectAssetTest(unittest.TestCase):
    def test_returns_unique_match(self):
        item = release(
            1,
            "2026-01-01T00:00:00Z",
            assets=["MAA-win-x86_64-v1.0.0.zip", "source.zip"],
        )
        self.assertEqual(
            select_asset(item, "MAA-win-x86_64-*.zip"),
            "MAA-win-x86_64-v1.0.0.zip",
        )

    def test_rejects_missing_or_ambiguous_match(self):
        missing = release(1, "2026-01-01T00:00:00Z", assets=["source.zip"])
        duplicate = release(
            2,
            "2026-01-01T00:00:00Z",
            assets=["MXU-win-x86_64-a.zip", "MXU-win-x86_64-b.zip"],
        )

        with self.assertRaises(ReleaseResolutionError):
            select_asset(missing, "MAA-win-x86_64-*.zip")
        with self.assertRaises(ReleaseResolutionError):
            select_asset(duplicate, "MXU-win-x86_64-*.zip")


class MaafwVersionTest(unittest.TestCase):
    def test_converts_supported_tags(self):
        cases = {
            "v5.13.0": "5.13.0",
            "v5.13.0-alpha.1": "5.13.0a1",
            "v5.13.0-beta.5": "5.13.0b5",
            "v5.13.0-rc.2": "5.13.0rc2",
        }
        for tag, expected in cases.items():
            with self.subTest(tag=tag):
                self.assertEqual(maafw_tag_to_pep440(tag), expected)

    def test_rejects_unknown_or_noncanonical_tags(self):
        for tag in (
            "5.13.0",
            "v5.13",
            "v5.13.0-dev.1",
            "v5.13.0-beta1",
            "v05.13.0",
        ):
            with self.subTest(tag=tag):
                with self.assertRaises(ReleaseResolutionError):
                    maafw_tag_to_pep440(tag)


if __name__ == "__main__":
    unittest.main()
