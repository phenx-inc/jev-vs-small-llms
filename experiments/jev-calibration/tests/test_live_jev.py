import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_live_jev import (  # noqa: E402
    discover_models,
    load_secret,
    select_model,
    select_smoke_rows,
    selection_sha256,
    validate_systemone_response,
)
from analyze_live_jev import audit_raw_responses, canonical_response_sha256  # noqa: E402


class LiveJevTest(unittest.TestCase):
    def test_loads_nonstandard_existing_key_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text("jev_api='secret-value'\n", encoding="utf-8")
            self.assertEqual(load_secret(path), "secret-value")

    def test_prefers_versioned_model_over_latest_alias(self):
        models = [
            {"name": "jev-latest", "release_date": "2026-09-15"},
            {"name": "jev-2026-09-15", "release_date": "2026-09-15"},
            {"name": "jev-2026-08-01", "release_date": "2026-08-01"},
        ]
        self.assertEqual(select_model(models, "auto"), "jev-2026-09-15")

    def test_smoke_selection_is_deterministic_and_covers_labels(self):
        rows = []
        for benchmark in ("a", "b", "c"):
            for index in range(10):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "id": str(index),
                        "label": index % 2,
                    }
                )
        first = select_smoke_rows(rows, 12, 7)
        second = select_smoke_rows(rows, 12, 7)
        self.assertEqual(selection_sha256(first), selection_sha256(second))
        self.assertEqual(len(first), 12)
        for benchmark in ("a", "b", "c"):
            labels = {row["label"] for row in first if row["benchmark"] == benchmark}
            self.assertEqual(labels, {0, 1})

    def test_rendered_state_limit_can_be_applied_without_truncation(self):
        from rlcd_protocol import render_state

        rows = [
            {"state": {"message": "short"}},
            {"state": {"message": "x" * 4100}},
        ]
        eligible = [row for row in rows if len(render_state(row["state"])) <= 4000]
        self.assertEqual(len(eligible), 1)
        self.assertEqual(render_state(eligible[0]["state"]), "### message\nshort")

    def test_raw_response_shape_is_state_free(self):
        response = {
            "model": "jev-1.13.0",
            "answers": {"q": {"type": "noul", "noul": 0.7}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "meta": {"latency_ms": 100},
        }
        self.assertFalse({"state", "questions"} & set(response))

    def test_raw_response_audit_links_exact_response(self):
        response = {
            "model": "jev-1.13.0",
            "answers": {"misbehaves_noul": {"type": "noul", "noul": 0.7}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "meta": {"latency_ms": 100},
        }
        digest = canonical_response_sha256(response)
        prediction = {
            "benchmark": "b",
            "id": "1",
            "live_jev_probability_yes": 0.7,
            "input_tokens": 12,
            "output_tokens": 0,
            "raw_response_sha256": digest,
        }
        envelope = {
            "benchmark": "b",
            "id": "1",
            "response_sha256": digest,
            "response": response,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.jsonl"
            path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
            audit = audit_raw_responses(path, [prediction])
        self.assertTrue(audit["response_hashes_verified"])

    def test_response_validation(self):
        response = {
            "model": "jev-2026-09-15",
            "answers": {"misbehaves_noul": {"type": "noul", "noul": 0.73}},
            "usage": {"input_tokens": 120, "output_tokens": 4},
        }
        self.assertEqual(
            validate_systemone_response(response), (0.73, 120, 4, "jev-2026-09-15")
        )
        bad = json.loads(json.dumps(response))
        bad["answers"]["misbehaves_noul"]["noul"] = 2
        with self.assertRaisesRegex(ValueError, "out_of_range"):
            validate_systemone_response(bad)

    @mock.patch("run_live_jev.http_json")
    def test_model_discovery_requires_nonempty_valid_metadata(self, http_json):
        http_json.return_value = (
            200,
            {
                "models": [
                    {
                        "name": "jev-latest",
                        "description": "General-purpose",
                        "release_date": "2026-09-15",
                    }
                ]
            },
        )
        self.assertEqual(
            discover_models("https://api.typesafe.ai", "secret", 1)[0]["name"],
            "jev-latest",
        )

    @mock.patch("run_live_jev.urllib.request.urlopen")
    def test_http_client_uses_early_access_compatible_user_agent(self, urlopen):
        from run_live_jev import http_json

        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = b'{"ok": true}'
        urlopen.return_value.__enter__.return_value = response
        http_json(url="https://example.test", key="secret")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.headers["User-agent"], "curl/8.7.1")


if __name__ == "__main__":
    unittest.main()
