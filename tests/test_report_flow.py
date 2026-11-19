import json
import unittest
from io import BytesIO
from unittest.mock import patch

from dr7_client import draft_report
from reporting import build_report


SAMPLE_RESULT = {
    "predictions": [
        {"class": "cataract", "score": 0.7},
        {"class": "glaucoma", "score": 0.2},
        {"class": "normal", "score": 0.1},
    ],
    "raw_softmax_scores": [
        {"class": "cataract", "score": 0.8},
        {"class": "glaucoma", "score": 0.15},
        {"class": "normal", "score": 0.05},
    ],
    "score_semantics": "Temperature-scaled model scores; not clinical probabilities.",
    "calibration": {"status": "internal dataset only", "temperature": 1.5},
    "metadata": {
        "model": "test model",
        "classes_in_logit_order": ["cataract", "glaucoma", "normal"],
        "input_image_size_px": [512, 512],
        "checkpoint_sha256": "abc123",
        "training_evaluation": {"final_test_accuracy": 0.85},
    },
    "limitations": ["Research use only."],
}


class ReportFlowTests(unittest.TestCase):
    def test_deterministic_report_discloses_missing_facts_and_uncertainty(self):
        text = build_report(SAMPLE_RESULT, {})
        for heading in (
            "1. Examination information",
            "2. CAD findings",
            "3. Predicted disease and confidence score",
            "4. Interpretation",
            "5. Recommendations for further evaluation",
        ):
            self.assertIn(heading, text)
        self.assertIn("Not provided.", text)
        self.assertIn("not establish or exclude disease", text)
        self.assertIn("not clinical probabilities", text)
        self.assertIn("patient-level independence is unknown", text)
        self.assertIn("label-quality concern", text)
        self.assertIn("Glaucoma: 20.0%", text)

    def test_dr7_payload_includes_full_output_but_no_image_identifiers_or_date(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data)
            captured["url"] = request.full_url
            return BytesIO(json.dumps({"choices": [{"message": {"content": "unverified draft"}}]}).encode())

        exam = {
            "date": "2026-01-02",
            "modality": "fundus",
            "laterality": "Left",
            "device": "device serial 123",
        }
        with patch("urllib.request.urlopen", fake_urlopen):
            draft = draft_report(SAMPLE_RESULT, exam, "test-key", "medgemma-4b-it")

        payload = json.loads(captured["body"]["messages"][1]["content"])
        self.assertEqual(draft, "unverified draft")
        self.assertIn("complete_cad_output", payload)
        self.assertEqual(payload["complete_cad_output"], SAMPLE_RESULT)
        self.assertFalse(payload["image_transmitted"])
        self.assertEqual(payload["examination_information"], {"modality": "fundus", "laterality": "Left"})
        serialized = json.dumps(payload)
        self.assertNotIn("2026-01-02", serialized)
        self.assertNotIn("device serial", serialized)
        self.assertNotIn("test-key", serialized)
        self.assertEqual(captured["url"], "https://dr7.ai/api/v1/medical/chat/completions")


if __name__ == "__main__":
    unittest.main()
