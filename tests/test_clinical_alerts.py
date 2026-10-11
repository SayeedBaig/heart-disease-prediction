"""
Unit tests for Clinical Alert Engine (clinical_module/alerts/clinical_alert_engine.py).
Tests standalone functionality without requiring database or full API dependencies.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clinical_module.alerts.clinical_alert_engine import (
    evaluate_clinical_alerts,
    alerts_to_dicts,
    THRESHOLDS,
)


class TestClinicalAlertEngine(unittest.TestCase):

    def test_low_systolic_bp_alert(self):
        alerts = evaluate_clinical_alerts(
            systolic_bp=85,
            diastolic_bp=65,
            is_hypothetical=False,
            context_id="test_low_sys",
        )
        alert_ids = [a.alert_id for a in alerts]
        self.assertIn("LOW_SYSTOLIC_BP", alert_ids)
        self.assertNotIn("LOW_DIASTOLIC_BP", alert_ids)
        self.assertNotIn("HIGH_SYSTOLIC_BP", alert_ids)
        low_alert = next(a for a in alerts if a.alert_id == "LOW_SYSTOLIC_BP")
        self.assertFalse(low_alert.is_hypothetical)
        self.assertIn("below the configured adult screening threshold", low_alert.explanation)
        self.assertIn("Do not change prescribed medication", low_alert.recommendation)

    def test_low_diastolic_bp_alert(self):
        alerts = evaluate_clinical_alerts(
            systolic_bp=110,
            diastolic_bp=55,
            is_hypothetical=False,
            context_id="test_low_dia",
        )
        alert_ids = [a.alert_id for a in alerts]
        self.assertIn("LOW_DIASTOLIC_BP", alert_ids)
        self.assertNotIn("LOW_SYSTOLIC_BP", alert_ids)
        low_dia = next(a for a in alerts if a.alert_id == "LOW_DIASTOLIC_BP")
        self.assertEqual(low_dia.measured_value, 55)

    def test_high_bp_alerts(self):
        alerts = evaluate_clinical_alerts(
            systolic_bp=150,
            diastolic_bp=95,
            is_hypothetical=False,
            context_id="test_high_bp",
        )
        alert_ids = [a.alert_id for a in alerts]
        self.assertIn("HIGH_SYSTOLIC_BP", alert_ids)
        self.assertIn("HIGH_DIASTOLIC_BP", alert_ids)

    def test_normal_bp_no_bp_alerts(self):
        alerts = evaluate_clinical_alerts(
            systolic_bp=120,
            diastolic_bp=80,
            glucose=90,
            cholesterol_category=1,
            is_hypothetical=False,
        )
        self.assertEqual(len(alerts), 0)

    def test_hypothetical_flag_in_digital_twin(self):
        alerts = evaluate_clinical_alerts(
            systolic_bp=80,
            diastolic_bp=50,
            is_hypothetical=True,
            context_id="scenario_weight_loss",
        )
        self.assertTrue(len(alerts) >= 1)
        for a in alerts:
            self.assertTrue(a.is_hypothetical)
            self.assertEqual(a.context_id, "scenario_weight_loss")
            self.assertIn("simulated", a.explanation.lower())

    def test_alerts_to_dicts_serialisation(self):
        alerts = evaluate_clinical_alerts(
            systolic_bp=85,
            diastolic_bp=55,
            is_hypothetical=True,
        )
        dicts = alerts_to_dicts(alerts)
        self.assertIsInstance(dicts, list)
        self.assertEqual(len(dicts), len(alerts))
        self.assertIn("alert_id", dicts[0])
        self.assertIn("threshold_rule", dicts[0])
        self.assertIn("is_hypothetical", dicts[0])


if __name__ == "__main__":
    unittest.main()
