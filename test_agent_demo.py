"""
test_agent_demo.py — Offline demo / unit tests for AutoStream Agent

Tests the intent classifier logic and lead extraction without calling
the Anthropic API. Run with:
    python test_agent_demo.py
"""

import json
import sys
import os

# ── Test: Knowledge Base Loading ────────────────────────────────────────────

def test_kb_loads():
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "autostream_kb.json")
    with open(kb_path) as f:
        kb = json.load(f)

    assert kb["company"] == "AutoStream"
    assert len(kb["plans"]) == 2
    assert kb["plans"][0]["price"] == "$29/month"
    assert kb["plans"][1]["price"] == "$79/month"
    assert len(kb["policies"]) >= 3
    print("✅ test_kb_loads passed")


# ── Test: Lead Field Extraction ─────────────────────────────────────────────

# Import just the extraction helper (no LLM needed)
sys.path.insert(0, os.path.dirname(__file__))
from agent import extract_lead_fields

def test_extract_email():
    result = extract_lead_fields("My email is alice@example.com", {})
    assert result.get("email") == "alice@example.com"
    print("✅ test_extract_email passed")

def test_extract_platform():
    result = extract_lead_fields("I make videos on YouTube", {})
    assert result.get("platform") == "Youtube"
    print("✅ test_extract_platform passed")

def test_extract_name():
    result = extract_lead_fields("Jane Smith", {})
    assert result.get("name") == "Jane Smith"
    print("✅ test_extract_name passed")

def test_all_fields():
    lead = {}
    lead = extract_lead_fields("John Doe", lead)
    lead = extract_lead_fields("john.doe@email.com", lead)
    lead = extract_lead_fields("I'm on Instagram", lead)
    assert lead.get("name") == "John Doe"
    assert lead.get("email") == "john.doe@email.com"
    assert lead.get("platform") == "Instagram"
    print("✅ test_all_fields passed")


# ── Test: Mock Lead Capture ─────────────────────────────────────────────────

from agent import mock_lead_capture

def test_mock_lead_capture(capsys=None):
    result = mock_lead_capture("Jane Doe", "jane@test.com", "YouTube")
    assert "Jane Doe" in result
    assert "jane@test.com" in result
    assert "YouTube" in result
    print("✅ test_mock_lead_capture passed")


# ── Test: KB Context Builder ────────────────────────────────────────────────

from agent import build_kb_context

def test_kb_context():
    ctx = build_kb_context()
    assert "AutoStream" in ctx
    assert "$29/month" in ctx
    assert "$79/month" in ctx
    assert "4K" in ctx
    assert "AI captions" in ctx
    assert "No refunds" in ctx
    print("✅ test_kb_context passed")


# ── Run All ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AutoStream Agent — Offline Tests ===\n")
    test_kb_loads()
    test_extract_email()
    test_extract_platform()
    test_extract_name()
    test_all_fields()
    test_mock_lead_capture()
    test_kb_context()
    print("\n🎉 All tests passed!\n")
