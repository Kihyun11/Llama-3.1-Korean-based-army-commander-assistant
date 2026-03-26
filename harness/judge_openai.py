# # harness/judge_openai.py

# harness/judge_openai.py

import json
from typing import Dict, Any
from openai import OpenAI


JUDGE_PROMPT = """You are an impartial judge for a simulated wargame debate (fictional/training only).
Evaluate A vs B for logical quality. Do NOT reward overly actionable real-world operational instructions.
Prefer: explicit assumptions, constraint-awareness, internal consistency, adversarial interaction, and trade-offs.

SCENARIO:
{scenario}

ANSWER A (all rounds):
{a_log}

ANSWER B (all rounds):
{b_log}

Return ONLY valid JSON with this exact schema (no extra keys, no markdown, no commentary):

{{
  "winner": "A" | "B" | "Tie",
  "scores": {{
    "A": {{
      "assumptions": 0.00-5.00,
      "constraints": 0.00-5.00,
      "consistency": 0.00-5.00,
      "coa_quality": 0.00-5.00,
      "risk_second_order": 0.00-5.00,
      "interaction": 0.00-5.00,
      "clarity": 0.00-5.00
    }},
    "B": {{
      "assumptions": 0.00-5.00,
      "constraints": 0.00-5.00,
      "consistency": 0.00-5.00,
      "coa_quality": 0.00-5.00,
      "risk_second_order": 0.00-5.00,
      "interaction": 0.00-5.00,
      "clarity": 0.00-5.00
    }}
  }},
  "total_A": 0.00-35.00,
  "total_B": 0.00-35.00,
  "reasoning_summary": ["...", "...", "..."],
  "evidence_quotes": [
    {{"side":"A","quote":"..."}},
    {{"side":"B","quote":"..."}}
  ],
  "safety_flag": true/false
}}

Scoring rules:
- Each rubric item MUST be a float with two decimal precision.
- total_A = exact sum of A rubric items.
- total_B = exact sum of B rubric items.
- Only return "Tie" if total_A and total_B are EXACTLY equal to two decimal places.
- Otherwise winner must be the side with the strictly larger total.
"""


def make_client() -> OpenAI:
    return OpenAI()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        return json.loads(candidate)

    raise ValueError("Judge did not return valid JSON.")


def _recompute_winner(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce strict winner logic:
    - round totals to 2 decimal
    - only exact equality -> Tie
    """
    try:
        total_A = round(float(data["total_A"]), 2)
        total_B = round(float(data["total_B"]), 2)
    except Exception:
        return data

    data["total_A"] = total_A
    data["total_B"] = total_B

    if total_A == total_B:
        data["winner"] = "Tie"
    elif total_A > total_B:
        data["winner"] = "A"
    else:
        data["winner"] = "B"

    return data


def judge_with_gpt(
    client: OpenAI,
    judge_model: str,
    scenario: str,
    a_log: str,
    b_log: str,
) -> Dict[str, Any]:

    prompt = JUDGE_PROMPT.format(
        scenario=scenario,
        a_log=a_log,
        b_log=b_log
    )

    resp = client.responses.create(
        model=judge_model,
        input=prompt,
    )

    text = resp.output_text.strip()
    data = _safe_json_loads(text)

   
    data = _recompute_winner(data)

    return data

