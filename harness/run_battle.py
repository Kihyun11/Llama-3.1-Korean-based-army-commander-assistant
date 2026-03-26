
# harness/run_battle.py

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time

from harness.prompts import SYSTEM_GUARD, ROUND0, ROUND1, ROUND2
from harness.local_models import build_default_models, LocalHFModel

# OpenAI judge
from harness.judge_openai import make_client, judge_with_gpt


def load_scenarios(jsonl_path: str) -> List[Dict[str, Any]]:
    scenarios = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))
    return scenarios


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_log(a0: str, a1: str, a2: str) -> str:
    
    return (
        "=== ROUND 0 ===\n" + a0.strip() + "\n\n"
        "=== ROUND 1 ===\n" + a1.strip() + "\n\n"
        "=== ROUND 2 ===\n" + a2.strip()
    )


def battle_one(
    scenario_obj: Dict[str, Any],
    modelA: LocalHFModel,
    modelB: LocalHFModel,
    max_new_tokens_round0: int = 700,
    max_new_tokens_round1: int = 450,
    max_new_tokens_round2: int = 512,
    # max_new_tokens_round0: int = 10,
    # max_new_tokens_round1: int = 10,
    # max_new_tokens_round2: int = 10,
    judge_enabled: bool = True,
    judge_model: str = "gpt-5-mini",
    judge_temperature: float = 0.0,
) -> Dict[str, Any]:

    sid = scenario_obj["id"]
    scenario = scenario_obj["scenario_description"]

    out_dir = Path("outputs") / sid / f"{modelA.name}_vs_{modelB.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"[SCENARIO] {sid}")
    print(f"[MATCHUP ] {modelA.name}  vs  {modelB.name}")
    print("=" * 90)

    # ---- Round 0
    print("[ROUND0] building prompts...")
    a0_prompt = ROUND0(system_guard=SYSTEM_GUARD, scenario=scenario)
    b0_prompt = ROUND0(system_guard=SYSTEM_GUARD, scenario=scenario)

    t0 = time.time()
    print("[ROUND0] A generating...")
    a0 = modelA.generate(a0_prompt, max_new_tokens=max_new_tokens_round0)
    modelA.unload()

    print("[ROUND0] B generating...")
    b0 = modelB.generate(b0_prompt, max_new_tokens=max_new_tokens_round0)
    modelB.unload()
    print(f"[ROUND0] done in {time.time()-t0:.2f}s")

    write_text(out_dir / "A_round0.txt", a0)
    write_text(out_dir / "B_round0.txt", b0)

    # ---- Round 1
    print("[ROUND1] building prompts...")
    a1_prompt = ROUND1(system_guard=SYSTEM_GUARD, scenario=scenario, opp_round0=b0)
    b1_prompt = ROUND1(system_guard=SYSTEM_GUARD, scenario=scenario, opp_round0=a0)

    t1 = time.time()
    print("[ROUND1] A generating...")
    a1 = modelA.generate(a1_prompt, max_new_tokens=max_new_tokens_round1)
    modelA.unload()

    print("[ROUND1] B generating...")
    b1 = modelB.generate(b1_prompt, max_new_tokens=max_new_tokens_round1)
    modelB.unload()
    print(f"[ROUND1] done in {time.time()-t1:.2f}s")

    write_text(out_dir / "A_round1.txt", a1)
    write_text(out_dir / "B_round1.txt", b1)

    # ---- Round 2
    print("[ROUND2] building prompts...")
    a2_prompt = ROUND2(system_guard=SYSTEM_GUARD, scenario=scenario, opp_round1=b1)
    b2_prompt = ROUND2(system_guard=SYSTEM_GUARD, scenario=scenario, opp_round1=a1)

    t2 = time.time()
    print("[ROUND2] A generating...")
    a2 = modelA.generate(a2_prompt, max_new_tokens=max_new_tokens_round2)
    modelA.unload()

    print("[ROUND2] B generating...")
    b2 = modelB.generate(b2_prompt, max_new_tokens=max_new_tokens_round2)
    modelB.unload()
    print(f"[ROUND2] done in {time.time()-t2:.2f}s")

    write_text(out_dir / "A_round2.txt", a2)
    write_text(out_dir / "B_round2.txt", b2)

    # ---- Judge (OpenAI)
    a_log = _build_log(a0, a1, a2)
    b_log = _build_log(b0, b1, b2)

    judge_result: Dict[str, Any]
    if judge_enabled:
        try:
            print(f"[JUDGE ] calling {judge_model} ...")
            client = make_client()
            judged = judge_with_gpt(
                client=client,
                judge_model=judge_model,
                scenario=scenario,
                a_log=a_log,
                b_log=b_log,
                #temperature=judge_temperature,
            )
            judge_result = {
                "scenario_id": sid,
                "modelA": modelA.name,
                "modelB": modelB.name,
                "judge_model": judge_model,
                **judged,
            }
        except Exception as e:
            
            judge_result = {
                "scenario_id": sid,
                "modelA": modelA.name,
                "modelB": modelB.name,
                "judge_model": judge_model,
                "winner": "N/A",
                "note": f"Judge failed: {type(e).__name__}: {e}",
            }
    else:
        judge_result = {
            "scenario_id": sid,
            "modelA": modelA.name,
            "modelB": modelB.name,
            "winner": "N/A",
            "note": "Judge disabled by config.",
        }

    (out_dir / "judge.json").write_text(
        json.dumps(judge_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return judge_result


def main():
    #SCENARIOS_PATH = "scenario1.jsonl"
    #SCENARIOS_PATH = "scenario2.jsonl"
    #SCENARIOS_PATH = "scenario3.jsonl"
    SCENARIOS_PATH = "scenario4.jsonl"

    MATCHUPS: List[Tuple[str, str]] = [
        #("kanana_1.5_8b_base", "Phi-3.5-mini-instruct"),
        #("kanana_1.5_8b_base", "Yi_1.5_9B_chat"),
        #("kanana_1.5_8b_base", "Deepseek_R1_Distill_Qwen_7B"),
        #("Phi-3.5-mini-instruct", "Yi_1.5_9B_chat"),
        #("Phi-3.5-mini-instruct", "Deepseek_R1_Distill_Qwen_7B"),
        ("Yi_1.5_9B_chat", "Deepseek_R1_Distill_Qwen_7B"),
    ]

    #
    TOKENS_R0, TOKENS_R1, TOKENS_R2 = 700, 450, 512
    

    #
    JUDGE_ENABLED = True
    JUDGE_MODEL = "gpt-5-mini"
    JUDGE_TEMPERATURE = 0.0

    scenarios = load_scenarios(SCENARIOS_PATH)
    models = build_default_models()

    results = []
    for s in scenarios[:1]:
        for a_name, b_name in MATCHUPS:
            res = battle_one(
                scenario_obj=s,
                modelA=models[a_name],
                modelB=models[b_name],
                max_new_tokens_round0=TOKENS_R0,
                max_new_tokens_round1=TOKENS_R1,
                max_new_tokens_round2=TOKENS_R2,
                judge_enabled=JUDGE_ENABLED,
                judge_model=JUDGE_MODEL,
                judge_temperature=JUDGE_TEMPERATURE,
            )
            results.append(res)

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Done. Wrote outputs/summary.json")


if __name__ == "__main__":
    main()





