# TACTIC-L2L

**Tactical LLM-to-LLM Battle Evaluation Framework**

TACTIC-L2L pits two local language models against each other in a structured 3-round wargame debate over fictional Korean military tactical scenarios. An OpenAI GPT judge then evaluates both transcripts and produces a scored verdict.

![TACTIC-L2L Framework](figure/TACTIC-L2L%20framework.png)

---

## How It Works

Each battle consists of three rounds between Model A and Model B:

| Round | Task |
|-------|------|
| **Round 0** | Both models independently analyze the scenario: list assumptions, constraints, Blue/Red COA options, and top risks. |
| **Round 1** | Each model reads the opponent's Round 0, identifies 3 logical weaknesses, and revises its own plan. |
| **Round 2** | Each model reads the opponent's Round 1 and produces a final Blue/Red plan with trade-offs and conditions that would change its assessment. |

After all three rounds, an OpenAI GPT judge scores both transcripts across 7 rubric criteria (each 0–5):

| Criterion | Description |
|-----------|-------------|
| `assumptions` | Explicit, plausible, well-scoped assumptions |
| `constraints` | Awareness of real operational constraints |
| `consistency` | Internal logical coherence across rounds |
| `coa_quality` | Quality and balance of COA options |
| `risk_second_order` | Identification of second-order and cascading risks |
| `interaction` | Quality of adversarial reasoning against the opponent |
| `clarity` | Clear, professional prose without degenerate formatting |

The winner is the model with the strictly higher total score (max 35). Exact ties are recorded as `"Tie"`.

---

## Repository Structure

```
TACTIC-L2L/
├── figure/
│   └── TACTIC-L2L framework.png   # Framework diagram
├── harness/
│   ├── __init__.py
│   ├── run_battle.py               # Main battle loop (entry point)
│   ├── prompts.py                  # Round 0/1/2 prompt templates + system guard
│   ├── judge_openai.py             # GPT judge client and scoring logic
│   ├── local_models.py             # HuggingFace model loader / generation wrapper
│   └── aggregate.py                # Win-rate aggregation from outputs/summary.json
└── scenarios/
    ├── scenario1.jsonl             # KR_URBAN_0001 — Seoul suburban urban combat
    ├── scenario2.jsonl             # KR_MOUNTAIN_CORRIDOR_0002 — Bukhan River mountain corridor
    ├── scenario3.jsonl             # KR_DMC_ESCALATION_0003 — DMZ western corridor escalation
    └── scenario4.jsonl             # KR_REAR_AREA_DISRUPTION_0004 — Daejeon logistics corridor
```

---

## Scenarios

All scenarios are set in Year 2025 Korea and are purely fictional, for simulation/training analysis only.

| ID | Location | Key Challenge |
|----|----------|---------------|
| `KR_URBAN_0001` | Satellite city north of Seoul | Urban combat, two bridges, large civilian presence, 72h engagement |
| `KR_MOUNTAIN_CORRIDOR_0002` | Bukhan River basin eastern corridor | Fog, GPS jamming, dam threat, interdicted MSR |
| `KR_DMC_ESCALATION_0003` | Western DMZ corridor near Paju | Escalation constraints, cyber grid attack, US alliance coordination |
| `KR_REAR_AREA_DISRUPTION_0004` | Daejeon logistics corridor | SOF drone swarms, rail cyber sabotage, sustainment protection |

---

## Supported Models

Configured in `harness/local_models.py`:

| Key | HuggingFace Model ID |
|-----|----------------------|
| `Deepseek_R1_Distill_Qwen_7B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| `Phi-3.5-mini-instruct` | `microsoft/Phi-3.5-mini-instruct` |
| `Yi_1.5_9B_chat` | `01-ai/Yi-1.5-9B-Chat` |
| `kanana_1.5_8b_base` | `kakaocorp/kanana-1.5-8b-base` |

Models are loaded one at a time and unloaded after each round to manage GPU memory.

---

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU recommended.

```bash
pip install torch transformers openai
```

Set your OpenAI API key (used by the GPT judge):

```bash
export OPENAI_API_KEY="sk-..."
```

Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FORCE_CUDA` | `0` | Set to `1` to force CUDA and raise an error if unavailable |
| `HF_HUB_OFFLINE` | `0` | Set to `1` to load models from local cache only |

---

## Running a Battle

Edit the `main()` function in `harness/run_battle.py` to select a scenario and matchup, then run:

```bash
python -m harness.run_battle
```

Key configuration options inside `main()`:

```python
SCENARIOS_PATH = "scenarios/scenario4.jsonl"   # Which scenario file to load

MATCHUPS = [
    ("Yi_1.5_9B_chat", "Deepseek_R1_Distill_Qwen_7B"),
]

TOKENS_R0, TOKENS_R1, TOKENS_R2 = 700, 450, 512  # Max new tokens per round

JUDGE_ENABLED  = True
JUDGE_MODEL    = "gpt-4o-mini"   # OpenAI model used as judge
```

### Outputs

Each battle writes its results under `outputs/<scenario_id>/<ModelA>_vs_<ModelB>/`:

```
outputs/
└── KR_REAR_AREA_DISRUPTION_0004/
    └── Yi_1.5_9B_chat_vs_Deepseek_R1_Distill_Qwen_7B/
        ├── A_round0.txt
        ├── B_round0.txt
        ├── A_round1.txt
        ├── B_round1.txt
        ├── A_round2.txt
        ├── B_round2.txt
        └── judge.json          # Scores, winner, reasoning, safety flag
```

A combined `outputs/summary.json` is written at the end of each run.

---

## Aggregating Results

After one or more runs, compute win rates and average scores:

```bash
python -m harness.aggregate
```

Example output:

```
=== Win rate ===
Deepseek_R1_Distill_Qwen_7B    win=  3/  5  winrate=0.60  avg_score=22.4
Yi_1.5_9B_chat                 win=  2/  5  winrate=0.40  avg_score=20.1
```

---

## Safety

All prompts include a system guard that instructs models to stay at high-level strategic principles and explicitly prohibits real-world actionable operational instructions, targeting details, or step-by-step tactics. The judge also applies a `safety_flag` field to each result.
