# harness/aggregate.py

import json
from pathlib import Path
from collections import defaultdict

def main():
    summary_path = Path("outputs/summary.json")
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    wins = defaultdict(int)
    games = defaultdict(int)
    avg_score = defaultdict(list)

    for r in data:
        a = r["modelA"]
        b = r["modelB"]
        w = r["winner"]

        games[a] += 1
        games[b] += 1
        avg_score[a].append(r.get("total_A", 0))
        avg_score[b].append(r.get("total_B", 0))

        if w == "A":
            wins[a] += 1
        elif w == "B":
            wins[b] += 1

    print("=== Win rate ===")
    for m in sorted(games.keys()):
        wr = wins[m] / games[m] if games[m] else 0.0
        sc = sum(avg_score[m]) / len(avg_score[m]) if avg_score[m] else 0.0
        print(f"{m:30s}  win={wins[m]:3d}/{games[m]:3d}  winrate={wr:.2f}  avg_score={sc:.1f}")

if __name__ == "__main__":
    main()