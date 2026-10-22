| condition | HumanEval | custom-cpp | custom-verilog | ALL | LLM calls | wall min |
|---|---|---|---|---|---|---|
| (a) bare model, single-shot | 76.2% (61/80) | 40.0% (2/5) | 0.0% (0/5) | 70.0% (63/90) | 90 | 10.0 |
| (b) agent + sandbox repair | 75.0% (60/80) | 40.0% (2/5) | 80.0% (4/5) | 73.3% (66/90) | 114 | 10.8 |
| (c) agent + repair + experience library | 68.8% (55/80) | 40.0% (2/5) | 80.0% (4/5) | 67.8% (61/90) | 114 | 4572.7 |
| learning round (TRAIN split) | 69.0% (58/84) | 66.7% (2/3) | 66.7% (2/3) | 68.9% (62/90) | 116 | 21.0 |
