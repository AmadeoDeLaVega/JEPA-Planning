# Goal-photo sensitivity

Repeats the goal-photo intervention from the proposal (Section 5): how much
does the goal photograph change the model's choice when only the hand differs
between two goal photos and the puck pose is identical?

Script and full protocol: `code/experiments/goal_photo_sensitivity.py`.

## Setup

- Released DINO-WM Meta-World checkpoint, frozen. Cost: mean squared
  difference of the visual features (no proprio term).
- Meta-World `push-v3`, task seeds 58000-58015 (16 tasks), 128 candidates each:
  the expert's actions plus noise of increasing size.
- Goal photos L and R: same puck pose, hands 11.5-13.6 cm apart (mean 12.9 cm).
- Puck error is measured to the puck position in the goal photos.

## Result (CPU run, 2026-10-25)

| Endpoint features | Selection changed | Success changed | Mean puck-error change | Mean regret |
|---|---|---|---|---|
| Predicted | 7 / 16 | 3 / 16 | 1.62 cm | 3.35 cm |
| Actual | 13 / 16 | 8 / 16 | 3.70 cm | 2.80 cm |
| Pilot, predicted (proposal) | 10 / 16 | 5 / 16 | 1.62 cm | |
| Pilot, actual (proposal) | 12 / 16 | 7 / 16 | 2.42 cm | |

As in the pilot, the hand in the goal photo changes the selected candidate in
many tasks, and more often with actual than with predicted endpoints. The
counts are not expected to match the pilot exactly: the proposal does not
describe how the pilot's candidates and goal photos were made.

Outputs: `results/goal_photo_sensitivity/` (`tasks.csv`, `summary.json`, and
an L|R image per task). Not tracked by Git.
