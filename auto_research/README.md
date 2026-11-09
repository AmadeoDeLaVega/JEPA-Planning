# JEPA research scratchpads

Two additive notebooks consolidate eight original research notebooks. The original team proposal receives credit for the rank-32 contrastive adapter. No original team file or research evidence was changed; no new experiment was run.

## Run

Keep the five core files together: the two notebooks, evidence.zip, provenance.json and this README. Open either notebook from the repository root or from auto_research and Run All with Python 3.10, NumPy, pandas and Matplotlib. Notebook 2 does not depend on notebook 1. Jupyter/IPython is needed for the notebook interface, not for the numerical analysis. No model/checkpoint, simulator, local jepa_planning imports or network access is used. The notebooks read the ZIP directly, verify hashes, and write nothing.

- 01_goal_geometry_and_metrics.ipynb: quotient and photo control; protected metric development failure; fresh original-adapter confirmation.
- 02_search_and_gradients.ipynb: optimizer audit; near-hold and planner-visited gradients; CEM confirmation; inconclusive continuous transfer; unopened factorial pilot.

## Evidence and scope

The ZIP is a selective portable evidence subset. Its paths are the exact original repository-relative paths. provenance.json maps every copied file to its original path, SHA-256 and byte size, records the eight source notebook hashes, and hashes the ZIP. Original source bytes are preserved, including historical path strings in frozen receipts; these strings are provenance, not runtime dependencies. Original notebooks are not copied, so this folder contains exactly two notebooks.

Reports and protocol/manifest receipts are preserved alongside tables and compact raw arrays. Large latent caches, model weights, fitted model matrices, archived implementations, videos and redundant plots are omitted. In particular the adapter/protected analyses recompute outcomes from complete selection tables rather than rerunning fitting or rescoring latent features. The quotient, CEM and gradient analyses additionally check saved physical labels or secants against raw arrays. Optimizer and engineering outcomes are explicitly historical receipts, not re-executed model or simulator tests.

Read studies separately: absolute errors use differing target cohorts and must not form a leaderboard. Bank/task pairing is retained; fixed calibration heads are averaged within tasks, never treated as independent tasks. Undefined cosines remain missing with denominators shown. No combined learned-metric plus adaptive-search gain is established.

Validation details and dependency versions are recorded in validation.json after execution. Stored notebook outputs are actual computed results. Validation uses in-process IPython, not a launched Jupyter kernel or a model-replication run. Base repository revision: d07cd7878cb40ecbd9e1004ef7db6794eb08d1df.
