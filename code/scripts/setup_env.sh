#!/usr/bin/env bash
set -euo pipefail

# Creates the single project .venv (PushT + Meta-World) from requirements/project.lock.
readonly repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly local_uv="$repo_root/.tools/uv/bin/uv"

if [[ -n "${UV_BIN:-}" ]]; then
  uv_bin="$UV_BIN"
elif [[ -x "$local_uv" ]]; then
  uv_bin="$local_uv"
elif command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
else
  echo "uv is required; install uv 0.9.4 or set UV_BIN explicitly." >&2
  exit 1
fi

readonly uv_bin
export UV_CACHE_DIR="$repo_root/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$repo_root/.tools/python"

echo "Using $($uv_bin --version) from $uv_bin"
"$uv_bin" python install 3.10.19 --managed-python
"$uv_bin" venv "$repo_root/.venv" --python 3.10.19 --managed-python --seed
"$uv_bin" pip install --python "$repo_root/.venv/bin/python" \
  --requirements "$repo_root/requirements/project.lock" \
  --require-hashes \
  --torch-backend cu126
"$uv_bin" pip install --python "$repo_root/.venv/bin/python" --no-deps \
  --editable "$repo_root/third_party/jepa-wms"

"$repo_root/.venv/bin/python" "$repo_root/code/scripts/check_pusht_env.py"
MUJOCO_GL="${MUJOCO_GL:-egl}" "$repo_root/.venv/bin/python" "$repo_root/code/scripts/check_metaworld_env.py"
