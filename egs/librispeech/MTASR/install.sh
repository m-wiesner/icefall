#!/usr/bin/env bash
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${recipe_dir}"

if command -v uv >/dev/null 2>&1; then
  uv_bin=$(command -v uv)
elif [[ -x "${recipe_dir}/.tools/uv" ]]; then
  uv_bin="${recipe_dir}/.tools/uv"
else
  installer=$(mktemp)
  trap 'rm -f "${installer}"' EXIT
  curl -LsSf https://astral.sh/uv/install.sh -o "${installer}"
  UV_INSTALL_DIR="${recipe_dir}/.tools" sh "${installer}"
  uv_bin="${recipe_dir}/.tools/uv"
fi

# Keep downloads out of potentially small home directories. Callers may
# override this on clusters with a preferred node-local cache.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${recipe_dir}/.uv-cache}"
unset VIRTUAL_ENV || true
"${uv_bin}" sync --locked

# shellcheck disable=SC1091
source "${recipe_dir}/activate.sh"
python "${recipe_dir}/smoke_test.py"
echo "Environment ready. Run: source ${recipe_dir}/activate.sh"
