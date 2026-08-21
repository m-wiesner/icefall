#!/usr/bin/env bash
# Source this file from the recipe directory after `uv sync`.
shuffle_recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shuffle_icefall_root=$(cd -- "${shuffle_recipe_dir}/../../.." && pwd)
export PATH="${shuffle_recipe_dir}/.venv/bin:${PATH}"
export PYTHONPATH="${shuffle_recipe_dir}:${shuffle_recipe_dir}/wavlm_spashl_ctc:${shuffle_icefall_root}${PYTHONPATH:+:${PYTHONPATH}}"
unset shuffle_recipe_dir shuffle_icefall_root
