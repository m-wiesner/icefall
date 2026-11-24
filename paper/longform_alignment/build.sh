#!/usr/bin/env bash
set -euo pipefail

paper_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir="$paper_dir/source"
build_dir="$paper_dir/build"

for command in pdflatex bibtex kpsewhich; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 1
  fi
done

missing=0
for package in multirow.sty shuffle.sty algorithm2e.sty stmaryrd.sty \
  subcaption.sty pgfplots.sty; do
  if ! kpsewhich "$package" >/dev/null 2>&1; then
    echo "Missing LaTeX package: $package" >&2
    missing=1
  fi
done
if [[ "$missing" == 1 ]]; then
  echo "Install a complete TeX Live distribution and retry." >&2
  exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/longform-paper.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT
cp -a "$source_dir/." "$work_dir/"

(
  cd "$work_dir"
  pdflatex -interaction=nonstopmode -halt-on-error Template.tex
  bibtex Template
  pdflatex -interaction=nonstopmode -halt-on-error Template.tex
  pdflatex -interaction=nonstopmode -halt-on-error Template.tex
)

mkdir -p "$build_dir"
cp "$work_dir/Template.pdf" "$build_dir/Longform_Multi_Talker_Alignment.pdf"
echo "Built $build_dir/Longform_Multi_Talker_Alignment.pdf"
