#!/usr/bin/env bash
# Builds the documentation and copies it into the Portfolio project, which serves it at
# https://fannar.fernandocsdm.com/docs/ (see middleware.js there). Commit and deploy the
# Portfolio afterwards.
#
#   bash scripts/publish_docs.sh [path/to/Portfolio]     # default: ../Portfolio
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
portfolio="${1:-${PORTFOLIO_DIR:-$root/../Portfolio}}"
dest="$portfolio/public/fannar/docs"
[ -f "$portfolio/middleware.js" ] || { echo "Portfolio project not found at $portfolio" >&2; exit 1; }

cd "$root"
python="python3"
[ -x .venv/bin/python ] && python=".venv/bin/python"

NO_MKDOCS_2_WARNING=1 "$python" -m mkdocs build --strict --clean

# What the site does not use: source maps and the search stemmers of other languages.
find build/docs -name '*.map' -delete
find build/docs/assets/javascripts/lunr -type f \
  ! -name 'lunr.stemmer.support.min.js' ! -name 'lunr.multi.min.js' -delete
# The site's Content-Security-Policy has no 'unsafe-inline' for scripts.
"$python" scripts/externalize_inline_scripts.py build/docs

rm -rf "$dest"
mkdir -p "$dest"
cp -r build/docs/. "$dest/"
echo "documentation copied to $dest ($(du -sh "$dest" | cut -f1))"
