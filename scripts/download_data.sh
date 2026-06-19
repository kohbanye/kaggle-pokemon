#!/usr/bin/env bash
# Download Pokemon TCG AI Battle competition data into ./data
# Prereq: accept the competition rules on the website first, otherwise you get 403:
#   https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules
set -euo pipefail

COMP="pokemon-tcg-ai-battle"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DEST"

echo "==> Downloading lightweight files (CSV card data + sample submission code)"
# Card data
kaggle competitions download "$COMP" -f EN_Card_Data.csv -p "$DEST"
kaggle competitions download "$COMP" -f JP_Card_Data.csv -p "$DEST"

# Sample submission bundle (the agent template + cg simulator)
for f in \
  sample_submission/main.py \
  sample_submission/deck.csv \
  sample_submission/cg/__init__.py \
  sample_submission/cg/api.py \
  sample_submission/cg/game.py \
  sample_submission/cg/sim.py \
  sample_submission/cg/utils.py \
  sample_submission/cg/cg.dll \
  sample_submission/cg/libcg.so ; do
  kaggle competitions download "$COMP" -f "$f" -p "$DEST/$(dirname "$f")"
done

echo "==> Unzipping any *.zip in $DEST"
find "$DEST" -name '*.zip' -print -exec sh -c 'unzip -o "$1" -d "$(dirname "$1")" && rm "$1"' _ {} \;

echo "==> The card-image PDFs are large (137MB / 182MB). Download them only if you want them:"
echo "    kaggle competitions download $COMP -f 'Card_ID List_EN.pdf' -p $DEST"
echo "    kaggle competitions download $COMP -f 'Card_ID List_JP.pdf' -p $DEST"

echo "==> Done. Contents:"
find "$DEST" -type f | sort
