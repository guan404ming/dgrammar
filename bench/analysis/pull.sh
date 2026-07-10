#!/bin/zsh
# pull.sh <remote_dir> <local_dir>  -- `modal volume get` silently no-ops on dirs; fetch per file.
# `modal volume get <f> -` also writes its success banner to stdout, so keep only JSON lines.
VOL=dgrammar-results
REMOTE=$1
LOCAL=$2
mkdir -p "$LOCAL"
python3 -m modal volume ls $VOL "$REMOTE" 2>/dev/null | grep '\.jsonl$' | while read -r rf; do
  bn=$(basename "$rf")
  [ -s "$LOCAL/$bn" ] && continue
  python3 -m modal volume get $VOL "$rf" - 2>/dev/null | grep '^{' > "$LOCAL/$bn"
done
echo "$LOCAL: $(cat "$LOCAL"/*.jsonl 2>/dev/null | wc -l | tr -d ' ') records"
