#!/usr/bin/env bash
# Publish the DAIN coordinator metadata (S23). See publish-meta.ps1.
#   ./Scripts/publish-meta.sh https://box.tail-scale.ts.net dain-client llama-3-2-1b-int4s [--push]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
coord="${1:?usage: publish-meta.sh <coord_url> [api_key] [model_id] [--push]}"
api="${2:-}"
model="${3:-}"
push="${4:-}"
path="$root/Client/public/meta.json"
cat > "$path" << JSON
{
  "coord_url": "$coord",
  "api_key": "$api",
  "model_id": "$model"
}
JSON
echo "meta.json written: $path"
cat "$path"
if [ "$push" = "--push" ]; then
  git -C "$root" add Client/public/meta.json
  git -C "$root" commit -m "meta: coordinator at $coord"
  git -C "$root" push
  echo "Pushed - the Pages site redeploys (~1-2 min)."
fi
