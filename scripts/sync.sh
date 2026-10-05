#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cmd="${1:-}"
if [[ "$cmd" != "pull" && "$cmd" != "push-lab" ]]; then
  echo "usage: $0 pull|push-lab" >&2
  exit 2
fi

assert_no_avi_staged() {
  if git diff --cached --name-only | grep -qiE '\.avi$'; then
    echo "Refusing: a staged path ends in .avi. Raw recordings must not be committed." >&2
    exit 1
  fi
}

if [[ "$cmd" == "pull" ]]; then
  git fetch
  if ! git pull --ff-only; then
    echo "git pull --ff-only failed. Stop. Do not rebase or force. Resolve with Sai." >&2
    exit 1
  fi
  echo "pull ok (fast-forward only)"
  exit 0
fi

# push-lab: markdown notes, evidence, writeup — not lab/tools
mapfile -t md < <(find lab -maxdepth 1 -type f -name '*.md' 2>/dev/null || true)
to_add=("${md[@]+"${md[@]}"}")
if [[ -d lab/evidence ]]; then
  to_add+=("lab/evidence")
fi
if [[ -d writeup ]]; then
  to_add+=("writeup")
fi

if [[ ${#to_add[@]} -eq 0 ]]; then
  echo "nothing to stage for push-lab"
  exit 0
fi

git add -- "${to_add[@]}"
assert_no_avi_staged

if [[ -z "$(git diff --cached --name-only)" ]]; then
  echo "no lab-note changes to push"
  exit 0
fi

git commit -m "lab: sync notebook"
git push
echo "push-lab ok"
