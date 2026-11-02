#!/bin/zsh
# Commit and push every 5 minutes until the given HHMM. Stops rather than pushing anything that looks like a secret.
cd "$(dirname "$0")/.." || exit 1
UNTIL="${1:-1645}"
SECRET='fw_[A-Za-z0-9]{10,}|wandb_v1_[A-Za-z0-9_]{20,}'
while [ "$(date +%H%M)" -lt "$UNTIL" ]; do
  git add -A
  if ! git diff --cached --quiet; then
    if git diff --cached --name-only | grep -qE '(^|/)\.env$' || git diff --cached -U0 -a | grep -qE "$SECRET"; then
      git reset -q
      echo "$(date +%H:%M:%S) STOPPED: possible secret in staged changes, nothing committed"
      exit 2
    fi
    # Files GitHub would reject stay local, and are reported.
    large=$(git diff --cached --name-only --diff-filter=AM | while read -r f; do [ -f "$f" ] && [ "$(stat -f%z "$f")" -gt 90000000 ] && echo "$f"; done)
    if [ -n "$large" ]; then
      echo "$large" | while read -r f; do git reset -q -- "$f"; echo "$(date +%H:%M:%S) skipped large file $f"; done
    fi
    git commit -q -m "Progress $(date +%H:%M): warehouse simulation and training" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    git push -q 2>&1 | tail -1
    echo "$(date +%H:%M:%S) pushed $(git rev-parse --short HEAD)"
  else
    echo "$(date +%H:%M:%S) no changes"
  fi
  python3 -c "import time; time.sleep(300)"
done
echo "$(date +%H:%M:%S) finished"
