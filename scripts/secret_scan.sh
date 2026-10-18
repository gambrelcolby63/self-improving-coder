#!/usr/bin/env bash
# Fails if tracked/staged files contain likely secrets or the chat password. Prints file names only.
set -uo pipefail
cd "$(dirname "$0")/.."
PW_FILE="${SIC_PASSWORD_FILE:-$HOME/.config/sic/chat_password}"
files=$(git ls-files -co --exclude-standard)
bad=0
pat='(ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
for f in $files; do
  [ -f "$f" ] || continue
  if grep -IEq "$pat" "$f"; then echo "possible secret in: $f"; bad=1; fi
  if [ -f "$PW_FILE" ] && grep -Fq -f "$PW_FILE" "$f"; then echo "chat password found in: $f"; bad=1; fi
done
echo "$files" | grep -Eq '(chat_password|\.password$|hosts\.yml|\.env$)' && { echo "sensitive filename tracked"; bad=1; }
[ $bad -eq 0 ] && echo "secret scan: clean ($(echo "$files" | wc -l) files)"
exit $bad
