#!/usr/bin/env bash
# One-time + repeatable deploy of deploy/modal_llm.py. Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in env.
# Never prints the API key: it is passed to Modal via a mode-600 temp dotenv file.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODAL_TOKEN_ID:?set MODAL_TOKEN_ID}" "${MODAL_TOKEN_SECRET:?set MODAL_TOKEN_SECRET}"
MODAL="${MODAL:-$(command -v modal || echo ~/.local/bin/modal)}"
if [ ! -s .modal_api_key ]; then
  (umask 077; python3 -c "import secrets;print(secrets.token_urlsafe(32))" > .modal_api_key)
fi
chmod 600 .modal_api_key
tmp=$(mktemp); chmod 600 "$tmp"; trap 'rm -f "$tmp"' EXIT
printf 'VLLM_API_KEY=%s\n' "$(cat .modal_api_key)" > "$tmp"
"$MODAL" secret create sic-llm-key --from-dotenv "$tmp" --force > /dev/null
echo "secret sic-llm-key updated"
"$MODAL" deploy deploy/modal_llm.py | tee logs/modal_deploy.log
url=$(grep -oE 'https://[a-z0-9-]+--sic-llm-chat[a-z0-9-]*\.modal\.run' logs/modal_deploy.log | head -1)
if [ -n "$url" ]; then
  echo "$url" > .modal_url; chmod 600 .modal_url
  echo "chat endpoint saved to .modal_url: $url"
  echo "enable with: SIC_BACKEND=modal (restart sic.server)"
fi
