#!/usr/bin/env bash
# Keep a public Pinggy tunnel (SSH wrapped in TLS on 443) to the chat UI on 127.0.0.1:8765.
# Free tunnels expire after ~60 min and get a NEW URL on each reconnect; this loop reconnects and
# rewrites .tunnel_url each time. Only port 8765 (password-protected UI) is forwarded.
cd "$(dirname "$0")/.."
KEY="${SIC_TUNNEL_KEY:-/workspace/dolphin/.tunnel_key}"
KH="${SIC_TUNNEL_KNOWN_HOSTS:-/workspace/dolphin/.known_hosts}"
mkdir -p logs
while true; do
  echo "$(date -u +%FT%TZ) connecting" >> logs/tunnel.log
  : > logs/tunnel.cur
  ssh -p 443 -T \
    -o ProxyCommand="openssl s_client -quiet -connect a.pinggy.io:443 -servername a.pinggy.io" \
    -o IdentityAgent=none -o IdentitiesOnly=yes -i "$KEY" \
    -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" \
    -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes \
    -R0:127.0.0.1:8765 a.pinggy.io < /dev/null > logs/tunnel.cur 2>&1 &
  pid=$!
  for i in $(seq 1 40); do
    URL=$(tr '\r' '\n' < logs/tunnel.cur | grep -aoE 'https://[a-z0-9.-]+\.free\.pinggy\.net' | head -1)
    [ -n "$URL" ] && { echo "$URL" > .tunnel_url.tmp; chmod 600 .tunnel_url.tmp; mv .tunnel_url.tmp .tunnel_url
                       echo "$(date -u +%FT%TZ) up: $URL" >> logs/tunnel.log; break; }
    sleep 1
  done
  wait $pid
  echo "$(date -u +%FT%TZ) ssh exited rc=$?; restarting in 5s" >> logs/tunnel.log
  sleep 5
done
