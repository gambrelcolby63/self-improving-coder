#!/usr/bin/env bash
# Creates the non-privileged "sicagent" user the coding agent runs as, and
# installs a uid-scoped nftables table: the agent may reach the public internet
# (pip install, git clone) but NOT loopback, RFC1918 / CGNAT / link-local ranges
# or cloud metadata (169.254.169.254). DNS to the configured resolver is allowed.
# Requires sudo once. Idempotent (the table is replaced on each run).
set -euo pipefail
AGENT_USER="${AGENT_USER:-sicagent}"
if ! id "$AGENT_USER" >/dev/null 2>&1; then
  sudo useradd -m -d "/home/$AGENT_USER" -s /bin/bash -U "$AGENT_USER"
fi
sudo chmod 700 "/home/$AGENT_USER"
sudo -u "$AGENT_USER" mkdir -p "/home/$AGENT_USER/ws"
UID_NUM=$(id -u "$AGENT_USER")
DNS_ALLOW=""
for ns in $(awk '/^nameserver/{print $2}' /etc/resolv.conf | grep -v ':' || true); do
  DNS_ALLOW+="    meta skuid $UID_NUM ip daddr $ns udp dport 53 accept
    meta skuid $UID_NUM ip daddr $ns tcp dport 53 accept
"
done
sudo nft delete table inet sic_agent 2>/dev/null || true
sudo nft -f - <<NFT
table inet sic_agent {
  chain out {
    type filter hook output priority 0; policy accept;
$DNS_ALLOW    meta skuid $UID_NUM ip daddr { 0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10, 127.0.0.0/8, 169.254.0.0/16, 172.16.0.0/12, 192.168.0.0/16, 224.0.0.0/4, 240.0.0.0/4 } reject
    meta skuid $UID_NUM ip6 daddr { ::1/128, fc00::/7, fe80::/10, ::ffff:0:0/96 } reject
  }
}
NFT
echo "sandbox user $AGENT_USER ready (uid $UID_NUM); nft table inet sic_agent installed"
