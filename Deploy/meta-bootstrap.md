# WAN deployment bootstrap (CGNAT edition): Tailscale Funnel + meta.json

This walkthrough puts a public DAIN coordinator behind a home internet
connection that has **no port forwarding** (CGNAT is fine), with no credit
card and no domain purchase. Clients and nodes discover the coordinator
through one small published JSON file, so migrating the coordinator between
machines is a one-command operation.

## 1. Coordinator machine (your Debian PC / laptop)

```bash
# Tailscale: outbound-only tunnel, stable public HTTPS URL, no ports opened.
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up                       # log in with Google/GitHub — no card
tailscale funnel 8000              # publishes https://<machine>.<tailnet>.ts.net -> localhost:8000
tailscale serve status             # confirm; survives reboots automatically
```

- No updater exists or is needed: the ts.net name is owned by Tailscale's
  DNS, and the connection is *outbound*, so home-IP churn and reboots are
  invisible to clients. TLS is issued and renewed automatically.
- Coordinator must listen on port 8000 (`DAIN_PORT=8000`) — the funnel
  forwards to `localhost:8000`.

## 2. Coordinator config (keys + model store)

```bash
cd Coordinator
# config.json is created on first run; set real secrets:
#   join_token / api_key / admin_api_key  (harden_production_secrets would
#   randomize them otherwise, since the host is not loopback)
uv run python -m dain_coordinator
# verify (from another network, e.g. your phone):
#   curl https://<machine>.<tailnet>.ts.net/healthz
```

Keep `DAIN_DISCOVERY_ENABLED=false` for a WAN coordinator (LAN-only UDP), and
set `DAIN_TRUSTED_PROXIES=127.0.0.1` (default) so rate limiting sees real
client IPs through the funnel.

**When migrating the coordinator to another machine**, copy `config.json`
(identical keys — otherwise nodes cannot re-authenticate) and `model_store/`
(the shards). The SQLite registry may stay behind: registered nodes self-heal
via the join token, and the assignment plan is rebuilt.

## 3. Publish the metadata (the single source of truth)

`Client/public/meta.json` is served by the Pages site at `/DAIN/meta.json`
and is fetched by every client on open and by every node at startup and on
each reconnect (URL only — **never put the node join token in this file**):

```json
{
  "coord_url": "https://<machine>.<tailnet>.ts.net",
  "api_key": "dain-client",
  "model_id": "llama-3-2-1b-int4s"
}
```

```powershell
powershell -File Scripts\publish-meta.ps1 `
  -CoordUrl https://<machine>.<tailnet>.ts.net `
  -ApiKey dain-client -ModelId llama-3-2-1b-int4s -Push
```

**Coordinator migration runbook**: stop on the old machine → start + funnel on
the new one → rerun `publish-meta -Push`. Clients adopt the new URL the next
time they open; running nodes self-migrate on their next reconnect (they
re-read the meta document every attempt).

## 4. Nodes

```bash
# config.json (node): join_token matches the coordinator; coord_url may stay
# empty when meta_url is set — it is resolved at startup.
{
  "join_token": "<coordinator's join token>",
  "meta_url": "https://velmet44.github.io/DAIN/meta.json",
  "cache_dir": "/path/to/shard_cache"
}
```

Home-LAN nodes reach the coordinator directly over the LAN (discovery or a
`http://<lan-ip>:8000` coord_url wins by being faster; set `coord_url`
explicitly for them). WAN nodes use the funnel URL. Nodes behind NAT should
keep `DAIN_DIRECT_RELAY=0` unless they share a LAN with their pipeline
neighbors.

## 5. GitHub Pages client

Set the repo variable `DAIN_API_URL=https://<machine>.<tailnet>.ts.net` and
secret `DAIN_API_KEY` (Settings → Secrets and variables → Actions) so the
deploy workflow's guard passes; the client additionally self-updates from
`meta.json` on every open.

## Limits to know

- Funnel traffic is gently resource-limited by Tailscale and relayed — fine
  for token streams and occasional shard provisioning.
- The ts.net URL is tied to the *machine* (its Tailscale node name); two
  coordinator machines have two URLs, which is exactly what the meta.json
  swap handles.
