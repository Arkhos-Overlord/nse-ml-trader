"""INDmoney MCP client — direct tool calls for research.

Uses the OAuth token Hermes cached at ~/AppData/Local/hermes/mcp-tokens/indmoney.json
(scope includes market:read). Talks JSON-RPC to https://mcp.indmoney.com/mcp.

Purpose: pull CLEAN intraday OHLC candles (1min..1month) for Nifty/indices/futures
— the synchronized data that no keyless source provided. Then re-run the
mean-reversion backtest honestly.

Run: python indmoney_client.py test
"""
import json
import os
import urllib.request
import urllib.error

TOKEN_PATH = os.path.expandvars(
    r"C:\Users\sydne\AppData\Local\hermes\mcp-tokens\indmoney.json")
URL = "https://mcp.indmoney.com/mcp"
PROTO = "2025-06-18"


def _load_token():
    with open(TOKEN_PATH) as f:
        return json.load(f)["access_token"]


class Ind:
    def __init__(self):
        self._id = 0
        self._tok = None
        self._sess = None

    def _rpc(self, method, params=None, notify=False):
        tok = self._tok or _load_token()
        body = {"jsonrpc": "2.0", "method": method}
        if not notify:
            self._id += 1
            body["id"] = self._id
        if params is not None:
            body["params"] = params
        req = urllib.request.Request(
            URL, data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        req.add_header("Authorization", f"Bearer {tok}")
        if self._sess:
            req.add_header("Mcp-Session-Id", self._sess)
        if self.protocol:
            req.add_header("MCP-Protocol-Version", PROTO)
        try:
            r = urllib.request.urlopen(req, timeout=90)
            self._sess = r.headers.get("Mcp-Session-Id") or self._sess
            raw = r.read().decode()
            # streamable-http may return SSE
            if raw.startswith("event:") or "\ndata:" in raw:
                for line in raw.splitlines():
                    if line.startswith("data:"):
                        return json.loads(line[5:].strip())
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")

    protocol = PROTO

    def init(self):
        r = self._rpc("initialize", {
            "protocolVersion": PROTO,
            "capabilities": {},
            "clientInfo": {"name": "hermes-research", "version": "1.0"},
        })
        self.protocol = r.get("result", {}).get("protocolVersion", PROTO)
        self._rpc("notifications/initialized", notify=True)
        return r

    def tools(self):
        return self._rpc("tools/list").get("result", {}).get("tools", [])

    def call(self, name, args):
        r = self._rpc("tools/call", {"name": name, "arguments": args})
        if "error" in r:
            raise RuntimeError(f"tool error: {r['error']}")
        res = r.get("result", {})
        # content blocks -> text
        out = []
        for c in res.get("content", []):
            if c.get("type") == "text":
                out.append(c["text"])
        txt = "\n".join(out)
        try:
            return json.loads(txt)
        except (json.JSONDecodeError, TypeError):
            return txt


def test():
    c = Ind()
    c.init()
    tools = c.tools()
    print(f"{len(tools)} tools available")
    for t in tools:
        name = t.get("name", "")
        if any(k in name for k in ["ohlc", "lookup", "option_chain", "greeks"]):
            print(f"  {name}: {(t.get('description') or '')[:100]}")
    return c


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        c = test()
        # try resolving Nifty
        try:
            r = c.call("lookup_ind_keys", {"query": "NIFTY 50"})
            print("\nlookup NIFTY 50 ->", json.dumps(r, indent=2)[:600])
        except Exception as e:  # noqa: BLE001
            print("\nlookup failed:", e)
