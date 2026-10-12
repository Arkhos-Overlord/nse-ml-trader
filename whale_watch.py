"""Whale watcher — pulls FII/DII flows + bulk deals via finstack MCP (NSE-direct).
Falls back gracefully. Emits a stable alert line for the cron change-detector."""
import asyncio, json, os, urllib.request
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PY = r"C:\Users\sydne\_assess\Kimi-Auto-trader\.venv\Scripts\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "whale_state.json")

async def call(s, name, args=None):
    try:
        r = await s.call_tool(name, args or {})
        return r.content[0].text if r.content else "{}"
    except Exception as e:
        return '{"err":"%s"}' % str(e)[:80]

async def main():
    p = StdioServerParameters(command=PY, args=["-m","finstack.server"])
    try:
        async with stdio_client(p) as (r,w):
            async with ClientSession(r,w) as s:
                await s.initialize()
                fii = await call(s,"nse_fii_dii_data",{})
                bulk = await call(s,"nse_bulk_deals",{})
    except Exception as e:
        print("WHALE_ERR:"+str(e)[:80]); return
    try: fj = json.loads(fii); bj = json.loads(bulk)
    except Exception: print("WHALE_PARSE_ERR"); return
    rows = fj.get("data",[])
    def _num(x):
        try: return float(str(x).replace(",",""))
        except (ValueError, TypeError): return None
    net = {r["category"]: _num(r.get("netValue")) for r in rows}
    date = rows[0].get("date") if rows else "?"
    nbulk = bj.get("count", 0) if isinstance(bj,dict) else 0
    # stable alert: only fires when FII/DII date or bulk count changes
    fii_v = net.get('FII/FPI'); dii_v = net.get('DII')
    alert = f"FII_DII:{date}:FII={fii_v}:DII={dii_v}:bulk={nbulk}"
    print(alert)
    json.dump({"ts": date, "fii_net": fii_v, "dii_net": dii_v,
               "bulk_deals": nbulk}, open(STATE,"w"))

asyncio.run(main())
