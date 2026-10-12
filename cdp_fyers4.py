"""Reach the real API app dashboard and dump the create-app form."""
import asyncio
import json

import websockets

MY_IP = "106.210.180.193"
REDIRECT = "http://127.0.0.1:8081/callback"


async def get_ws():
    import urllib.request
    tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=6))
    return next((t for t in tabs if t.get("type") == "page"), None)["webSocketDebuggerUrl"]


async def send(ws, method, params=None, _id=[0]):
    _id[0] += 1
    await ws.send(json.dumps({"id": _id[0], "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == _id[0]:
            return msg.get("result", {})


async def evaluate(ws, expr):
    r = await send(ws, "Runtime.evaluate",
                   {"expression": expr, "returnByValue": True, "awaitPromise": True})
    if r.get("exceptionDetails"):
        return {"__error": str(r["exceptionDetails"].get("text"))}
    return r.get("result", {}).get("value")


async def dump(ws, tag, all_inputs=False):
    vis = "" if all_inputs else ".filter(e => e.vis)"
    expr = """
      (() => JSON.stringify({
        url: location.href,
        inputs: [...document.querySelectorAll('input,textarea')].map((el,i)=>({
          i, type: el.type, name: el.name, id: el.id, ph: el.placeholder,
          val: el.value, vis: !!(el.offsetParent || el.offsetHeight)
        }))""" + vis + """,
        btns: [...document.querySelectorAll('button,a')].map((el,i)=>({
          i, text:(el.innerText||'').trim().slice(0,50), tag: el.tagName
        })).filter(e=>e.text)
      }))()
    """
    info = await evaluate(ws, expr)
    print(f"\n=== {tag} ===")
    try:
        s = json.loads(info)
        print("url:", s["url"])
        print("-- INPUTS --")
        for el in s["inputs"]:
            print(f"  [{el['i']}] {el['type']:9} name={el['name']!r} id={el['id']!r} ph={el['ph']!r} val={el['val']!r}")
        print("-- BUTTONS/LINKS --")
        for el in s["btns"]:
            print(f"  [{el['i']}] <{el['tag']}> {el['text']!r}")
        return s
    except Exception:
        print("raw:", info)
        return None


async def click_text(ws, text):
    return await evaluate(ws, f"""
      (() => {{
        const els = [...document.querySelectorAll('button,a')];
        const el = els.find(e => (e.innerText||'').trim().toLowerCase().includes({json.dumps(text.lower())}));
        if (el) {{ el.click(); return 'clicked: ' + el.innerText.trim(); }}
        return 'NOT FOUND';
      }})()
    """)


async def main():
    ws_url = await get_ws()
    async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        await send(ws, "Page.enable")
        await send(ws, "Page.navigate", {"url": "https://myapi.fyers.in/dashboard"})
        await asyncio.sleep(6)
        r = await click_text(ws, "go to api dashboard")
        print("click ->", r)
        await asyncio.sleep(7)
        # the button may open a new tab/window; re-fetch tab list
        import urllib.request
        tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=6))
        print("\n-- open pages --")
        for t in tabs:
            if t.get("type") == "page":
                print("  ", t["url"][:75])
        # connect to the tab whose url mentions api-dashboard or user-apps
        target = next((t for t in tabs if t.get("type") == "page"
                       and ("api-dashboard" in t["url"] or "user-apps" in t["url"])), None)
        if target:
            print(f"\nconnecting to: {target['url']}")
            async with websockets.connect(target["webSocketDebuggerUrl"],
                                          max_size=50 * 1024 * 1024) as ws2:
                await asyncio.sleep(3)
                await dump(ws2, "API APP DASHBOARD", all_inputs=True)
        else:
            print("\nno api-dashboard tab found; re-dumping current tab")
            await dump(ws, "current", all_inputs=True)


if __name__ == "__main__":
    asyncio.run(main())
