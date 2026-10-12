"""Navigate to the FYERS app-creation form and fill it via CDP.

Now logged in. Walk: Get Started -> Create app -> fill Primary IP + Redirect ->
Activate -> read back App ID + Secret. All through the real DOM.
"""
import asyncio
import json

import websockets

MY_IP = "106.210.180.193"
REDIRECT = "http://127.0.0.1:8081/callback"


async def get_ws():
    import urllib.request
    tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=6))
    page = next((t for t in tabs if t.get("type") == "page"), None)
    return page["webSocketDebuggerUrl"]


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


async def dump(ws, tag):
    info = await evaluate(ws, """
      (() => JSON.stringify({
        url: location.href,
        inputs: [...document.querySelectorAll('input,textarea')].map((el,i)=>({
          i, type: el.type, name: el.name, id: el.id, ph: el.placeholder,
          val: el.value, vis: !!(el.offsetParent)
        })).filter(e => e.vis),
        btns: [...document.querySelectorAll('button,a')].map((el,i)=>({
          i, text:(el.innerText||'').trim().slice(0,45), tag: el.tagName
        })).filter(e=>e.text)
      }))()
    """)
    print(f"\n=== {tag} ===")
    try:
        s = json.loads(info)
        print("url:", s["url"])
        print("-- INPUTS (visible) --")
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


async def fill(ws, sel, value):
    return await evaluate(ws, f"""
      (() => {{
        const el = document.querySelector({json.dumps(sel)});
        if (!el) return 'NOT FOUND: ' + {json.dumps(sel)};
        el.focus();
        el.value = {json.dumps(value)};
        el.dispatchEvent(new Event('input', {{bubbles:true}}));
        el.dispatchEvent(new Event('change', {{bubbles:true}}));
        return 'filled ' + {json.dumps(sel)} + ' = ' + el.value;
      }})()
    """)


async def main():
    ws_url = await get_ws()
    async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        await send(ws, "Page.enable")
        await send(ws, "Page.navigate", {"url": "https://myapi.fyers.in"})
        await asyncio.sleep(6)
        s = await dump(ws, "LANDING")

        # click Get Started / Create
        for label in ["get started", "create app", "create", "new app"]:
            r = await click_text(ws, label)
            print(f"click {label!r} -> {r}")
            if r != "NOT FOUND":
                await asyncio.sleep(3)
                break
        await dump(ws, "AFTER GET STARTED")


if __name__ == "__main__":
    asyncio.run(main())
