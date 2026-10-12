"""Go to the real FYERS app manager and dump the create-app form via CDP."""
import asyncio
import json
import urllib.request

import websockets

APP_URL = "https://fyers.in/web/api-dashboard/user-apps"
MY_IP = "106.210.180.193"
REDIRECT = "http://127.0.0.1:8081/callback"


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
    expr = """
      (() => JSON.stringify({
        url: location.href,
        inputs: [...document.querySelectorAll('input,textarea')].map((el,i)=>({
          i, type: el.type, name: el.name, id: el.id, ph: el.placeholder,
          val: el.value, vis: !!(el.offsetParent || el.offsetHeight)
        })),
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
        print("-- INPUTS (visible only) --")
        for el in s["inputs"]:
            if el["vis"]:
                print(f"  [{el['i']}] {el['type']:9} name={el['name']!r} id={el['id']!r} ph={el['ph']!r} val={el['val']!r}")
        print("-- BUTTONS/LINKS --")
        for el in s["btns"]:
            print(f"  [{el['i']}] <{el['tag']}> {el['text']!r}")
        return s
    except Exception:
        print("raw:", info)
        return None


async def main():
    tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=6))
    page = next((t for t in tabs if t.get("type") == "page"), None)
    async with websockets.connect(page["webSocketDebuggerUrl"],
                                  max_size=50 * 1024 * 1024) as ws:
        await send(ws, "Page.enable")
        await send(ws, "Page.navigate", {"url": APP_URL})
        await asyncio.sleep(9)
        await dump(ws, "APP MANAGER")


if __name__ == "__main__":
    asyncio.run(main())
