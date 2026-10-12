"""Drive the FYERS dashboard via Chrome DevTools Protocol — real DOM, no blind clicks.

Connects to Chrome's debug port, navigates to the FYERS dashboard, finds the
actual <input> elements by their real attributes, fills them, clicks Activate,
and reads back whatever credentials appear. No pixel-guessing.
"""
import asyncio
import json

import websockets

CDP = None
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
    res = r.get("result", {})
    if r.get("exceptionDetails"):
        return {"__error": str(r["exceptionDetails"].get("text"))}
    return res.get("value")


async def main():
    ws_url = await get_ws()
    async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        print("connected to Chrome via CDP")

        # 1. navigate to the dashboard
        await send(ws, "Page.enable")
        await send(ws, "Page.navigate", {"url": "https://myapi.fyers.in"})
        print("navigating to myapi.fyers.in ... waiting for login/load")
        await asyncio.sleep(8)

        # 2. dump what's actually on the page
        info = await evaluate(ws, """
          (() => {
            const inputs = [...document.querySelectorAll('input')].map((el,i) => ({
              i, type: el.type, name: el.name, id: el.id,
              ph: el.placeholder, val: el.value, cls: (el.className||'').slice(0,40)
            }));
            const btns = [...document.querySelectorAll('button')].map((el,i) => ({
              i, text: (el.innerText||'').trim().slice(0,40), cls:(el.className||'').slice(0,30)
            }));
            return JSON.stringify({url: location.href, title: document.title,
                                   loggedIn: !location.href.includes('login'),
                                   inputs, btns});
          })()
        """)
        print("\n=== PAGE STATE ===")
        try:
            s = json.loads(info) if isinstance(info, str) else info
            print("url:", s["url"], "| title:", s["title"])
            print("logged in:", s["loggedIn"])
            print("\n-- INPUTS --")
            for el in s["inputs"]:
                print(f"  [{el['i']}] type={el['type']} name={el['name']!r} id={el['id']!r} ph={el['ph']!r} val={el['val']!r}")
            print("\n-- BUTTONS --")
            for el in s["btns"]:
                print(f"  [{el['i']}] {el['text']!r}")
        except Exception as e:
            print("raw:", info)


if __name__ == "__main__":
    asyncio.run(main())
