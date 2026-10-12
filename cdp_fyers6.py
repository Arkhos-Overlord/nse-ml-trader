"""Trigger the FYERS app-dashboard redirect FROM the logged-in myapi tab.

The fyers.in app-manager is an empty shell on a direct hit because the session
cookie lives on myapi.fyers.in. So: from the logged-in myapi tab, click 'Go to
API Dashboard' (which does window.open to fyers.in carrying the session), let
the new tab hydrate, then dump + fill its form.
"""
import asyncio
import json
import urllib.request

import websockets

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
        return {"__error": str(r["exceptionDetails"].get("text"))[:120]}
    return r.get("result", {}).get("value")


async def dump(ws, tag):
    expr = """
      (() => JSON.stringify({
        url: location.href, title: document.title,
        bodyLen: document.body.innerText.length,
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
        print(f"url: {s['url']} | title: {s['title']} | bodyLen: {s['bodyLen']}")
        vis_inputs = [el for el in s["inputs"] if el["vis"]]
        print(f"-- INPUTS ({len(vis_inputs)} visible / {len(s['inputs'])} total) --")
        for el in vis_inputs:
            print(f"  [{el['i']}] {el['type']:9} name={el['name']!r} id={el['id']!r} ph={el['ph']!r} val={el['val']!r}")
        print("-- BUTTONS/LINKS --")
        for el in s["btns"][:25]:
            print(f"  [{el['i']}] <{el['tag']}> {el['text']!r}")
        return s
    except Exception:
        print("raw:", info)
        return None


def list_pages():
    return json.load(urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=6))


async def main():
    # 1. find the logged-in myapi tab
    pages = [t for t in list_pages() if t.get("type") == "page"]
    myapi = next((t for t in pages if "myapi.fyers.in" in t["url"]), None)
    if not myapi:
        print("no myapi tab found; pages:", [t["url"] for t in pages])
        return
    async with websockets.connect(myapi["webSocketDebuggerUrl"],
                                  max_size=50 * 1024 * 1024) as ws:
        await send(ws, "Page.enable")
        await send(ws, "Page.navigate", {"url": "https://myapi.fyers.in/dashboard"})
        await asyncio.sleep(6)
        # capture the URL the button opens
        await evaluate(ws, "window.__navs=[]; const oo=window.open; window.open=(u)=>{window.__navs.push(u);return oo.apply(window,arguments)};")
        await evaluate(ws, "(()=>{const b=[...document.querySelectorAll('button')].find(e=>(e.innerText||'').includes('Go to API Dashboard')); if(b){b.click();return 'clicked'} return 'no btn'})()")
        await asyncio.sleep(4)
        navs = await evaluate(ws, "JSON.stringify(window.__navs)")
        print("window.open target:", navs)
        target = json.loads(navs)[0] if navs and navs != "[]" else "https://fyers.in/web/api-dashboard/user-apps"
        # 2. navigate the SAME tab to the target (session cookie carries)
        await send(ws, "Page.navigate", {"url": target})
        print("navigating to", target, "... waiting for SPA hydrate")
        await asyncio.sleep(12)
        await dump(ws, "FYERS APP MANAGER")


if __name__ == "__main__":
    asyncio.run(main())
