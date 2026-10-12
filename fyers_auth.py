"""Complete the FYERS OAuth: capture auth_code -> access_token -> verify.

Runs the local :8081 callback server, opens the FYERS login, waits for the
redirect, exchanges the code for a token, caches it, and verifies with
get_profile(). Then pulls the live Nifty option chain and prints the condor.
"""
import http.server
import json
import os
import re
import socketserver
import urllib.parse
import webbrowser

from fyers_apiv3 import fyersModel

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, "fyers_env.txt")
TOKEN_FILE = os.path.join(HERE, "fyers_token.json")

creds = {}
for line in open(ENV):
    m = re.match(r'\s*(\w+)\s*=\s*"([^"]*)"', line)
    if m:
        creds[m.group(1)] = m.group(2)

CID = creds["FYERS_CLIENT_ID"]
SEC = creds["FYERS_SECRET_KEY"]
RED = creds["FYERS_REDIRECT_URI"]

s = fyersModel.SessionModel(client_id=CID, secret_key=SEC, redirect_uri=RED,
                            response_type="code", grant_type="authorization_code")
url = s.generate_authcode()

box = {}


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        box["auth_code"] = q.get("auth_code", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<h2>Auth captured. Return to the terminal.</h2>")

    def log_message(self, *a):
        pass


print("Opening FYERS login in Firefox ... log in, and the token is captured automatically.")
import subprocess
FIREFOX = r"C:\Users\sydne\AppData\Local\Mozilla Firefox\firefox.exe"
subprocess.Popen([FIREFOX, url])

socketserver.TCPServer.allow_reuse_address = True
port = urllib.parse.urlparse(RED).port or 80
with socketserver.TCPServer(("127.0.0.1", port), H) as httpd:
    httpd.handle_request()

code = box.get("auth_code")
if not code:
    raise SystemExit("no auth_code captured")
print("auth_code captured, exchanging for access_token ...")

s.set_token(code)
resp = s.generate_token()
print("token response keys:", list(resp.keys()))
if "access_token" not in resp:
    raise SystemExit(f"FAILED: {resp}")

json.dump(resp, open(TOKEN_FILE, "w"))
print(f"access_token cached -> {os.path.basename(TOKEN_FILE)}")

fy = fyersModel.FyersModel(client_id=CID, token=resp["access_token"],
                           is_async=False, log_path="")
prof = fy.get_profile()
print("profile:", prof.get("s"), prof.get("data", {}).get("name", ""))
