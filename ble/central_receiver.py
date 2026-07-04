"""
Laptop-as-central PoC.

The phone acts as a BLE PERIPHERAL (set up via nRF Connect's "Configure
GATT Server" feature - see README). This script scans for that peripheral,
connects, and watches one characteristic for the message text. Whenever
the value changes (via notify, or via polling as a fallback), it decodes
the text, runs the local fake-news classifier, and shows:
    Message / Prediction / Confidence

Results print to the console AND are served on a tiny local web UI at
http://localhost:8000 (auto-refreshing, newest first).

Run:  python central_receiver.py
Stop: Ctrl+C
"""

import asyncio
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from classifier import classify  # noqa: E402

from bleak import BleakClient, BleakScanner

SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

POLL_INTERVAL_SEC = 2.0
WEB_PORT = 8000

history = []  # newest first

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BLE Fake News Detector</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: #0d1117;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 0 0 60px;
  }

  /* ── header ── */
  .header {
    background: #161b22;
    border-bottom: 1px solid #21262d;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #1d4ed8, #6d28d9);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .brand-name { font-size: 16px; font-weight: 700; color: #f1f5f9; letter-spacing: -.01em; }
  .brand-sub  { font-size: 12px; color: #64748b; margin-top: 1px; }
  .status-pill {
    display: flex; align-items: center; gap: 7px;
    background: #0d2318; border: 1px solid #1a4a2e;
    color: #34d399; font-size: 12px; font-weight: 600;
    padding: 6px 14px; border-radius: 999px;
  }
  .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #34d399;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

  /* ── main ── */
  .main { max-width: 680px; margin: 0 auto; padding: 32px 20px 0; }

  .stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 28px;
  }
  .stat {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 18px;
  }
  .stat-label { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }
  .stat-value { font-size: 24px; font-weight: 700; color: #f1f5f9; font-variant-numeric: tabular-nums; }
  .stat-value.fake-col { color: #f87171; }
  .stat-value.true-col { color: #34d399; }

  /* ── feed ── */
  .feed-label {
    font-size: 11px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: #475569;
    margin-bottom: 14px;
  }
  #list { display: flex; flex-direction: column; gap: 10px; }

  .empty-state {
    background: #161b22;
    border: 1px dashed #21262d;
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    color: #475569;
  }
  .empty-icon { font-size: 32px; margin-bottom: 12px; }
  .empty-title { font-size: 15px; font-weight: 600; color: #64748b; margin-bottom: 4px; }
  .empty-sub { font-size: 13px; }

  /* ── message card ── */
  .card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 20px;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px 20px;
    align-items: start;
    transition: border-color .2s;
  }
  .card:hover { border-color: #30363d; }
  .card.fake { border-left: 3px solid #f87171; }
  .card.true { border-left: 3px solid #34d399; }

  .card-msg {
    font-size: 14.5px;
    color: #cbd5e1;
    line-height: 1.55;
    grid-column: 1 / -1;
    margin-bottom: 2px;
  }

  .card-meta { font-size: 12px; color: #475569; display: flex; align-items: center; gap: 6px; }
  .sep { color: #2d3748; }

  .verdict {
    display: flex; align-items: center; gap: 8px;
    justify-content: flex-end;
    flex-shrink: 0;
  }
  .verdict-badge {
    font-size: 11px; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase;
    padding: 4px 11px; border-radius: 6px;
  }
  .verdict-badge.fake { background: #2d1515; color: #f87171; border: 1px solid #4a1f1f; }
  .verdict-badge.true { background: #0d2318; color: #34d399; border: 1px solid #1a4a2e; }

  .conf-bar-wrap { width: 72px; }
  .conf-label { font-size: 10px; color: #475569; text-align: right; margin-bottom: 4px; font-variant-numeric: tabular-nums; }
  .conf-bar { height: 4px; background: #21262d; border-radius: 99px; overflow: hidden; }
  .conf-fill { height: 100%; border-radius: 99px; }
  .conf-fill.fake { background: #f87171; }
  .conf-fill.true { background: #34d399; }
</style>
</head>
<body>

<div class="header">
  <div class="brand">
    <div class="brand-icon">&#x1F4E1;</div>
    <div>
      <div class="brand-name">BLE Fake News Detector</div>
      <div class="brand-sub">GCN · Offline · BLE Mesh PoC</div>
    </div>
  </div>
  <div class="status-pill"><div class="dot"></div> Live</div>
</div>

<div class="main">
  <div class="stats-row">
    <div class="stat">
      <div class="stat-label">Total</div>
      <div class="stat-value" id="s-total">0</div>
    </div>
    <div class="stat">
      <div class="stat-label">Fake</div>
      <div class="stat-value fake-col" id="s-fake">0</div>
    </div>
    <div class="stat">
      <div class="stat-label">True</div>
      <div class="stat-value true-col" id="s-true">0</div>
    </div>
  </div>

  <div class="feed-label">Message Feed</div>
  <div id="list">
    <div class="empty-state">
      <div class="empty-icon">&#x1F4F6;</div>
      <div class="empty-title">Waiting for messages</div>
      <div class="empty-sub">Send a message from nRF Connect to see results here.</div>
    </div>
  </div>
</div>

<script>
async function poll() {
  let items;
  try {
    const res = await fetch('/api/latest');
    items = await res.json();
  } catch(e) { return; }

  document.getElementById('s-total').textContent = items.length;
  document.getElementById('s-fake').textContent  = items.filter(i => i.prediction === 'Fake').length;
  document.getElementById('s-true').textContent  = items.filter(i => i.prediction === 'True').length;

  const list = document.getElementById('list');
  if (items.length === 0) {
    list.innerHTML = '<div class="empty-state"><div class="empty-icon">&#x1F4F6;</div><div class="empty-title">Waiting for messages</div><div class="empty-sub">Send a message from nRF Connect to see results here.</div></div>';
    return;
  }
  list.innerHTML = items.map(it => {
    const cls = it.prediction.toLowerCase();
    return '<div class="card ' + cls + '">' +
      '<div class="card-msg">' + it.message + '</div>' +
      '<div class="card-meta"><span>' + it.time + '</span><span class="sep">·</span><span>GCN</span></div>' +
      '<div class="verdict">' +
        '<div class="conf-bar-wrap">' +
          '<div class="conf-label">' + it.confidence + '%</div>' +
          '<div class="conf-bar"><div class="conf-fill ' + cls + '" style="width:' + it.confidence + '%"></div></div>' +
        '</div>' +
        '<div class="verdict-badge ' + cls + '">' + it.prediction + '</div>' +
      '</div>' +
    '</div>';
  }).join('');
}
poll();
setInterval(poll, 1500);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/latest":
            self._send(200, json.dumps(history).encode("utf-8"), "application/json")
        else:
            self._send(404, b"Not found", "text/plain")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def start_web_server():
    server = HTTPServer(("0.0.0.0", WEB_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Web UI: http://localhost:{WEB_PORT}")


def handle_message(raw: bytes, last_seen: list):
    if not raw:
        return
    try:
        text = bytes(raw).decode("utf-8").strip()
    except UnicodeDecodeError:
        return
    if not text or text == last_seen[0]:
        return
    last_seen[0] = text

    result = classify(text)
    print("=" * 60)
    print(f"Message:    {result['message']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print("=" * 60)

    history.insert(
        0,
        {
            "message": result["message"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "time": time.strftime("%H:%M:%S"),
        },
    )
    del history[50:]


async def find_peripheral():
    print(f"Scanning for peripheral...")
    attempt = 0
    while True:
        attempt += 1
        devices = await BleakScanner.discover(timeout=5.0, return_adv=True)

        # Priority 1: match by service UUID in advertisement
        for device, adv in devices.values():
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            if SERVICE_UUID in uuids:
                print(f"Found by UUID: {device.name} ({device.address})")
                return device

        # Priority 2: match by nRF Connect default name
        for device, adv in devices.values():
            name = (device.name or "").lower()
            if any(k in name for k in ("nrf", "fakenews", "nordic")):
                print(f"Found by name: {device.name} ({device.address})")
                return device

        # Every 2 attempts print all visible devices to help diagnose
        if attempt % 2 == 0 and devices:
            print("  Visible BLE devices right now:")
            for device, adv in devices.values():
                print(f"    {device.address}  name={device.name or '(unknown)'}  uuids={adv.service_uuids or []}")

        print(f"  Not found (attempt {attempt}). Toggle Advertiser OFF→ON on phone.")


async def run():
    start_web_server()

    print("Loading classifier model (first run downloads it)...")
    classify("warm up")
    print("Model ready.")

    while True:
        try:
            device = await find_peripheral()
            last_seen = [""]

            async with BleakClient(device) as client:
                print(f"Connected to {device.address}")

                def on_notify(_handle, data: bytearray):
                    handle_message(data, last_seen)

                notify_supported = True
                try:
                    await client.start_notify(CHAR_UUID, on_notify)
                    print("Subscribed to notifications.")
                except Exception as e:
                    notify_supported = False
                    print(f"Notify not available ({e}); polling only.")

                print("Waiting for messages... (Ctrl+C to stop)\n")

                while True:
                    if not client.is_connected:
                        print("Connection lost.")
                        break
                    try:
                        value = await client.read_gatt_char(CHAR_UUID)
                        handle_message(value, last_seen)
                    except Exception as e:
                        print(f"Read failed: {e}")
                        break
                    await asyncio.sleep(POLL_INTERVAL_SEC)

                if notify_supported:
                    try:
                        await client.stop_notify(CHAR_UUID)
                    except Exception:
                        pass

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")

        print("Reconnecting in 3 seconds...\n")
        await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(run())
