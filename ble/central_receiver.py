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

ble
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
from store_forward import StoreForward

SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
CHAR_UUID    = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
WEB_PORT     = 8000

# ── Adaptive scanning ─────────────────────────────────────────────────────────
POLL_FAST          = 2.0    # seconds — recent activity
POLL_SLOW          = 8.0    # seconds — idle
ACTIVITY_WINDOW    = 60     # seconds — no message → switch to slow poll
_last_message_time = 0.0

def get_poll_interval() -> float:
    if time.time() - _last_message_time < ACTIVITY_WINDOW:
        return POLL_FAST
    return POLL_SLOW

# ── Reshare tracking ──────────────────────────────────────────────────────────
RESHARE_WARN   = 2   # show warning badge
RESHARE_VIRAL  = 4   # show "Widely Shared" badge (does NOT override GCN verdict)
_reshare_counts = {}

def _normalize(text: str) -> str:
    return " ".join(text.lower().split())

def check_reshare(text: str) -> int:
    key = _normalize(text)
    _reshare_counts[key] = _reshare_counts.get(key, 0) + 1
    return _reshare_counts[key]

# ── State ─────────────────────────────────────────────────────────────────────
history = []          # newest first, served to browser
sf      = StoreForward()  # store & forward engine

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
    background: #f4f6f9;
    color: #1a1f2e;
    min-height: 100vh;
    padding: 0 0 60px;
  }

  /* ── header ── */
  .header {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
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
    border-radius: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800; color: #fff; letter-spacing: -.02em;
  }
  .brand-name { font-size: 16px; font-weight: 700; color: #1a1f2e; letter-spacing: -.01em; }
  .brand-sub  { font-size: 12px; color: #94a3b8; margin-top: 1px; }
  .status-pill {
    display: flex; align-items: center; gap: 7px;
    font-size: 12px; font-weight: 600;
    padding: 6px 14px; border-radius: 0;
    transition: background .3s, color .3s, border-color .3s;
  }
  .status-pill.online  { background: #f0fdf4; border: 1px solid #bbf7d0; color: #16a34a; }
  .status-pill.offline { background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; }
  .dot {
    width: 7px; height: 7px; border-radius: 50%;
    flex-shrink: 0;
    transition: background .3s;
  }
  .status-pill.online  .dot { background: #16a34a; animation: pulse 2s ease-in-out infinite; }
  .status-pill.offline .dot { background: #dc2626; animation: none; }
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
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 0;
    padding: 16px 18px;
  }
  .stat-label { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px; }
  .stat-value { font-size: 24px; font-weight: 700; color: #1a1f2e; font-variant-numeric: tabular-nums; }
  .stat-value.fake-col { color: #f87171; }
  .stat-value.true-col { color: #34d399; }

  /* ── feed ── */
  .feed-label {
    font-size: 11px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: #94a3b8;
    margin-bottom: 14px;
  }
  #list { display: flex; flex-direction: column; gap: 10px; }

  .empty-state {
    background: #ffffff;
    border: 1px dashed #e2e8f0;
    border-radius: 0;
    padding: 48px 24px;
    text-align: center;
    color: #94a3b8;
  }
  .empty-icon { display: none; }
  .empty-title { font-size: 15px; font-weight: 600; color: #64748b; margin-bottom: 4px; }
  .empty-sub { font-size: 13px; }

  /* ── message card ── */
  .card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 0;
    padding: 18px 20px;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px 20px;
    align-items: start;
    transition: border-color .2s;
  }
  .card:hover { border-color: #cbd5e1; }
  .card.fake { border-left: 3px solid #f87171; }
  .card.true { border-left: 3px solid #34d399; }

  .card-msg {
    font-size: 14.5px;
    color: #334155;
    line-height: 1.55;
    grid-column: 1 / -1;
    margin-bottom: 2px;
  }

  .card-meta { font-size: 12px; color: #94a3b8; display: flex; align-items: center; gap: 6px; }
  .sep { color: #e2e8f0; }

  .verdict {
    display: flex; align-items: center; gap: 8px;
    justify-content: flex-end;
    flex-shrink: 0;
  }
  .verdict-badge {
    font-size: 11px; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase;
    padding: 4px 11px; border-radius: 0;
  }
  .verdict-badge.fake { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
  .verdict-badge.true { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }

  .conf-bar-wrap { width: 72px; }
  .conf-label { font-size: 10px; color: #94a3b8; text-align: right; margin-bottom: 4px; font-variant-numeric: tabular-nums; }
  .conf-bar { height: 4px; background: #e2e8f0; border-radius: 0; overflow: hidden; }
  .conf-fill { height: 100%; border-radius: 0; }
  .conf-fill.fake { background: #f87171; }
  .conf-fill.true { background: #34d399; }

  .reshare-badge {
    font-size: 10px; font-weight: 700; letter-spacing: .05em;
    text-transform: uppercase; padding: 3px 8px;
    background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa;
  }
  .reshare-badge.danger {
    background: #fef2f2; color: #dc2626; border: 1px solid #fecaca;
  }
  .sf-badge {
    font-size: 10px; font-weight: 700; letter-spacing: .05em;
    text-transform: uppercase; padding: 3px 8px;
    background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;
  }
  .hop-badge {
    font-size: 10px; font-weight: 600; padding: 3px 8px;
    background: #f5f3ff; color: #6d28d9; border: 1px solid #ddd6fe;
  }
  .card.reshared { border-left-color: #f97316; }
  .card.reshared.fake { border-left-color: #dc2626; }
  .card.stored { border-left-color: #1d4ed8; border-style: dashed; }
</style>
</head>
<body>

<div class="header">
  <div class="brand">
    <div class="brand-icon">BLE</div>
    <div>
      <div class="brand-name">BLE Fake News Detector</div>
      <div class="brand-sub">GCN &middot; Offline &middot; BLE Mesh PoC</div>
    </div>
  </div>
  <div class="status-pill online" id="status"><div class="dot"></div> <span id="status-text">Live</span></div>
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
      <div class="empty-title">Waiting for messages</div>
      <div class="empty-sub">Send a message from nRF Connect to see results here.</div>
    </div>
  </div>
</div>

<script>
const statusEl = document.getElementById('status');
const statusTxt = document.getElementById('status-text');

function setOnline() {
  statusEl.className = 'status-pill online';
  statusTxt.textContent = 'Live';
}
function setOffline() {
  statusEl.className = 'status-pill offline';
  statusTxt.textContent = 'Reconnecting...';
}

async function poll() {
  let items;
  try {
    const res = await fetch('/api/latest');
    items = await res.json();
    setOnline();
  } catch(e) {
    setOffline();
    return;
  }

  document.getElementById('s-total').textContent = items.length;
  document.getElementById('s-fake').textContent  = items.filter(i => i.prediction === 'Fake').length;
  document.getElementById('s-true').textContent  = items.filter(i => i.prediction === 'True').length;

  const list = document.getElementById('list');
  if (items.length === 0) {
    list.innerHTML = '<div class="empty-state"><div class="empty-title">Waiting for messages</div><div class="empty-sub">Send a message from nRF Connect to see results here.</div></div>';
    return;
  }
  list.innerHTML = items.map(it => {
    const cls      = it.prediction.toLowerCase();
    const reshared = it.reshare_count > 1;
    const danger   = it.reshare_count >= 4;
    const stored   = it.stored || false;
    const hops     = it.hop_count || 1;
    let cardCls    = 'card ' + cls;
    if (reshared) cardCls += ' reshared';
    if (stored)   cardCls += ' stored';
    const rsLabel  = danger
      ? 'Reshared ' + it.reshare_count + 'x &mdash; Widely Shared'
      : 'Reshared ' + it.reshare_count + 'x';
    const rsBadge  = reshared
      ? '<span class="reshare-badge' + (danger ? ' danger' : '') + '">' + rsLabel + '</span>'
      : '';
    const sfBadge  = stored
      ? '<span class="sf-badge">Stored &amp; Forwarded</span>'
      : '';
    const hopBadge = '<span class="hop-badge">Hop ' + hops + '</span>';
    return '<div class="' + cardCls + '">' +
      '<div class="card-msg">' + it.message + '</div>' +
      '<div class="card-meta"><span>' + it.time + '</span><span class="sep">·</span>' +
        hopBadge +
        (reshared ? '<span class="sep">·</span>' + rsBadge : '') +
        (stored   ? '<span class="sep">·</span>' + sfBadge  : '') +
      '</div>' +
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


def handle_message(raw: bytes, last_seen: list, hop_count: int = 1):
    global _last_message_time
    if not raw:
        return
    try:
        text = bytes(raw).decode("utf-8").strip()
    except UnicodeDecodeError:
        return
    if not text or text == last_seen[0]:
        return
    last_seen[0] = text
    _last_message_time = time.time()

    result        = classify(text)
    reshare_count = check_reshare(text)

    # GCN prediction is never overridden by reshare count.
    # High reshare count is shown as context — true news also spreads widely.

    poll = get_poll_interval()
    reshare_note = (f"  [RESHARED {reshare_count}x — "
                    f"{'WIDELY SHARED — VERIFY' if reshare_count >= RESHARE_VIRAL else 'WARN: reshared'}]"
                    if reshare_count >= RESHARE_WARN else "")

    print("=" * 60)
    print(f"Message:    {result['message']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Hop count:  {hop_count}  |  Poll interval: {poll}s")
    if reshare_note:
        print(reshare_note)
    print("=" * 60)

    entry = {
        "message":       result["message"],
        "prediction":    result["prediction"],
        "confidence":    result["confidence"],
        "reshare_count": reshare_count,
        "hop_count":     hop_count,
        "stored":        False,
        "time":          time.strftime("%H:%M:%S"),
    }
    history.insert(0, entry)
    del history[50:]

    # Persist to store & forward
    sf.save(entry)


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


def _load_stored_messages():
    """Inject stored messages into history on reconnect (Store & Forward delivery)."""
    pending = sf.pending()
    if not pending:
        return
    print(f"[Store&Forward] Delivering {len(pending)} stored message(s)...")
    for entry in reversed(pending):
        entry["stored"] = True
        history.insert(0, dict(entry))
    del history[50:]
    sf.mark_all_delivered()
    print("[Store&Forward] Delivery complete.")


async def run():
    start_web_server()

    print("Loading classifier model (first run downloads it)...")
    classify("warm up")
    print("Model ready.")

    # Deliver any messages stored from previous session
    _load_stored_messages()

    hop_count    = 1   # phone→laptop = 1 hop (multi-hop future extension)
    reconnect_wait = 3

    while True:
        try:
            device    = await find_peripheral()
            last_seen = [""]

            async with BleakClient(device) as client:
                print(f"Connected to {device.address}")

                # Deliver messages stored while we were disconnected
                _load_stored_messages()

                def on_notify(_handle, data: bytearray):
                    handle_message(data, last_seen, hop_count)

                notify_supported = True
                try:
                    await client.start_notify(CHAR_UUID, on_notify)
                    print("Subscribed to notifications.")
                except Exception as e:
                    notify_supported = False
                    print(f"Notify not available ({e}); polling only.")

                print("Waiting for messages... (Ctrl+C to stop)")
                print(f"Adaptive polling: fast={POLL_FAST}s / slow={POLL_SLOW}s "
                      f"(switches after {ACTIVITY_WINDOW}s idle)\n")

                while True:
                    if not client.is_connected:
                        print("Connection lost.")
                        break
                    try:
                        value = await client.read_gatt_char(CHAR_UUID)
                        handle_message(value, last_seen, hop_count)
                    except Exception as e:
                        print(f"Read failed: {e}")
                        break

                    # Adaptive scan: slow down when idle, speed up when active
                    interval = get_poll_interval()
                    await asyncio.sleep(interval)

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

        print(f"Reconnecting in {reconnect_wait} seconds...\n")
        await asyncio.sleep(reconnect_wait)


if __name__ == "__main__":
    asyncio.run(run())
