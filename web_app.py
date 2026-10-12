"""Minimal Python-only local web page for reviewing sample traffic records."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from html import escape
import json
from detection_engine import analyze_traffic
from data_generator import FEATURES

HTML = """<!doctype html><html><head><meta charset='utf-8'><title>5G Traffic Review</title>
<style>body{font:16px system-ui;max-width:850px;margin:3rem auto;padding:0 1rem;color:#17202a;background:#f5f7fa}h1{color:#123b5d}.card{background:white;padding:1.5rem;border-radius:12px;box-shadow:0 3px 14px #ccd3da}label{display:inline-block;width:220px;margin:.45rem}input{padding:.45rem;width:180px}button{padding:.7rem 1.2rem;background:#146c94;color:white;border:0;border-radius:6px;cursor:pointer}pre{white-space:pre-wrap;background:#eef3f7;padding:1rem;border-radius:8px}</style></head><body><h1>5G Telecom Traffic Review</h1><p>Local defensive simulation. Anomaly signals require human review and never block traffic.</p><div class='card'><form method='post'>"""
HTML += "".join(f"<label>{name.replace('_',' ').title()} <input name='{name}' type='number' step='any' required value='{value}'></label>" for name, value in zip(FEATURES, [120, 18000, 24, 0, 9, 750]))
HTML += "<p><button>Analyze traffic</button></p></form>__RESULT__</div></body></html>"


class Handler(BaseHTTPRequestHandler):
    def respond(self, body):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.respond(HTML.replace("__RESULT__", ""))

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            record = {key: form[key][0] for key in FEATURES}
            result = analyze_traffic(record)
            rendered = "<h2>Analysis result</h2><pre>" + escape(json.dumps(result, indent=2)) + "</pre>"
        except (ValueError, TypeError, FileNotFoundError, KeyError) as exc:
            rendered = "<p style='color:#a22'>" + escape(str(exc)) + "</p>"
        self.respond(HTML.replace("__RESULT__", rendered))


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in your browser (Ctrl+C to stop).")
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
