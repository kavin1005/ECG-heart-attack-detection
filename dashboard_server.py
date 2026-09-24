"""
dashboard_server.py – Real-time Interactive ECG Classification Dashboard Server
================================================================================
Serves the web dashboard and handles REST API calls for:
  - ECG sample retrieval (187 data points)
  - 1D CNN model inference
  - Beat catalog by class (Normal, Ventricular, Supraventricular, etc.)
  - Model evaluation metrics

Run:
    python dashboard_server.py
Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.parse
import numpy as np

# Ensure UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PREP_DIR      = os.path.join(BASE_DIR, "preprocessed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results", "cnn")
STATIC_DIR    = os.path.join(BASE_DIR, "dashboard")

MODEL_PATH    = os.path.join(MODELS_DIR, "cnn_mitbih.keras")
PORT          = 5000

# ── Class Metadata ─────────────────────────────────────────────────────────
CLASSES = [
    {
        "id": 0,
        "name": "Normal (N)",
        "code": "NORM",
        "description": "Normal sinus rhythm beat",
        "severity": "low",
        "color": "#00e676"
    },
    {
        "id": 1,
        "name": "Supraventricular Ectopy (S)",
        "code": "SVEB",
        "description": "Atrial premature or supraventricular premature beat",
        "severity": "medium",
        "color": "#ffb300"
    },
    {
        "id": 2,
        "name": "Ventricular Ectopy (V)",
        "code": "VEB",
        "description": "Premature ventricular contraction (PVC) - high clinical concern",
        "severity": "high",
        "color": "#ff1744"
    },
    {
        "id": 3,
        "name": "Fusion Beat (F)",
        "code": "FUS",
        "description": "Fusion of ventricular and normal beat",
        "severity": "medium",
        "color": "#ff9100"
    },
    {
        "id": 4,
        "name": "Unclassifiable (Q)",
        "code": "UNKNOWN",
        "description": "Paced beat or distorted waveform artifact",
        "severity": "info",
        "color": "#b388ff"
    }
]

# ── Global Cache ───────────────────────────────────────────────────────────
DATA_CACHE = {
    "X_test": None,
    "y_test": None,
    "model": None,
    "catalog": {},      # {class_id: [indices...]}
    "metrics": {}
}


def load_dataset_and_model():
    print("\n[Dashboard Server] Initializing resources...")
    x_path = os.path.join(PREP_DIR, "mitbih_test_X.npy")
    y_path = os.path.join(PREP_DIR, "mitbih_test_y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"[ERROR] Test data not found in {PREP_DIR}. Please run preprocess.py first.")
        sys.exit(1)

    print("  Loading MIT-BIH test data...")
    X_test = np.load(x_path, allow_pickle=False)
    y_test = np.load(y_path, allow_pickle=False)
    DATA_CACHE["X_test"] = X_test
    DATA_CACHE["y_test"] = y_test
    print(f"  Test samples: {len(X_test)} (187 points each)")

    # Build catalog of sample indices per class for quick exploration
    catalog = {c["id"]: [] for c in CLASSES}
    for idx, label in enumerate(y_test):
        lbl = int(label)
        if lbl in catalog and len(catalog[lbl]) < 100:  # store top 100 per class
            catalog[lbl].append(idx)
    DATA_CACHE["catalog"] = catalog
    print(f"  Catalog indexed: {', '.join(f'Class {k}: {len(v)} samples' for k, v in catalog.items())}")

    # Load 1D CNN model
    if os.path.exists(MODEL_PATH):
        print(f"  Loading trained CNN model from {MODEL_PATH}...")
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        # Warm up inference
        dummy = np.zeros((1, 187, 1), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        DATA_CACHE["model"] = model
        print("  Model loaded and warmed up successfully!")
    else:
        print(f"  [WARNING] Model file {MODEL_PATH} not found. Real-time inference will be unavailable until trained.")

    # Load metrics if available
    metrics_csv = os.path.join(RESULTS_DIR, "cnn_mitbih_metrics.csv")
    if os.path.exists(metrics_csv):
        try:
            with open(metrics_csv, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                if len(lines) > 1:
                    headers = [h.strip() for h in lines[0].split(",")]
                    values  = [v.strip() for v in lines[1].split(",")]
                    DATA_CACHE["metrics"] = dict(zip(headers, values))
        except Exception as e:
            print(f"  [Notice] Could not read metrics CSV: {e}")


# ── Threaded HTTP Server ───────────────────────────────────────────────────
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ECGRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        # Concise logging
        if "/api/" in args[0]:
            print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # ── API: Server & Model Info ───────────────────────────────────────
        if path == "/api/info":
            return self.send_json({
                "status": "ready",
                "model_available": DATA_CACHE["model"] is not None,
                "total_test_samples": len(DATA_CACHE["X_test"]) if DATA_CACHE["X_test"] is not None else 0,
                "classes": CLASSES,
                "metrics": DATA_CACHE["metrics"]
            })

        # ── API: Catalog of Sample Indices by Class ────────────────────────
        elif path == "/api/catalog":
            return self.send_json({
                "catalog": DATA_CACHE["catalog"],
                "classes": CLASSES
            })

        # ── API: Get Single Sample Data ────────────────────────────────────
        elif path == "/api/sample":
            idx = params.get("index", ["0"])[0]
            try:
                idx = int(idx)
            except ValueError:
                return self.send_json({"error": "Invalid index"}, 400)

            X_test = DATA_CACHE["X_test"]
            y_test = DATA_CACHE["y_test"]

            if X_test is None or idx < 0 or idx >= len(X_test):
                return self.send_json({"error": f"Index out of range (0..{len(X_test)-1})"}, 404)

            signal = X_test[idx].tolist()
            true_label = int(y_test[idx])
            class_meta = CLASSES[true_label] if true_label < len(CLASSES) else {"name": "Unknown"}

            return self.send_json({
                "index": idx,
                "signal": signal,
                "true_label": true_label,
                "class_meta": class_meta
            })

        # ── API: Get Model Evaluation Metrics ──────────────────────────────
        elif path == "/api/metrics":
            return self.send_json({
                "metrics": DATA_CACHE["metrics"],
                "classes": CLASSES
            })

        # ── Serve Frontend Static Files ────────────────────────────────────
        else:
            return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # ── API: Predict ECG Beat ──────────────────────────────────────────
        if path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                return self.send_json({"error": "Empty body"}, 400)

            raw_body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(raw_body)
            except Exception:
                return self.send_json({"error": "Invalid JSON"}, 400)

            model = DATA_CACHE["model"]
            if model is None:
                return self.send_json({"error": "Model not loaded on server"}, 503)

            # Option A: sample index provided
            if "index" in payload:
                idx = int(payload["index"])
                X_test = DATA_CACHE["X_test"]
                y_test = DATA_CACHE["y_test"]
                if idx < 0 or idx >= len(X_test):
                    return self.send_json({"error": "Index out of range"}, 404)
                signal_np = X_test[idx]
                true_label = int(y_test[idx])
            # Option B: raw 187 points array provided
            elif "signal" in payload:
                signal_raw = payload["signal"]
                if len(signal_raw) != 187:
                    return self.send_json({"error": f"Signal must contain exactly 187 values (got {len(signal_raw)})"}, 400)
                signal_np = np.array(signal_raw, dtype=np.float32)
                true_label = payload.get("true_label", None)
            else:
                return self.send_json({"error": "Provide either 'index' or 'signal'"}, 400)

            # Run inference
            t0 = time.perf_counter()
            inp = signal_np.reshape(1, 187, 1)
            proba = model.predict(inp, verbose=0)[0].tolist()
            inference_ms = (time.perf_counter() - t0) * 1000

            pred_class = int(np.argmax(proba))
            confidence = proba[pred_class]
            class_meta = CLASSES[pred_class] if pred_class < len(CLASSES) else {"name": "Unknown", "severity": "info"}

            # Build response
            response = {
                "predicted_label": pred_class,
                "confidence": round(confidence * 100, 2),
                "probabilities": [
                    {
                        "class_id": c["id"],
                        "name": c["name"],
                        "code": c["code"],
                        "color": c["color"],
                        "prob": round(p * 100, 2)
                    }
                    for c, p in zip(CLASSES, proba)
                ],
                "class_meta": class_meta,
                "inference_time_ms": round(inference_ms, 2)
            }

            if true_label is not None:
                response["true_label"] = true_label
                response["is_correct"] = (pred_class == true_label)
                response["true_class_meta"] = CLASSES[true_label] if true_label < len(CLASSES) else None

            return self.send_json(response)

        else:
            return self.send_json({"error": "Not Found"}, 404)


def main():
    load_dataset_and_model()

    server_address = ("", PORT)
    httpd = ThreadedHTTPServer(server_address, ECGRequestHandler)

    print("\n" + "=" * 65)
    print("  🏥 ECG Arrhythmia Classification Dashboard")
    print("=" * 65)
    print(f"  Web Dashboard running at: http://localhost:{PORT}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Shutting down dashboard server...]")
        httpd.server_close()


if __name__ == "__main__":
    main()
