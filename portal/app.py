import base64
import io
import os
import threading
import time
from collections import deque
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, redirect, render_template_string, request, url_for

from tgcn_pipeline import TGCN, build_graph_timeseries, build_sequences

app = Flask(__name__)

APP_URL = os.getenv("APP_URL", "http://app:8000/")
TELEMETRY_FILE = os.getenv("TELEMETRY_FILE", "/data/telemetry_data.csv")
MODEL_BUNDLE = os.getenv("MODEL_BUNDLE", "/artifacts/tgcn/tgcn_inference_bundle.pt")
PORT = int(os.getenv("PORT", "5050"))

HEADER = "Timestamp,Hour_Sin,Hour_Cos,Source,Target,Latency_ms,CPU_Usage,Mem_Usage,Users,Status\n"

CONTROL_LOCK = threading.Lock()
CONTROL_STATE: Dict[str, object] = {
    "users": 180,
    "mode": "day",
    "req_per_cycle": 2,
    "interval_sec": 2.0,
    "force_crash": False,
    "running": True,
}
DRIVER_EVENTS = deque(maxlen=60)

INFER_RUNTIME = {
    "ready": False,
    "error": "Model not loaded",
}


def init_log_file() -> None:
    if not os.path.isfile(TELEMETRY_FILE) or os.path.getsize(TELEMETRY_FILE) == 0:
        os.makedirs(os.path.dirname(TELEMETRY_FILE), exist_ok=True)
        with open(TELEMETRY_FILE, "w", encoding="utf-8") as f:
            f.write(HEADER)


def reset_log_file() -> None:
    with open(TELEMETRY_FILE, "w", encoding="utf-8") as f:
        f.write(HEADER)


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_telemetry() -> pd.DataFrame:
    if not os.path.isfile(TELEMETRY_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(TELEMETRY_FILE, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

    required = {
        "Timestamp",
        "Source",
        "Target",
        "Latency_ms",
        "CPU_Usage",
        "Mem_Usage",
        "Users",
        "Status",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df["Timestamp"] = _to_float(df["Timestamp"]).fillna(0.0)
    df["Latency_ms"] = _to_float(df["Latency_ms"]).fillna(0.0)
    df["CPU_Usage"] = _to_float(df["CPU_Usage"]).fillna(0.0)
    df["Mem_Usage"] = _to_float(df["Mem_Usage"]).fillna(0.0)
    df["Users"] = _to_float(df["Users"]).fillna(0.0)

    if "Hour_Sin" not in df.columns or "Hour_Cos" not in df.columns:
        dt = pd.to_datetime(df["Timestamp"], unit="s", errors="coerce")
        hour = dt.dt.hour.fillna(0).astype(int)
        df["Hour_Sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["Hour_Cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    else:
        df["Hour_Sin"] = _to_float(df["Hour_Sin"]).fillna(0.0)
        df["Hour_Cos"] = _to_float(df["Hour_Cos"]).fillna(1.0)

    df["Status"] = df["Status"].astype(str)
    df["Source"] = df["Source"].astype(str)
    df["Target"] = df["Target"].astype(str)

    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def load_model_bundle() -> None:
    global INFER_RUNTIME
    if not os.path.isfile(MODEL_BUNDLE):
        INFER_RUNTIME = {"ready": False, "error": f"Model bundle not found: {MODEL_BUNDLE}"}
        return

    try:
        bundle = torch.load(MODEL_BUNDLE, map_location="cpu")
        config = bundle["config"]
        adj_norm = np.array(bundle["adj_norm"], dtype=np.float32)

        model = TGCN(
            adj=torch.from_numpy(adj_norm),
            in_features=int(config["in_features"]),
            gcn_hidden=int(config["gcn_hidden"]),
            gru_hidden=int(config["gru_hidden"]),
            dropout=float(config["dropout"]),
        )
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        INFER_RUNTIME = {
            "ready": True,
            "bundle": bundle,
            "model": model,
            "nodes": list(bundle["nodes"]),
            "edges": [tuple(e) for e in bundle["edges"]],
            "mean": np.array(bundle["feature_mean"], dtype=np.float32).reshape(1, 1, 1, -1),
            "std": np.array(bundle["feature_std"], dtype=np.float32).reshape(1, 1, 1, -1),
        }
    except Exception as exc:
        INFER_RUNTIME = {"ready": False, "error": f"Failed to load model bundle: {exc}"}


def infer_latest(df: pd.DataFrame) -> Dict[str, object]:
    if not INFER_RUNTIME.get("ready", False):
        return {"ok": False, "message": INFER_RUNTIME.get("error", "Model unavailable")}

    bundle = INFER_RUNTIME["bundle"]
    config = bundle["config"]

    try:
        graph_data = build_graph_timeseries(df, window_seconds=float(config["window_seconds"]))
        if graph_data.nodes != INFER_RUNTIME["nodes"]:
            return {
                "ok": False,
                "message": f"Node mismatch. trained={INFER_RUNTIME['nodes']} live={graph_data.nodes}",
            }

        x, _ = build_sequences(
            graph_data.features,
            graph_data.labels,
            seq_len=int(config["seq_len"]),
            horizon=1,
        )
        if len(x) == 0:
            return {"ok": False, "message": "Not enough temporal windows for inference yet."}

        x_latest = x[-1:].astype(np.float32)
        mean = INFER_RUNTIME["mean"]
        std = np.where(INFER_RUNTIME["std"] < 1e-6, 1.0, INFER_RUNTIME["std"])
        x_latest = (x_latest - mean) / std

        with torch.no_grad():
            logits = INFER_RUNTIME["model"](torch.from_numpy(x_latest))
            probs = torch.sigmoid(logits).numpy()[0]

        node_scores = []
        for idx, node in enumerate(INFER_RUNTIME["nodes"]):
            score = float(probs[idx])
            node_scores.append(
                {
                    "node": node,
                    "score": score,
                    "prediction": "ALERT" if score >= 0.5 else "STABLE",
                }
            )

        node_scores.sort(key=lambda x: x["score"], reverse=True)
        high_risk = [n for n in node_scores if n["score"] >= 0.5]
        system_state = "HIGH-TRAFFIC / CRASH RISK" if high_risk else "STABLE"

        return {
            "ok": True,
            "state": system_state,
            "top_score": node_scores[0]["score"] if node_scores else 0.0,
            "node_scores": node_scores,
            "high_risk_nodes": [n["node"] for n in high_risk],
        }
    except Exception as exc:
        return {"ok": False, "message": f"Inference failed: {exc}"}


def _plot_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def build_metrics_plot(df: pd.DataFrame) -> str:
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No telemetry data yet", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return _plot_to_base64(fig)

    plot_df = df.tail(3500).copy()
    plot_df["dt"] = pd.to_datetime(plot_df["Timestamp"], unit="s", errors="coerce")
    plot_df = plot_df.dropna(subset=["dt"])

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Telemetry timestamps unavailable", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return _plot_to_base64(fig)

    metric_agg = (
        plot_df.groupby([pd.Grouper(key="dt", freq="15s"), "Source"])[["CPU_Usage", "Mem_Usage", "Latency_ms"]]
        .mean()
        .reset_index()
    )
    net_agg = (
        plot_df.groupby([pd.Grouper(key="dt", freq="15s"), "Source"])
        .size()
        .reset_index(name="Net_Usage")
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    services = sorted(plot_df["Source"].unique().tolist())

    for svc in services:
        svc_metric = metric_agg[metric_agg["Source"] == svc]
        svc_net = net_agg[net_agg["Source"] == svc]
        axes[0, 0].plot(svc_metric["dt"], svc_metric["CPU_Usage"], label=svc, linewidth=1.8)
        axes[0, 1].plot(svc_metric["dt"], svc_metric["Mem_Usage"], label=svc, linewidth=1.8)
        axes[1, 0].plot(svc_metric["dt"], svc_metric["Latency_ms"], label=svc, linewidth=1.8)
        axes[1, 1].plot(svc_net["dt"], svc_net["Net_Usage"], label=svc, linewidth=1.8)

    axes[0, 0].set_title("CPU Usage (%)")
    axes[0, 1].set_title("Memory Usage (%)")
    axes[1, 0].set_title("Latency (ms)")
    axes[1, 1].set_title("Network Usage (requests/15s)")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _plot_to_base64(fig)


def get_latest_logs(df: pd.DataFrame, n: int = 40) -> List[Dict[str, object]]:
    if df.empty:
        return []
    recent = df.tail(n).copy()
    recent["Time"] = pd.to_datetime(recent["Timestamp"], unit="s", errors="coerce").dt.strftime("%H:%M:%S")
    cols = ["Time", "Source", "Target", "Users", "Latency_ms", "CPU_Usage", "Mem_Usage", "Status"]
    for c in cols:
        if c not in recent.columns:
            recent[c] = ""
    rows = recent[cols].iloc[::-1].to_dict(orient="records")
    return rows


def traffic_driver_loop() -> None:
    while True:
        with CONTROL_LOCK:
            cfg = dict(CONTROL_STATE)

        if cfg["running"]:
            simulated_hour = 13 if cfg["mode"] == "day" else 2
            params = {
                "users": int(cfg["users"]),
                "simulated_hour": simulated_hour,
                "high_resource": False,
                "inject_crash": bool(cfg["force_crash"]),
            }

            for _ in range(max(1, int(cfg["req_per_cycle"]))):
                try:
                    resp = requests.get(APP_URL, params=params, timeout=20)
                    DRIVER_EVENTS.appendleft(
                        f"{time.strftime('%H:%M:%S')} req users={params['users']} mode={cfg['mode']} status={resp.status_code}"
                    )
                except Exception as exc:
                    DRIVER_EVENTS.appendleft(f"{time.strftime('%H:%M:%S')} driver error: {exc}")
                    break

        time.sleep(max(0.5, float(cfg["interval_sec"])))


HOME_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>TGCN Live Inference Home</title>
  <meta http-equiv="refresh" content="5">
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f7fafc; color: #111827; }
    .top { display: flex; justify-content: space-between; align-items: center; }
    .badge { padding: 8px 12px; border-radius: 8px; font-weight: 700; }
    .risk { background: #fee2e2; color: #991b1b; }
    .stable { background: #dcfce7; color: #166534; }
    .card { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-top: 16px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 8px; font-size: 13px; text-align: left; }
    th { background: #f3f4f6; }
    a.button { background: #111827; color: white; padding: 8px 12px; border-radius: 8px; text-decoration: none; }
  </style>
</head>
<body>
  <div class="top">
    <h1>/home - TGCN Live Inference</h1>
    <a class="button" href="{{ url_for('control') }}">Go to /control</a>
  </div>

  <div class="card">
    {% if pred.ok %}
      <div class="badge {{ 'risk' if pred.high_risk_nodes else 'stable' }}">System State: {{ pred.state }}</div>
      <p>Top score: {{ '%.3f'|format(pred.top_score) }}</p>
      <ul>
        {% for item in pred.node_scores %}
          <li>{{ item.node }} -> {{ '%.3f'|format(item.score) }} ({{ item.prediction }})</li>
        {% endfor %}
      </ul>
    {% else %}
      <div class="badge risk">Inference unavailable</div>
      <p>{{ pred.message }}</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Resource Visualization (CPU, Memory, Latency, Network)</h2>
    <img src="data:image/png;base64,{{ metrics_img }}" style="width:100%; max-width:1300px;"/>
  </div>

  <div class="card">
    <h2>Latest Service Logs</h2>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Source</th><th>Target</th><th>Users</th><th>Latency (ms)</th><th>CPU %</th><th>Mem %</th><th>Status</th>
        </tr>
      </thead>
      <tbody>
      {% for row in logs %}
        <tr>
          <td>{{ row.Time }}</td>
          <td>{{ row.Source }}</td>
          <td>{{ row.Target }}</td>
          <td>{{ row.Users }}</td>
          <td>{{ '%.2f'|format(row.Latency_ms) if row.Latency_ms != '' else '' }}</td>
          <td>{{ '%.2f'|format(row.CPU_Usage) if row.CPU_Usage != '' else '' }}</td>
          <td>{{ '%.2f'|format(row.Mem_Usage) if row.Mem_Usage != '' else '' }}</td>
          <td>{{ row.Status }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


CONTROL_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>TGCN Control Panel</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #111827; }
    .card { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); max-width: 850px; }
    label { display: block; margin-top: 10px; font-weight: 600; }
    input, select { width: 100%; padding: 8px; margin-top: 6px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    button { margin-top: 12px; margin-right: 10px; padding: 8px 12px; }
    .events { background: #0f172a; color: #e2e8f0; padding: 10px; border-radius: 8px; font-family: monospace; max-height: 260px; overflow-y: auto; }
    a { text-decoration: none; }
  </style>
</head>
<body>
  <h1>/control - Controlled Load Panel</h1>
  <p><a href="{{ url_for('home') }}">Back to /home</a></p>

  <div class="card">
    <form method="post">
      <div class="row">
        <div>
          <label>Number of Users</label>
          <input type="number" name="users" min="1" max="2000" value="{{ state.users }}" required>
        </div>
        <div>
          <label>Day/Night Mode</label>
          <select name="mode">
            <option value="day" {% if state.mode == 'day' %}selected{% endif %}>Day</option>
            <option value="night" {% if state.mode == 'night' %}selected{% endif %}>Night</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Requests per Cycle</label>
          <input type="number" name="req_per_cycle" min="1" max="30" value="{{ state.req_per_cycle }}" required>
        </div>
        <div>
          <label>Cycle Interval (seconds)</label>
          <input type="number" step="0.1" name="interval_sec" min="0.5" max="30" value="{{ state.interval_sec }}" required>
        </div>
      </div>

      <label><input type="checkbox" name="force_crash" value="1" {% if state.force_crash %}checked{% endif %}> Force Crash Injection</label>

      <div>
        <button type="submit" name="action" value="save">Save Controls</button>
        <button type="submit" name="action" value="start">Start Driver</button>
        <button type="submit" name="action" value="stop">Stop Driver</button>
        <button type="submit" name="action" value="reset_logs">Reset Telemetry Logs</button>
      </div>
    </form>

    <p><strong>Driver status:</strong> {{ 'RUNNING' if state.running else 'STOPPED' }}</p>
    <p><strong>App URL:</strong> {{ app_url }}</p>

    <h3>Driver Events</h3>
    <div class="events">
      {% for e in events %}
        <div>{{ e }}</div>
      {% endfor %}
    </div>
  </div>
</body>
</html>
"""


@app.route("/")
def root():
    return redirect(url_for("home"))


@app.route("/home")
def home():
    df = read_telemetry()
    pred = infer_latest(df)
    metrics_img = build_metrics_plot(df)
    logs = get_latest_logs(df, n=50)
    return render_template_string(HOME_TEMPLATE, pred=pred, metrics_img=metrics_img, logs=logs)


@app.route("/control", methods=["GET", "POST"])
def control():
    if request.method == "POST":
        action = request.form.get("action", "save")
        with CONTROL_LOCK:
            if action in {"save", "start", "stop"}:
                CONTROL_STATE["users"] = max(1, min(2000, int(request.form.get("users", CONTROL_STATE["users"]))))
                mode = request.form.get("mode", str(CONTROL_STATE["mode"]))
                CONTROL_STATE["mode"] = "night" if mode == "night" else "day"
                CONTROL_STATE["req_per_cycle"] = max(
                    1, min(30, int(request.form.get("req_per_cycle", CONTROL_STATE["req_per_cycle"])))
                )
                CONTROL_STATE["interval_sec"] = max(
                    0.5, min(30.0, float(request.form.get("interval_sec", CONTROL_STATE["interval_sec"])))
                )
                CONTROL_STATE["force_crash"] = request.form.get("force_crash") == "1"

            if action == "start":
                CONTROL_STATE["running"] = True
            elif action == "stop":
                CONTROL_STATE["running"] = False
            elif action == "reset_logs":
                reset_log_file()
                DRIVER_EVENTS.appendleft(f"{time.strftime('%H:%M:%S')} telemetry log reset")

    with CONTROL_LOCK:
        state = dict(CONTROL_STATE)
    return render_template_string(
        CONTROL_TEMPLATE,
        state=state,
        events=list(DRIVER_EVENTS),
        app_url=APP_URL,
    )


@app.route("/api/state")
def api_state():
    with CONTROL_LOCK:
        state = dict(CONTROL_STATE)
    return {
        "control": state,
        "driver_events": list(DRIVER_EVENTS),
        "model_ready": INFER_RUNTIME.get("ready", False),
        "model_error": INFER_RUNTIME.get("error", ""),
    }


def start_background_driver() -> None:
    thread = threading.Thread(target=traffic_driver_loop, daemon=True)
    thread.start()


if __name__ == "__main__":
    init_log_file()
    load_model_bundle()
    start_background_driver()
    app.run(host="0.0.0.0", port=PORT, debug=False)
