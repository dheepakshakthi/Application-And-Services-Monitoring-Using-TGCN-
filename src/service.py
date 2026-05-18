import csv
import math
import os
import random
import time
from typing import Optional

import psutil
import requests
from fastapi import FastAPI, HTTPException, Request
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown")
LOG_FILE = os.getenv("LOG_FILE", "/data/telemetry_data.csv")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")

resource = Resource(attributes={"service.name": SERVICE_NAME})

trace_provider = TracerProvider(resource=resource)
trace_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(trace_provider)

metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True)
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()


def init_log_file():
    try:
        if not os.path.isfile(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
            with open(LOG_FILE, "w") as f:
                f.write(
                    "Timestamp,Hour_Sin,Hour_Cos,Source,Target,Latency_ms,CPU_Usage,Mem_Usage,Users,Status\n"
                )
    except Exception:
        pass


init_log_file()


def write_telemetry(
    target: str, latency: float, users: int, status: str, simulated_hour: int
):
    try:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        timestamp = time.time()

        # Cyclical encoding
        hour_sin = math.sin(2 * math.pi * simulated_hour / 24.0)
        hour_cos = math.cos(2 * math.pi * simulated_hour / 24.0)

        line = f"{timestamp},{hour_sin:.4f},{hour_cos:.4f},{SERVICE_NAME},{target},{latency:.2f},{cpu},{mem},{users},{status}\n"
        with open(LOG_FILE, "a") as f:
            f.write(line)
    except Exception as e:
        print(f"Failed to write telemetry: {e}")


def simulate_workload(is_high_resource: bool):
    iterations = 2000000 if is_high_resource else 100000
    x = 0
    for _ in range(iterations):
        x += 1
    sleep_time = (
        random.uniform(0.5, 1.5) if is_high_resource else random.uniform(0.05, 0.2)
    )
    time.sleep(sleep_time)


def call_service(target_name: str, url: str, params: dict):
    start = time.time()
    status = "OK"
    simulated_hour = int(params.get("simulated_hour", 0))
    users = int(params.get("users", 1))
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        status = "Error"
    latency = (time.time() - start) * 1000

    write_telemetry(target_name, latency, users, status, simulated_hour)

    if status == "Error":
        raise HTTPException(
            status_code=502, detail=f"Upstream error from {target_name}"
        )


@app.get("/")
def handle_request(
    request: Request,
    users: int = 1,
    inject_crash: bool = False,
    high_resource: bool = False,
    simulated_hour: int = 0,
):
    start_time = time.time()

    simulate_workload(high_resource)

    if inject_crash:
        write_telemetry(
            "self", (time.time() - start_time) * 1000, users, "Crash", simulated_hour
        )
        raise HTTPException(status_code=500, detail="Injected crash")

    write_telemetry(
        "self", (time.time() - start_time) * 1000, users, "OK", simulated_hour
    )

    params = {
        "users": users,
        "inject_crash": inject_crash,
        "high_resource": high_resource,
        "simulated_hour": simulated_hour,
    }
    safe_params = {
        "users": users,
        "inject_crash": False,
        "high_resource": high_resource,
        "simulated_hour": simulated_hour,
    }

    if SERVICE_NAME == "app":
        call_service("svc1", "http://svc1:8000/", params)
        call_service("svc2", "http://svc2:8000/", params)
        call_service("svc3", "http://svc3:8000/", safe_params)
    elif SERVICE_NAME == "svc1":
        call_service("svc2", "http://svc2:8000/", params)
    elif SERVICE_NAME == "svc2":
        call_service("svc3", "http://svc3:8000/", safe_params)

    return {"status": "success", "service": SERVICE_NAME}
