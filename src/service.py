import math
import os
import time
from typing import Dict, Tuple

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


def init_log_file() -> None:
    header = "Timestamp,Hour_Sin,Hour_Cos,Source,Target,Latency_ms,CPU_Usage,Mem_Usage,Users,Status\n"
    if not os.path.isfile(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(header)


init_log_file()


def _is_day(simulated_hour: int) -> bool:
    return 6 <= simulated_hour <= 20


def _derive_flags(users: int, simulated_hour: int, explicit_high_resource: bool, explicit_crash: bool) -> Tuple[bool, bool]:
    is_day = _is_day(simulated_hour)
    high_resource_threshold = 420 if is_day else 260
    crash_threshold = 880 if is_day else 700

    derived_high_resource = users >= high_resource_threshold
    derived_crash = users >= crash_threshold

    high_resource = explicit_high_resource or derived_high_resource
    inject_crash = explicit_crash or derived_crash
    return high_resource, inject_crash


def _service_weight() -> float:
    weights = {"app": 1.25, "svc1": 1.10, "svc2": 1.15, "svc3": 0.95}
    return weights.get(SERVICE_NAME, 1.0)


def _compute_metrics(users: int, simulated_hour: int, high_resource: bool, inject_crash: bool) -> Dict[str, float]:
    is_day = _is_day(simulated_hour)
    load_factor = users / 500.0
    day_factor = 1.15 if is_day else 0.85
    stress = 1.45 if high_resource else 1.0
    crash_factor = 1.35 if inject_crash else 1.0
    service_factor = _service_weight()

    latency_ms = (85.0 + 460.0 * load_factor * day_factor) * stress * service_factor * crash_factor
    latency_ms = float(max(25.0, min(5000.0, latency_ms)))

    synthetic_cpu = (8.0 + 68.0 * load_factor * day_factor * service_factor) * stress
    synthetic_mem = (22.0 + 38.0 * load_factor * day_factor * service_factor) * stress

    if inject_crash:
        synthetic_cpu = min(99.0, synthetic_cpu + 8.0)
        synthetic_mem = min(99.0, synthetic_mem + 5.0)

    return {
        "latency_ms": float(latency_ms),
        "cpu": float(max(0.0, min(99.0, synthetic_cpu))),
        "mem": float(max(0.0, min(99.0, synthetic_mem))),
    }


def write_telemetry(target: str, latency: float, users: int, status: str, simulated_hour: int, cpu: float, mem: float) -> None:
    timestamp = time.time()
    hour_sin = math.sin(2 * math.pi * simulated_hour / 24.0)
    hour_cos = math.cos(2 * math.pi * simulated_hour / 24.0)

    line = (
        f"{timestamp},{hour_sin:.4f},{hour_cos:.4f},{SERVICE_NAME},{target},"
        f"{latency:.2f},{cpu:.2f},{mem:.2f},{users},{status}\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def simulate_workload(latency_ms: float) -> None:
    # Controlled synthetic service latency to emulate load without randomness.
    time.sleep(max(0.01, min(3.5, latency_ms / 1000.0)))


def call_service(target_name: str, url: str, params: dict, users: int, simulated_hour: int, high_resource: bool, inject_crash: bool) -> None:
    start = time.time()
    status = "OK"
    metric = _compute_metrics(users, simulated_hour, high_resource, inject_crash)

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except Exception:
        status = "Error"

    downstream_latency = (time.time() - start) * 1000.0
    write_telemetry(
        target=target_name,
        latency=downstream_latency,
        users=users,
        status=status,
        simulated_hour=simulated_hour,
        cpu=metric["cpu"],
        mem=metric["mem"],
    )

    if status == "Error":
        raise HTTPException(status_code=502, detail=f"Upstream error from {target_name}")


@app.get("/")
def handle_request(
    request: Request,
    users: int = 50,
    inject_crash: bool = False,
    high_resource: bool = False,
    simulated_hour: int = 12,
):
    users = max(1, min(2000, int(users)))
    simulated_hour = int(simulated_hour) % 24

    high_resource, inject_crash = _derive_flags(
        users=users,
        simulated_hour=simulated_hour,
        explicit_high_resource=bool(high_resource),
        explicit_crash=bool(inject_crash),
    )

    metric = _compute_metrics(users, simulated_hour, high_resource, inject_crash)
    start_time = time.time()
    simulate_workload(metric["latency_ms"])

    if inject_crash:
        own_latency = (time.time() - start_time) * 1000.0
        write_telemetry(
            target="self",
            latency=own_latency,
            users=users,
            status="Crash",
            simulated_hour=simulated_hour,
            cpu=metric["cpu"],
            mem=metric["mem"],
        )
        raise HTTPException(status_code=500, detail="Crash triggered by controlled overload")

    own_latency = (time.time() - start_time) * 1000.0
    write_telemetry(
        target="self",
        latency=own_latency,
        users=users,
        status="OK",
        simulated_hour=simulated_hour,
        cpu=metric["cpu"],
        mem=metric["mem"],
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
        call_service("svc1", "http://svc1:8000/", params, users, simulated_hour, high_resource, inject_crash)
        call_service("svc2", "http://svc2:8000/", params, users, simulated_hour, high_resource, inject_crash)
        call_service("svc3", "http://svc3:8000/", safe_params, users, simulated_hour, high_resource, False)
    elif SERVICE_NAME == "svc1":
        call_service("svc2", "http://svc2:8000/", params, users, simulated_hour, high_resource, inject_crash)
    elif SERVICE_NAME == "svc2":
        call_service("svc3", "http://svc3:8000/", safe_params, users, simulated_hour, high_resource, False)

    return {
        "status": "success",
        "service": SERVICE_NAME,
        "users": users,
        "simulated_hour": simulated_hour,
        "day_mode": "day" if _is_day(simulated_hour) else "night",
        "derived_high_resource": high_resource,
        "derived_crash": inject_crash,
        "latency_ms": round(metric["latency_ms"], 2),
        "cpu": round(metric["cpu"], 2),
        "mem": round(metric["mem"], 2),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": SERVICE_NAME}
