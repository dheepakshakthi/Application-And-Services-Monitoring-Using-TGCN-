# T-GCN Monitoring Framework Implementation Plan

## Background & Motivation
Modern microservice architectures suffer from cascading dependencies that static threshold monitoring cannot effectively capture. This project implements a novel predictive monitoring framework using Temporal Graph Convolutional Networks (T-GCNs). By representing the infrastructure as a dynamic interaction graph (nodes = microservices, edges = network traces), we aim to transition from reactive troubleshooting to proactive failure prediction and automated Root Cause Analysis (RCA).

## Scope & Impact
- **Scope**: Development of a custom dummy microservice application (testbed) with fault-injection capabilities, a local telemetry agent, a centralized web portal (akin to Hadoop's management interfaces), and a Machine Learning pipeline for real-time anomaly detection.
- **Impact**: Reduced Mean Time to Detection (MTTD), improved failure forecasting, and automated bottleneck identification in distributed environments, validated against our controlled testbed.

## Proposed Solution
The system will be built using the **OpenTelemetry + Prometheus + Deep Graph Library (DGL)** stack, optimizing for cloud-native integration and standardized tooling.

1.  **Data Collection (Local Agent / Sidecar)**
    *   Deploy **OpenTelemetry (OTel) Collectors** as sidecars or daemonsets to gather distributed traces (representing edges) and metrics (CPU, Memory, Latency, representing node features).
    *   Use **Prometheus** to scrape and store metric time-series data.
    *   Use **Jaeger** or an OTel backend to store trace span data.

2.  **Data Aggregation & Graph Construction (Web Portal/Backend)**
    *   A custom backend service (Node.js or Go) will periodically query Prometheus and the trace backend to construct the dynamic interaction graph at discrete time steps $t$.
    *   The web portal (React) will visualize the live service mesh, highlighting predicted failure paths and current anomalies.

3.  **Machine Learning Pipeline (T-GCN Model)**
    *   The model will be developed using Python and **DGL (Deep Graph Library)** combined with PyTorch.
    *   **Architecture**: A GCN layer to extract spatial features from the microservice topology, followed by an LSTM or GRU layer to capture temporal performance degradation over time.
    *   **Initial Training Data**: We will utilize the **Alibaba Cluster Trace** dataset, which provides massive-scale, highly complex microservice topology data to pre-train the model's structural awareness.

## Phased Implementation Plan

### Phase 1: Testbed & Data Pipeline Foundation
1.  Setup local development environment (Docker Compose).
2.  Develop a **Custom Dummy Microservice App**:
    *   Create a simple topology (e.g., API Gateway -> Service A -> Service B & DB).
    *   Implement fault-injection endpoints (e.g., `/inject-latency`, `/inject-cpu-spike`, `/inject-error-logs`) to simulate abnormal activities.
3.  Instrument the dummy services with OpenTelemetry.
4.  Configure Prometheus and OTel Collector to gather metrics, traces, and logs.

### Phase 2: ML Model Development
1.  Download and preprocess the Alibaba Cluster Trace dataset into temporal graph snapshots.
2.  Develop the T-GCN model in PyTorch + DGL (Graph Convolution -> LSTM -> Linear Classifier).
3.  Train the model on the Alibaba dataset for node classification (anomaly vs. normal).

### Phase 3: Integration & Backend
1.  Develop the aggregation backend to build live graph snapshots from Prometheus/OTel data.
2.  Integrate the PyTorch inference script into the backend to run periodic predictions on the live graph.

### Phase 4: Web Portal Visualization
1.  Develop the React web portal to query the backend.
2.  Implement a dynamic graph visualization (e.g., using D3.js or Cytoscape.js) to display real-time node health, metric gauges, and predicted RCA paths.

## Verification & Testing
- **Unit Tests**: Ensure graph construction accurately maps OTel traces to adjacency matrices.
- **Model Validation**: Evaluate the T-GCN model using precision, recall, and F1-score on a holdout set of the Alibaba dataset.
- **Integration Tests**: Inject artificial CPU spikes or network delays into the testbed and verify that the web portal correctly flags the anomaly and predicts the root cause node.

## Migration & Rollback
- The local agents (OTel collectors) operate non-intrusively and can be disabled or removed without affecting the core application microservices.
- The web portal and ML backend are deployed as isolated services and will not impact application uptime during deployment.

# Telemetry Data Generator for TGCN

This project sets up a distributed microservices architecture to generate realistic telemetry data for training a Temporal Graph Convolutional Network (TGCN). The generated dataset is automatically saved to `data/telemetry_data.csv`.

## Architecture Overview
- **App Service**: Entry point. Calls Svc1, Svc2, and Svc3.
- **Svc1**: Calls Svc2.
- **Svc2**: Calls Svc3.
- **Svc3**: Leaf node.
- **OpenTelemetry Collector, Prometheus, Jaeger**: Full observability stack included to simulate a real-world environment.

## Generated Telemetry Data
The telemetry data is stored in `data/telemetry_data.csv` with the following columns:
1. `Timestamp`: Unix timestamp.
2. `Source`: The service making the request (or 'self' for the host processing the request).
3. `Target`: The downstream target service.
4. `Latency_ms`: Request processing or downstream response time.
5. `CPU_Usage`: System CPU utilization at request time.
6. `Mem_Usage`: System Memory utilization at request time.
7. `Users`: Simulated concurrent users.
8. `Status`: 'OK', 'Error', or 'Crash'.

## Normal and Abnormal Traffic
The `traffic_generator.py` script mimics:
- **Normal Day/Night Cycles:** High traffic during the day (simulated hours 6-20), lower traffic at night.
- **Abnormalities injected randomly:**
  - **DDoS/Batch Job:** Huge traffic spikes occurring in the middle of the night.
  - **Resource Exhaustion:** High CPU/Memory usage during normal hours.
  - **Service Crashes:** Simulated random service failures resulting in `Crash` logs and cascading HTTP errors.

## How to run

1. **Start the microservices:**
   ```bash
   docker-compose up -d --build
   ```
2. **Run the Traffic Generator:**
   ```bash
   pip install requests
   python traffic_generator.py
   ```
   Leave it running as long as you want to collect data.

3. **Check the Output:**
   The `telemetry_data.csv` file will be continuously populated inside the `./data` directory.
   - Traces are also accessible at `http://localhost:16686` (Jaeger)
   - Metrics are available at `http://localhost:9090` (Prometheus)

To stop the services later:
```bash
docker-compose down
```

## TGCN Training and Inference Pipeline

The repository now includes a complete TGCN training pipeline for `data/telemetry_data.csv`:

- **Trainer**: `src/train_tgcn.py`
- **Inference script**: `src/predict_tgcn.py`
- **Reusable model/data utilities**: `src/tgcn_pipeline.py`

### Run with the existing virtual environment in `master`

```powershell
.\master\Scripts\python.exe src\train_tgcn.py --data data\telemetry_data.csv --output artifacts\tgcn
```

This generates:

- `artifacts/tgcn/tgcn_state_dict.pt` (model weights)
- `artifacts/tgcn/tgcn_inference_bundle.pt` (weights + config + graph/scaler metadata)
- `artifacts/tgcn/training_report.json` (evaluation summary)
- `artifacts/tgcn/plots/graph_temporal_instances.png` (nodes, edges, temporal graph snapshots)
- `artifacts/tgcn/plots/training_loss_accuracy.png` (training loss/accuracy curves)

### Run inference on latest telemetry windows

```powershell
.\master\Scripts\python.exe src\predict_tgcn.py --data data\telemetry_data.csv --model-bundle artifacts\tgcn\tgcn_inference_bundle.pt
```

## Live Controlled Inference Portal

The Docker stack now includes a Flask `portal` service for controlled load generation and live TGCN inference.

- `http://localhost:5050/control`: control panel for `users` and `day/night` mode (plus driver controls).
- `http://localhost:5050/home`: live inference status + CPU/memory/network/latency charts + latest service logs.

### Workflow

1. Train once to create the model bundle:
   ```powershell
   .\master\Scripts\python.exe src\train_tgcn.py --data data\telemetry_data.csv --output artifacts\tgcn
   ```
2. Start services including the portal:
   ```bash
   docker-compose up -d --build
   ```
3. Open `/control`, set `users` and `day/night`, then start the driver.
4. Open `/home` to monitor live TGCN predictions and telemetry trends.
