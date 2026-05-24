# Temporal Graph Convolutional Network (T-GCN)

This project utilizes a **Temporal Graph Convolutional Network (T-GCN)** to analyze and predict service health within a microservices architecture. T-GCN is designed to capture both **spatial dependencies** (how services affect each other) and **temporal dynamics** (how service metrics change over time).

---

## 1. Conceptual Overview

T-GCN is a fusion of two powerful neural network architectures:
1.  **Graph Convolutional Network (GCN):** Used to extract spatial features from the graph topology. It captures the influence of neighboring nodes (services) on a target node.
2.  **Gated Recurrent Unit (GRU):** Used to capture temporal patterns from time-series data. It maintains a "memory" of past states to predict future behavior.

By combining these, T-GCN can understand that a performance dip in `Service A` might be caused by a previous failure in its dependency `Service B` (spatial) and that this failure has been escalating over the last few minutes (temporal).

---

## 2. Mathematical Foundation

### A. Graph Representation
The microservice architecture is represented as a graph $G = (V, E, A)$:
- $V$: A set of nodes representing services (e.g., `app`, `svc1`, `svc2`).
- $E$: A set of edges representing communication/dependencies between services.
- $A \in \mathbb{R}^{N \times N}$: The adjacency matrix, where $A_{ij} = 1$ if there is a connection between service $i$ and $j$, and $0$ otherwise.

### B. Spatial Component: Graph Convolution (GCN)
The GCN captures the spatial relationship by aggregating features from a node's neighbors. The operation for a single layer is:

$$f(X, A) = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} X W)$$

Where:
- $\hat{A} = A + I_N$: Adjacency matrix with added self-loops (so a node considers its own features).
- $\hat{D}$: Degree matrix of $\hat{A}$, where $\hat{D}_{ii} = \sum_j \hat{A}_{ij}$.
- $X \in \mathbb{R}^{N \times F}$: Feature matrix ( $N$ nodes, $F$ features).
- $W$: Learnable weight matrix.
- $\sigma$: Activation function (e.g., ReLU).

In this project, we use a normalized adjacency matrix $\tilde{A} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$ to ensure numerical stability during training.

### C. Temporal Component: Gated Recurrent Unit (GRU)
The GRU processes the spatial features extracted by the GCN over a sequence of time steps $t=1, \dots, T$. For each time step $t$, the GRU computes:

1.  **Update Gate ($u_t$):** Decides how much of the previous state to keep.
    $$u_t = \sigma(W_u \cdot [GC(X_t, A), h_{t-1}] + b_u)$$
2.  **Reset Gate ($r_t$):** Decides how much of the previous state to forget.
    $$r_t = \sigma(W_r \cdot [GC(X_t, A), h_{t-1}] + b_r)$$
3.  **Candidate Memory ($\tilde{h}_t$):**
    $$\tilde{h}_t = \tanh(W_{\tilde{h}} \cdot [GC(X_t, A), (r_t \odot h_{t-1})] + b_{\tilde{h}})$$
4.  **Final Hidden State ($h_t$):**
    $$h_t = u_t \odot h_{t-1} + (1 - u_t) \odot \tilde{h}_t$$

Where $GC(X_t, A)$ is the output of the GCN layer at time $t$, and $h_{t-1}$ is the hidden state from the previous time step.

---

## 3. Project-Specific Implementation

### A. Data Processing Pipeline (`tgcn_pipeline.py`)
-   **Telemetry Data:** Collected from `telemetry_data.csv`, including `Latency_ms`, `CPU_Usage`, `Mem_Usage`, `Users`, and `Status`.
-   **Node Features:** For each service (node), 12 features are calculated per time window:
    - Average Latency, CPU, Memory, Users.
    - Incoming/Outgoing request counts.
    - Periodic time encoding (Sine/Cosine of the hour).
    - Rate of OK, Error, and Crash statuses.
-   **Graph Construction:** Built from the `Source` and `Target` columns. If `Service A` calls `Service B`, an edge is created. The adjacency matrix is symmetric (undirected) to allow bidirectional influence in the spatial convolution.
-   **Labeling:** A service is labeled "Critical" (1) if it exceeds p90 thresholds for resources/latency or reports crashes/errors; otherwise, it is "Healthy" (0).

### B. Model Architecture (`TGCN` class)
1.  **GCN Layer:** Processes node features for each time step using the pre-computed normalized adjacency matrix.
2.  **GRU Cell:** Iterates through a sequence of 12 time windows (default `seq-len`), updating the hidden state of each node based on the spatial features.
3.  **Classifier:** A fully connected linear layer that maps the final GRU hidden state to a single logit per node.
4.  **Output:** Sigmoid activation is applied to the logit to get the probability of the service being in a critical state at $t + 1$ (default `horizon`).

### C. Training & Evaluation (`train_tgcn.py`)
-   **Loss Function:** `BCEWithLogitsLoss` with positive class weighting to handle class imbalance (healthy services are more common than critical ones).
-   **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-score are calculated on a test set to ensure the model generalizes well.
-   **Visualization:** 
    - `training_loss_accuracy.png`: Monitors convergence.
    - `graph_temporal_instances.png`: Visualizes the graph topology and node states over various time snapshots.

---

## 4. Project Summary & TGCN Usage

### How TGCN is used in this project
The TGCN model in this project is used as a **predictive monitoring engine** for the microservices architecture. It treats the entire distributed system as a dynamic graph to transition from reactive troubleshooting to proactive failure prediction.

*   **Spatial Features (GCN):** Captures the dependencies between services. For example, if `Svc2` is failing, the GCN layer understands its relationship with `Svc1` (the caller) and `Svc3` (the dependency), allowing it to model cascading failures.
*   **Temporal Features (GRU):** Captures the "trend" of metrics over time. The GRU layer analyzes a sequence of previous states (e.g., the last 12 windows of 5 seconds each) to identify patterns like gradual memory leaks or slow-building traffic spikes.
*   **Node Features:** Each node (service) in the graph has a 12-dimensional feature vector (latency, CPU, memory, etc.) along with temporal context (hour encoding).

### How the project works (Workflow)

The project follows a complete lifecycle from data generation to live inference:

#### Step A: Data Generation & Collection
1.  **Microservices Stack:** A set of services (`App`, `Svc1`, `Svc2`, `Svc3`) runs in Docker containers.
2.  **Telemetry Pipeline:** OpenTelemetry collectors gather traces and metrics, while Prometheus and Jaeger store them.
3.  **Traffic Generator:** A script (`traffic_generator.py`) simulates realistic traffic, injecting both normal patterns and abnormalities (DDoS, resource exhaustion, crashes).
4.  **Telemetry Aggregation:** All data is logged into `data/telemetry_data.csv`.

#### Step B: Training Phase (`train_tgcn.py`)
1.  **Graph Construction:** The system reads the CSV and builds an adjacency matrix based on service-to-service call traces.
2.  **Windowing & Labeling:** It slices telemetry data into discrete windows and labels them as "abnormal" if metrics exceed thresholds or crashes are detected.
3.  **Training:** The TGCN model is trained to predict the state of the *next* window based on a sequence of previous ones.
4.  **Artifacts:** The trained weights and metadata are saved into an `inference_bundle.pt`.

#### Step C: Inference & Monitoring (`predict_tgcn.py` & Portal)
1.  **Live Prediction:** The inference script takes the latest window of telemetry and predicts which services are most likely to experience an anomaly in the near future.
2.  **Visualization Portal:** A Flask-based web interface displays live risk scores, telemetry trends, and allows manual fault injection to test the model.

### Summary of Component Roles
| Component | Purpose |
| :--- | :--- |
| `src/tgcn_pipeline.py` | Defines the TGCN model architecture (GCN + GRU) and data preprocessing logic. |
| `src/train_tgcn.py` | Trains the model on historical telemetry and generates evaluation plots. |
| `src/predict_tgcn.py` | Loads the trained model to perform real-time node risk classification. |
| `portal/app.py` | Provides the UI for controlling the simulation and viewing live TGCN insights. |
