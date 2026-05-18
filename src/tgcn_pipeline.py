import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, adj: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("adj", adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        out = torch.einsum("ij,bjf->bif", self.adj, support) + self.bias
        return out


class TGCN(nn.Module):
    def __init__(
        self,
        adj: torch.Tensor,
        in_features: int,
        gcn_hidden: int,
        gru_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gcn_hidden = gcn_hidden
        self.gru_hidden = gru_hidden
        self.gcn = GraphConvolution(in_features, gcn_hidden, adj)
        self.gru_cell = nn.GRUCell(gcn_hidden, gru_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_nodes, _ = x_seq.shape
        h = torch.zeros(batch_size, num_nodes, self.gru_hidden, device=x_seq.device)
        for t in range(x_seq.size(1)):
            x_t = x_seq[:, t, :, :]
            spatial = torch.relu(self.gcn(x_t))
            spatial = self.dropout(spatial)
            h = self.gru_cell(
                spatial.reshape(batch_size * num_nodes, self.gcn_hidden),
                h.reshape(batch_size * num_nodes, self.gru_hidden),
            ).reshape(batch_size, num_nodes, self.gru_hidden)
        logits = self.classifier(h).squeeze(-1)
        return logits


@dataclass
class GraphTimeSeriesData:
    features: np.ndarray
    labels: np.ndarray
    nodes: List[str]
    edges: List[Tuple[str, str]]
    edge_activity: np.ndarray
    thresholds: Dict[str, float]


def _safe_numeric(df: pd.DataFrame, col: str, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(fill)


def load_telemetry(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Timestamp", "Source", "Target", "Latency_ms", "CPU_Usage", "Mem_Usage", "Users", "Status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Telemetry file missing columns: {sorted(missing)}")

    df["Timestamp"] = _safe_numeric(df, "Timestamp")
    df["Latency_ms"] = _safe_numeric(df, "Latency_ms")
    df["CPU_Usage"] = _safe_numeric(df, "CPU_Usage")
    df["Mem_Usage"] = _safe_numeric(df, "Mem_Usage")
    df["Users"] = _safe_numeric(df, "Users")
    df["Status"] = df["Status"].fillna("OK").astype(str)
    df["Source"] = df["Source"].astype(str)
    df["Target"] = df["Target"].astype(str)

    if "Hour_Sin" in df.columns and "Hour_Cos" in df.columns:
        df["Hour_Sin"] = _safe_numeric(df, "Hour_Sin")
        df["Hour_Cos"] = _safe_numeric(df, "Hour_Cos")
    else:
        # Fall back to timestamp-based cyclical encoding if columns are absent.
        dt = pd.to_datetime(df["Timestamp"], unit="s", errors="coerce")
        hour = dt.dt.hour.fillna(0).astype(int)
        df["Hour_Sin"] = np.sin(2.0 * math.pi * hour / 24.0)
        df["Hour_Cos"] = np.cos(2.0 * math.pi * hour / 24.0)

    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def build_graph_timeseries(df: pd.DataFrame, window_seconds: float) -> GraphTimeSeriesData:
    nodes = sorted(df["Source"].dropna().unique().tolist())
    if not nodes:
        raise ValueError("No source nodes found in telemetry data.")
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    edge_pairs = (
        df.loc[df["Target"] != "self", ["Source", "Target"]]
        .drop_duplicates()
        .sort_values(["Source", "Target"])
    )
    edges = [(r["Source"], r["Target"]) for _, r in edge_pairs.iterrows()]
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    latency_threshold = float(df["Latency_ms"].quantile(0.90))
    users_threshold = float(df["Users"].quantile(0.90))
    cpu_threshold = float(df["CPU_Usage"].quantile(0.90))
    mem_threshold = float(df["Mem_Usage"].quantile(0.90))
    thresholds = {
        "latency_p90": latency_threshold,
        "users_p90": users_threshold,
        "cpu_p90": cpu_threshold,
        "mem_p90": mem_threshold,
    }

    min_ts = float(df["Timestamp"].min())
    df["bin_idx"] = ((df["Timestamp"] - min_ts) / window_seconds).astype(int)
    max_bin = int(df["bin_idx"].max())
    num_windows = max_bin + 1
    num_nodes = len(nodes)
    num_features = 12
    num_edges = len(edges)

    features = np.zeros((num_windows, num_nodes, num_features), dtype=np.float32)
    labels = np.zeros((num_windows, num_nodes), dtype=np.float32)
    edge_activity = np.zeros((num_windows, num_edges), dtype=np.float32)

    grouped = {idx: g for idx, g in df.groupby("bin_idx", sort=True)}
    for w in range(num_windows):
        g = grouped.get(w)
        if g is None:
            continue

        for edge_key, edge_grp in g[g["Target"] != "self"].groupby(["Source", "Target"]):
            edge_idx = edge_to_idx.get(edge_key)
            if edge_idx is not None:
                edge_activity[w, edge_idx] = float(len(edge_grp))

        for node in nodes:
            idx = node_to_idx[node]
            node_rows = g[g["Source"] == node]
            if node_rows.empty:
                continue

            out_count = float((node_rows["Target"] != "self").sum())
            in_count = float(((g["Target"] == node) & (g["Source"] != node)).sum())
            latency_mean = float(node_rows["Latency_ms"].mean())
            cpu_mean = float(node_rows["CPU_Usage"].mean())
            mem_mean = float(node_rows["Mem_Usage"].mean())
            users_mean = float(node_rows["Users"].mean())
            hour_sin_mean = float(node_rows["Hour_Sin"].mean())
            hour_cos_mean = float(node_rows["Hour_Cos"].mean())
            request_count = float(len(node_rows))
            crash_rate = float((node_rows["Status"].str.lower() == "crash").mean())
            error_rate = float((node_rows["Status"].str.lower() == "error").mean())
            ok_rate = float((node_rows["Status"].str.lower() == "ok").mean())

            features[w, idx, :] = np.array(
                [
                    latency_mean,
                    cpu_mean,
                    mem_mean,
                    users_mean,
                    out_count,
                    in_count,
                    request_count,
                    hour_sin_mean,
                    hour_cos_mean,
                    crash_rate,
                    error_rate,
                    ok_rate,
                ],
                dtype=np.float32,
            )

            high_resource = cpu_mean >= cpu_threshold or mem_mean >= mem_threshold
            high_traffic = users_mean >= users_threshold or latency_mean >= latency_threshold
            has_crash_error = crash_rate > 0.0 or error_rate > 0.0
            labels[w, idx] = 1.0 if (high_resource or high_traffic or has_crash_error) else 0.0

    return GraphTimeSeriesData(
        features=features,
        labels=labels,
        nodes=nodes,
        edges=edges,
        edge_activity=edge_activity,
        thresholds=thresholds,
    )


def build_normalized_adjacency(nodes: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
    n = len(nodes)
    node_to_idx = {name: i for i, name in enumerate(nodes)}
    adj = np.zeros((n, n), dtype=np.float32)

    for src, dst in edges:
        if src not in node_to_idx or dst not in node_to_idx:
            continue
        i = node_to_idx[src]
        j = node_to_idx[dst]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    adj += np.eye(n, dtype=np.float32)
    degree = np.sum(adj, axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-8)))
    adj_norm = inv_sqrt @ adj @ inv_sqrt
    return adj_norm.astype(np.float32)


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_samples = []
    y_samples = []
    total_windows = features.shape[0]
    for t in range(seq_len - 1, total_windows - horizon):
        x_samples.append(features[t - seq_len + 1 : t + 1])
        y_samples.append(labels[t + horizon])

    if not x_samples:
        raise ValueError(
            "Not enough temporal windows to form training sequences. "
            f"Need more data or smaller seq_len/horizon (windows={total_windows})."
        )
    return np.stack(x_samples).astype(np.float32), np.stack(y_samples).astype(np.float32)


def split_dataset(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(x)
    train_end = max(1, int(n * 0.70))
    val_end = max(train_end + 1, int(n * 0.85))
    val_end = min(val_end, n - 1)
    return {
        "x_train": x[:train_end],
        "y_train": y[:train_end],
        "x_val": x[train_end:val_end],
        "y_val": y[train_end:val_end],
        "x_test": x[val_end:],
        "y_test": y[val_end:],
    }


def normalize_features(data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    x_train = data["x_train"]
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    scaled = {}
    for k, v in data.items():
        if k.startswith("x_"):
            scaled[k] = (v - mean) / std
        else:
            scaled[k] = v
    scaler = {"mean": mean.squeeze().astype(np.float32), "std": std.squeeze().astype(np.float32)}
    return scaled, scaler


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _logit_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == targets).float().mean().item()
    return acc, preds.mean().item()


def train_model(
    model: TGCN,
    data: Dict[str, np.ndarray],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    y_train = data["y_train"]
    positives = float(y_train.sum())
    negatives = float(y_train.size - positives)
    if positives <= 0.0:
        pos_weight = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor(max(1.0, negatives / positives), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )

    train_loader = _make_loader(data["x_train"], data["y_train"], batch_size, shuffle=True)
    val_loader = _make_loader(data["x_val"], data["y_val"], batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            acc, _ = _logit_metrics(logits.detach(), yb)
            train_loss += loss.item()
            train_acc += acc
            batches += 1

        train_loss /= max(batches, 1)
        train_acc /= max(batches, 1)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        v_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                acc, _ = _logit_metrics(logits, yb)
                val_loss += loss.item()
                val_acc += acc
                v_batches += 1

        val_loss /= max(v_batches, 1)
        val_acc /= max(v_batches, 1)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_model(model: TGCN, x_test: np.ndarray, y_test: np.ndarray, device: torch.device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(x_test).to(device)
        yb = torch.from_numpy(y_test).to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        y_true = yb.cpu().numpy().reshape(-1)
        y_pred = (probs >= 0.5).astype(np.int32)

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "test_accuracy": acc,
        "test_precision": float(p),
        "test_recall": float(r),
        "test_f1": float(f1),
    }


def plot_graph_temporal_instances(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    edge_activity: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    snapshots: int = 6,
) -> None:
    if edge_activity.shape[0] == 0:
        return

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pos = nx.spring_layout(graph, seed=42)

    idx = np.linspace(0, edge_activity.shape[0] - 1, num=min(snapshots, edge_activity.shape[0]), dtype=int)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8))
    axes = axes.ravel()

    for ax_i, w in enumerate(idx):
        ax = axes[ax_i]
        edge_weights = edge_activity[w]
        max_w = max(float(edge_weights.max()), 1.0)
        widths = []
        colors = []
        for e_idx, edge in enumerate(edges):
            val = float(edge_weights[e_idx])
            widths.append(0.8 + 4.0 * (val / max_w))
            colors.append(val)

        node_colors = ["#e63946" if labels[w, i] >= 0.5 else "#7fb3d5" for i in range(len(nodes))]
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=1200, node_color=node_colors, edgecolors="#1f1f1f")
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, font_weight="bold")
        if edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=edges,
                width=widths,
                edge_color=colors,
                edge_cmap=plt.cm.Blues,
                arrows=True,
                arrowsize=18,
                connectionstyle="arc3,rad=0.08",
            )
        ax.set_title(f"Temporal Graph Snapshot t={w}", fontsize=10)
        ax.axis("off")

    for j in range(len(idx), rows * cols):
        axes[j].axis("off")

    fig.suptitle("Graph Network Temporal Instances (nodes/edges over time)", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#d62728")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", color="#2ca02c")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#9467bd")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_training_outputs(
    output_dir: Path,
    model: TGCN,
    config: Dict,
    scaler: Dict[str, np.ndarray],
    nodes: List[str],
    edges: List[Tuple[str, str]],
    adj_norm: np.ndarray,
    thresholds: Dict[str, float],
    history: Dict[str, List[float]],
    metrics: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "tgcn_inference_bundle.pt"
    state_dict_path = output_dir / "tgcn_state_dict.pt"
    report_path = output_dir / "training_report.json"

    torch.save(model.state_dict(), state_dict_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "nodes": nodes,
            "edges": edges,
            "adj_norm": adj_norm.tolist(),
            "feature_mean": scaler["mean"].tolist(),
            "feature_std": scaler["std"].tolist(),
            "thresholds": thresholds,
        },
        bundle_path,
    )

    report = {
        "config": config,
        "nodes": nodes,
        "edges": edges,
        "thresholds": thresholds,
        "history_last": {k: float(v[-1]) for k, v in history.items() if v},
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

