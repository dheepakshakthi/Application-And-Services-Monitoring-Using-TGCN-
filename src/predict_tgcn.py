import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from tgcn_pipeline import TGCN, build_graph_timeseries, build_sequences, load_telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TGCN inference on telemetry CSV.")
    parser.add_argument("--data", type=Path, default=Path("data/telemetry_data.csv"))
    parser.add_argument("--model-bundle", type=Path, default=Path("artifacts/tgcn/tgcn_inference_bundle.pt"))
    parser.add_argument("--top-k", type=int, default=4, help="Show top-k risky nodes.")
    return parser.parse_args()


def _load_bundle(bundle_path: Path, device: torch.device) -> Dict:
    bundle = torch.load(bundle_path, map_location=device)
    required = {"model_state_dict", "config", "nodes", "edges", "adj_norm", "feature_mean", "feature_std"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Inference bundle is missing fields: {sorted(missing)}")
    return bundle


def _prepare_latest_sequence(
    csv_path: Path, nodes: List[str], edges: List[Tuple[str, str]], window_seconds: float, seq_len: int
) -> np.ndarray:
    df = load_telemetry(csv_path)
    graph_data = build_graph_timeseries(df, window_seconds=window_seconds)

    if graph_data.nodes != nodes:
        raise ValueError(
            f"Node mismatch between model and telemetry. Model nodes={nodes}, telemetry nodes={graph_data.nodes}"
        )
    if graph_data.edges != edges:
        raise ValueError(
            f"Edge mismatch between model and telemetry. Model edges={edges}, telemetry edges={graph_data.edges}"
        )

    x, _ = build_sequences(graph_data.features, graph_data.labels, seq_len=seq_len, horizon=1)
    return x[-1:]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = _load_bundle(args.model_bundle, device=device)

    config = bundle["config"]
    nodes = bundle["nodes"]
    edges = [tuple(e) for e in bundle["edges"]]
    adj_norm = np.array(bundle["adj_norm"], dtype=np.float32)
    feature_mean = np.array(bundle["feature_mean"], dtype=np.float32).reshape(1, 1, 1, -1)
    feature_std = np.array(bundle["feature_std"], dtype=np.float32).reshape(1, 1, 1, -1)

    x_latest = _prepare_latest_sequence(
        csv_path=args.data,
        nodes=nodes,
        edges=edges,
        window_seconds=float(config["window_seconds"]),
        seq_len=int(config["seq_len"]),
    )
    x_latest = (x_latest - feature_mean) / np.where(feature_std < 1e-6, 1.0, feature_std)

    model = TGCN(
        adj=torch.from_numpy(adj_norm).to(device),
        in_features=int(config["in_features"]),
        gcn_hidden=int(config["gcn_hidden"]),
        gru_hidden=int(config["gru_hidden"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x_latest.astype(np.float32)).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    order = np.argsort(-probs)
    top_k = min(args.top_k, len(order))
    print("Predicted node risk scores (next temporal window):")
    for i in range(top_k):
        idx = int(order[i])
        print(f"  {nodes[idx]}: {probs[idx]:.4f}")


if __name__ == "__main__":
    main()

