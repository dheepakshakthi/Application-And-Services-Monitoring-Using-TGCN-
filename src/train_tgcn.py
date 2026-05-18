import argparse
from pathlib import Path

import torch

from tgcn_pipeline import (
    TGCN,
    build_graph_timeseries,
    build_normalized_adjacency,
    build_sequences,
    evaluate_model,
    load_telemetry,
    normalize_features,
    plot_graph_temporal_instances,
    plot_training_curves,
    save_training_outputs,
    set_seed,
    split_dataset,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export TGCN on telemetry logs.")
    parser.add_argument("--data", type=Path, default=Path("data/telemetry_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tgcn"))
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1, help="Predict labels at t + horizon windows.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gcn-hidden", type=int, default=32)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = load_telemetry(args.data)
    graph_data = build_graph_timeseries(df, window_seconds=args.window_seconds)
    adj_norm = build_normalized_adjacency(graph_data.nodes, graph_data.edges)

    x, y = build_sequences(
        graph_data.features,
        graph_data.labels,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
    split = split_dataset(x, y)
    split_scaled, scaler = normalize_features(split)

    in_features = split_scaled["x_train"].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TGCN(
        adj=torch.from_numpy(adj_norm).to(device),
        in_features=in_features,
        gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden,
        dropout=args.dropout,
    ).to(device)

    history = train_model(
        model=model,
        data=split_scaled,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )
    metrics = evaluate_model(model, split_scaled["x_test"], split_scaled["y_test"], device=device)

    config = {
        "window_seconds": args.window_seconds,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "gcn_hidden": args.gcn_hidden,
        "gru_hidden": args.gru_hidden,
        "dropout": args.dropout,
        "seed": args.seed,
        "in_features": in_features,
        "num_nodes": len(graph_data.nodes),
    }

    save_training_outputs(
        output_dir=args.output,
        model=model,
        config=config,
        scaler=scaler,
        nodes=graph_data.nodes,
        edges=graph_data.edges,
        adj_norm=adj_norm,
        thresholds=graph_data.thresholds,
        history=history,
        metrics=metrics,
    )

    plots_dir = args.output / "plots"
    plot_graph_temporal_instances(
        nodes=graph_data.nodes,
        edges=graph_data.edges,
        edge_activity=graph_data.edge_activity,
        labels=graph_data.labels,
        output_path=plots_dir / "graph_temporal_instances.png",
        snapshots=6,
    )
    plot_training_curves(history, plots_dir / "training_loss_accuracy.png")

    print("Training complete.")
    print(f"Output directory: {args.output.resolve()}")
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

