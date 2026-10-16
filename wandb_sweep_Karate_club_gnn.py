"""Train KarateClub GCN, GAT, GraphSAGE or DNN with YAML and W&B overrides.

Sweep Register: python -m wandb sweep --entity kjsong --project karate-club-gcn-sweep configs/sweeps/gcn_config.yaml
Sweep Agent: python -m wandb agent --count 1 ENTITY/PROJECT/SWEEP_ID

Train locally: python wandb_sweep_Karate_club_gnn.py
Sweep assignments are read from wandb.config after wandb.init().
"""

import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from utils.model import GCN, GAT, GraphSAGE, DNN
from utils.wandb import load_config, init_wandb, apply_sweep_overrides, build_wandb_run_name, resolve_checkpoint_dir

def seed_everything(seed):
    """ Set random seeds for reproducibility.
    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    parser = argparse.ArgumentParser(description="GNN Training with PyG")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--gnn_model_config",
        type=str,
        default="configs/gnn_model_config.yaml",
        help="Path to gnn model config YAML (gnn.yaml)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="GCN",
        choices=["GCN", "GAT", "GraphSAGE", "DNN"],
        help="Model type to train (default: GCNConv)",
    )
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="karate-club-gcn-sweep",
        help="Weights & Biases project name",
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team name")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name override (default: auto)",
    )
    parser.add_argument(
        "--wandb_group_name",
        type=str,
        default=None,
        help="Weights & Biases group name (default: none)",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode override",
    )
    return parser


def load_data():
    dataset = KarateClub()
    data = dataset[0]  # Get the first graph object.

    # 학습 노드를 제외하고 각 클래스에서 검증 노드를 하나씩 선택
    val_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)

    for class_id in range(dataset.num_classes):
        candidates = torch.where(
            (data.y == class_id) & (~data.train_mask)
        )[0]

        if candidates.numel() == 0:
            raise ValueError(
                f'클래스 {class_id}에 검증용으로 선택할 노드가 없습니다.'
            )

        val_mask[candidates[0]] = True

    # 학습 및 검증에 사용하지 않은 나머지 노드를 테스트에 사용
    test_mask = ~(data.train_mask | val_mask)

    data.val_mask = val_mask
    data.test_mask = test_mask

    return dataset, data


@torch.no_grad()
def evaluate(model, data, masks):
    model.eval()
    logits = model(data.x, data.edge_index)
    metrics = {}
    for name, mask in masks.items():
        metrics[f"{name}_loss"] = F.cross_entropy(logits[mask], data.y[mask]).item()

        metrics[f"{name}_accuracy"] = (
            F.softmax(logits[mask], dim=1).argmax(dim=1) == data.y[mask]
        ).float().mean().item()
    return metrics


def train(args, config):
    dataset, data = load_data()
    model_config = config["models"][args.model_type]

    if args.model_type == "GCN":
        model_class = GCN
        kwargs = {
        "in_channels": dataset.num_features,
        "hidden_channels": model_config["hidden_channels"],
        "num_classes": dataset.num_classes,
        "num_layers": model_config["num_layers"],
        "dropout": model_config["dropout"],
        }
    elif args.model_type == "GAT":
        model_class = GAT
        kwargs = {
        "in_channels": dataset.num_features,
        "hidden_channels": model_config["hidden_channels"],
        "heads": model_config["heads"],
        "num_classes": dataset.num_classes,
        "num_layers": model_config["num_layers"],
        "dropout": model_config["dropout"],
        }
    elif args.model_type == "GraphSAGE":
        model_class = GraphSAGE
        kwargs = {
        "in_channels": dataset.num_features,
        "hidden_channels": model_config["hidden_channels"],
        "aggr": model_config["aggr"],
        "num_classes": dataset.num_classes,
        "num_layers": model_config["num_layers"],
        "dropout": model_config["dropout"],
        }
    elif args.model_type == "DNN":
        model_class = DNN
        kwargs = {
        "in_channels": dataset.num_features,
        "hidden_channels": model_config["hidden_channels"],
        "num_classes": dataset.num_classes,
        "num_layers": model_config["num_layers"],
        "dropout": model_config["dropout"],
        }

    model = model_class(**kwargs)

    optimizer_config = config["optimizer"]
    if "Adam" in optimizer_config:
        optimizer_class, optimizer_kwargs = torch.optim.Adam, optimizer_config["Adam"]
    elif "AdamW" in optimizer_config:
        optimizer_class, optimizer_kwargs = torch.optim.AdamW, optimizer_config["AdamW"]
    else:
        raise ValueError("Configure an Adam or AdamW optimizer")
    # Accept scientific notation that PyYAML may load as strings.
    optimizer_kwargs = dict(optimizer_kwargs)
    for key in ("lr", "weight_decay", "eps"):
        if key in optimizer_kwargs:
            optimizer_kwargs[key] = float(optimizer_kwargs[key])
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    training = config["training"]
    max_epochs, patience = int(training["max_epochs"]), int(training["patience"])
    if max_epochs < 1 or patience < 1:
        raise ValueError("training.max_epochs and training.patience must be positive")
    clip_value = float(training.get("gradient_clip_val", 0.0))
    clip_algorithm = training.get("gradient_clip_algorithm", "norm")
    if clip_value < 0 or clip_algorithm not in ("norm", "value"):
        raise ValueError("Invalid gradient clipping configuration")
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    active_run = wandb.run if args.wandb and WANDB_AVAILABLE else None

    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        if clip_value > 0:
            if clip_algorithm == "norm":
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            else:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        metrics = evaluate(
            model, data, {"val": data.val_mask}
        )

        if not torch.isfinite(torch.tensor(metrics["val_loss"])):
            raise RuntimeError("Training loss is non-finite")
        if metrics["val_loss"] < best_loss:
            best_loss, best_epoch, stale = metrics["val_loss"], epoch, 0
            best_state = {key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()}
            checkpoint = {
                "epoch": best_epoch,
                "config": config,
                "model_class": model_class.__name__,
                "model_kwargs": kwargs,
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_loss,}
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            if active_run is not None:
                active_run.summary["best_model_path"] = str(checkpoint_path.resolve())
        else:
            stale += 1
        if active_run is not None:
            active_run.log({"epoch": epoch, "loss": loss.item(),
                           **metrics, "best_val_loss": best_loss}, step=epoch)
        print(f"epoch{epoch}, loss={loss.item():.4f}")

        if stale >= patience:
            break
    if active_run is not None:
        active_run.summary.update({"best_val_loss": best_loss, "best_epoch": best_epoch,
                                   "epochs_trained": epoch})
    return checkpoint_path


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    config = load_config(
        config_path=args.config,
        model_config_path=args.gnn_model_config,
    )

    sweep_overrides = init_wandb(args, config)

    if sweep_overrides:
        apply_sweep_overrides(args, config, sweep_overrides)

    seed = args.seed
    if seed is None:
        seed = config.get("seed")
    if seed is not None:
        seed_everything(seed)

    run_name = None
    if args.wandb:
        run_name = build_wandb_run_name(args)
        if WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
            try:
                wandb.run.name = run_name
            except Exception:
                pass
        resolve_checkpoint_dir(config, run_name)

    try:
        train(args, config)
    except Exception:
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish(exit_code=1)
        raise
    else:
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()



if __name__ == "__main__":
    main()
