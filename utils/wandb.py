import yaml
import os
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


def load_config(config_path, model_config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "models" in config:
        return config
    else:
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
        config["models"] = {}
        config["models"].update(model_config["models"])
    return config


def init_wandb(args, config):
    sweep_overrides = {}
    if args.wandb:
        if not WANDB_AVAILABLE:
            raise RuntimeError("Install wandb or omit --wandb.")
        else:
            logging_dir = config.get("logging_dir")
            wandb_kwargs = {"project": args.wandb_project, "config": config}
            if args.wandb_mode:
                wandb_kwargs["mode"] = args.wandb_mode
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if args.wandb_group_name:
                wandb_kwargs["group"] = args.wandb_group_name
            if logging_dir:
                Path(logging_dir).mkdir(parents=True, exist_ok=True)
                wandb_kwargs["dir"] = logging_dir
            wandb.init(**wandb_kwargs)
            sweep_overrides = dict(wandb.config)
    return sweep_overrides


def apply_nested(target_dict, dotted_key, value):
    if not isinstance(target_dict, dict):
        return
    if not isinstance(dotted_key, str):
        return

    keys = dotted_key.split(".")
    current = target_dict
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def apply_sweep_overrides(args, config, sweep_overrides):
    if not sweep_overrides:
        return

    reserved_override_keys = {"wandb_version"}
    reserved_args = {"wandb", "wandb_mode"}

    # Apply full sections before dotted keys so defaults cannot undo assignments.
    for key, value in sorted(sweep_overrides.items(),
                             key=lambda item: str(item[0]).count(".")):
        if not isinstance(key, str):
            continue
        if key in reserved_override_keys or key.startswith("_"):
            continue
        if key in reserved_args:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
            continue
        apply_nested(config, key, value)

def build_wandb_run_name(args):
    if args.wandb_run_name:
        return args.wandb_run_name
    if wandb is not None and wandb.run is not None:
        run_id = getattr(wandb.run, "id", None)
        return f"{args.model_type}-{run_id}"
    return f"{args.model_type}"

def sanitize_run_name(run_name):
    if not run_name:
        return "run"
    safe_chars = []
    for ch in str(run_name):
        if ch.isascii() and (ch.isalnum() or ch in ("-", "_", ".")):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe_name = "".join(safe_chars).strip("_")
    return safe_name or "run"


def resolve_checkpoint_dir(config, run_name):
    checkpoint_dir = config.get("checkpoint_dir")
    if not checkpoint_dir:
        return
    checkpointing_config = config.get("checkpointing", {})
    run_scoped = checkpointing_config.get("run_scoped", True)
    if not run_scoped:
        return
    run_id = None
    if wandb is not None and wandb.run is not None:
        run_id = getattr(wandb.run, "id", None)
    if not run_id:
        return
    safe_name = sanitize_run_name(run_name)
    resolved_dir = os.path.join(checkpoint_dir, f"{safe_name}-{run_id}")
    config["checkpoint_dir"] = resolved_dir

    print(f"Using run-scoped checkpoint_dir: {resolved_dir}")
    if wandb is not None and wandb.run is not None:
        try:
            wandb.run.summary["checkpoint_dir"] = resolved_dir
        except Exception:
            pass
