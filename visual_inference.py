"""
Inference Visualization Script with Subplot Isolation Engine.
Extracts test frames dynamically up to batch size, outputs structural visual comparisons,
and isolates executions into timestamped run directories.
"""

import argparse
from datetime import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.dataset import Sentinel
from src.pix2pix import Pix2Pix
from utils.config import Config
from utils.utils import setup_logging


def run_visual_inference(kw_args=defaultdict(lambda: None)):
    config = Config("config.yaml")
    
    # 1. Resolve Target Workspace Tree from Checkpoint Flag
    gen_checkpoint = Path(kw_args["gen_checkpoint"] or config["inference"]["gen_checkpoint"])
    if not gen_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {gen_checkpoint}")

    run_dir = gen_checkpoint.parent.parent if gen_checkpoint.parent.name == "checkpoints" else gen_checkpoint.parent
    
    # 2. Setup Timestamped Visual Grid Subfolder Workspace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_dir = run_dir / "inference" / f"visual_grid_{timestamp}"
    inf_dir.mkdir(parents=True, exist_ok=True)

    orig_plot_name = Path(kw_args["output_plot"] or "./runs/inference_grid.png").name
    output_plot_path = inf_dir / orig_plot_name

    # 3. Setup Logging Configuration Target
    setup_logging(config, run_log_dir=inf_dir, log_name="visualize_inference.log")
    logging.info(f"Resolved Root Run Folder Location: {run_dir}")
    logging.info(f"Created Visual Workspace Directory: {inf_dir}")
    
    device_str = kw_args["device"] or config["inference"].get("device", "cuda")
    device = torch.device(device_str)

    base_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    normalize_transform = v2.Normalize(mean=[0.5], std=[0.5])

    # 4. Mount Test Dataset and Pull Evaluation Batch Samples
    dataset = Sentinel(
        root_dir=config["dataset"]["root_dir"],
        split_type="test",
        input_transform=base_transform,
        target_transform=base_transform,
        split_mode=config["dataset"]["split_mode"],
        split_ratio=config["dataset"]["split_ratio"],
        seed=config["dataset"]["seed"],
    )

    # Request up to 5, but we check what we actually get to avoid IndexErrors
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    real_batch, target_batch = next(iter(dataloader))
    
    num_samples = real_batch.size(0)
    logging.info(f"Successfully loaded {num_samples} random test samples from split.")

    # 5. Build Model Framework
    model = Pix2Pix(
        c_in=config["model"]["c_in"],
        c_out=config["model"]["c_out"],
        netD=config["model"].get("netD"),
        is_CGAN=config["model"].get("is_CGAN", True),
        use_upsampling=config["model"]["use_upsampling"],
        mode=config["model"]["mode"],
    ).to(device).eval()

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_model(gen_path=str(gen_checkpoint))
    logging.info(f"Loaded network weight mapping matrix from path: {gen_checkpoint}")

    # 6. Run Execution Prediction Pipeline
    logging.info(f"Processing {num_samples} images through generator execution layers...")
    predictions = []
    
    with torch.no_grad():
        for i in range(num_samples):
            img_input = real_batch[i].unsqueeze(0).to(device)
            normalized_input = normalize_transform(img_input)
            pred_uint8 = raw_model.generate(normalized_input, is_scaled=True, to_uint8=True)
            pred_float = pred_uint8.squeeze(0).to(dtype=torch.float32) / 255.0
            predictions.append(pred_float.numpy().transpose(1, 2, 0))

    # 7. Generate Comparative Multi-Row Plot Layout
    logging.info(f"Building 3x{num_samples} subplot layout canvas...")
    
    # squeeze=False ensures axes array remains 2D even if num_samples == 1
    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9.5), squeeze=False)
    row_titles = ["Input Samples", "Predicted Outputs", "Actual S2 (Ground Truth)"]
    
    for col_idx in range(num_samples):
        input_np = real_batch[col_idx].numpy().transpose(1, 2, 0)
        target_np = target_batch[col_idx].numpy().transpose(1, 2, 0)
        pred_np = predictions[col_idx]
        
        # Row 1: Structural Input Frames
        axes[0, col_idx].imshow(input_np)
        axes[0, col_idx].axis("off")
        
        # Row 2: Generated Output Variations
        axes[1, col_idx].imshow(pred_np)
        axes[1, col_idx].axis("off")
        
        # Row 3: Target Ground Truth Layouts
        axes[2, col_idx].imshow(target_np)
        axes[2, col_idx].axis("off")
        
    for row_idx, title in enumerate(row_titles):
        axes[row_idx, 0].text(
            -15, 128, title, rotation=90, verticalalignment="center",
            horizontalalignment="right", fontsize=14, weight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    logging.info(f"Visual matrix export mapping complete! Plot saved to -> {output_plot_path}")

    # 8. Export Visualization Runtime Metadata
    visual_meta = {
        "visualization_timestamp": timestamp,
        "status": "SUCCESS",
        "saved_plot_path": str(output_plot_path),
        "model_checkpoint_used": str(gen_checkpoint),
        "samples_count": num_samples
    }
    meta_json_path = inf_dir / "visualize_results.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(visual_meta, f, indent=4)
    logging.info(f"Visual grid metadata summary file logged to -> {meta_json_path}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Generate visual inference grid summary plot.")
    cli_parser.add_argument("--gen_checkpoint", type=str, default=None)
    cli_parser.add_argument("--output_plot", type=str, default=None)
    cli_parser.add_argument("--device", type=str, default=None)
    cli_args = cli_parser.parse_args()
    
    run_visual_inference(kw_args=defaultdict(lambda: None, vars(cli_args)))