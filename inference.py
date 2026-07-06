"""
Inference Script with Automated Workspace Isolation.
Generates an isolated timestamped subdirectory per image processing execution thread.
"""

import argparse
from datetime import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import v2

from src.pix2pix import Pix2Pix
from utils.config import Config
from utils.utils import setup_logging


def run_inference(kw_args=defaultdict(lambda: None)):
    config = Config("config.yaml")
    
    # 1. Resolve Paths from Checkpoint Target
    gen_checkpoint = Path(kw_args["gen_checkpoint"] or config["inference"]["gen_checkpoint"])
    if not gen_checkpoint.exists():
        raise FileNotFoundError(f"Generator checkpoint file not found: {gen_checkpoint}")

    run_dir = gen_checkpoint.parent.parent if gen_checkpoint.parent.name == "checkpoints" else gen_checkpoint.parent
    
    # 2. Build Timestamped Session Folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_dir = run_dir / "inference" / f"inf_{timestamp}"
    inf_dir.mkdir(parents=True, exist_ok=True)

    # Extract target file base name to preserve file extension profiles
    orig_output_name = Path(kw_args["output_path"] or config["inference"]["output_path"]).name
    output_path = inf_dir / orig_output_name

    # 3. Setup Logging Pipeline
    setup_logging(config, run_log_dir=inf_dir, log_name="inference.log")
    logging.info(f"Resolved Root Run Folder: {run_dir}")
    logging.info(f"Created Session Directory: {inf_dir}")
    
    device_str = kw_args["device"] or config["inference"]["device"]
    device = torch.device(device_str)
    img_path = Path(kw_args["image_path"] or config["inference"]["image_path"])

    # 4. Model Construction
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
    logging.info(f"Weights loaded successfully from: {gen_checkpoint}")

    if not img_path.exists() or not img_path.is_file():
        raise FileNotFoundError(f"Target input image file not found at: {img_path}")

    img = Image.open(img_path).convert("RGB")

    transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])
    img_tensor = transforms(img).unsqueeze(0).to(device)

    # 5. Execute Prediction Forward Pass
    logging.info(f"Processing image {img_path.name} through the generator network...")
    with torch.no_grad():
        pred_uint8 = raw_model.generate(img_tensor, is_scaled=True, to_uint8=True)
        pred_array = pred_uint8.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    output_image = Image.fromarray(pred_array)
    output_image.save(output_path)
    logging.info(f"Inference complete! Output image saved to -> {output_path}")

    # 6. Save Independent Runtime Metadata
    inference_meta = {
        "inference_timestamp": timestamp,
        "status": "SUCCESS",
        "input_image_path": str(img_path),
        "output_image_path": str(output_path),
        "model_checkpoint_used": str(gen_checkpoint),
        "device_used": device_str
    }
    
    meta_json_path = inf_dir / "inference_results.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(inference_meta, f, indent=4)
    logging.info(f"Inference execution summary metadata saved to -> {meta_json_path}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Run inference using a trained Pix2Pix generator model.")
    cli_parser.add_argument("--image_path", type=str, default=None)
    cli_parser.add_argument("--output_path", type=str, default=None)
    cli_parser.add_argument("--gen_checkpoint", type=str, default=None)
    cli_parser.add_argument("--device", type=str, default=None)
    cli_args = cli_parser.parse_args()
    
    run_inference(kw_args=defaultdict(lambda: None, vars(cli_args)))