"""
Inference Script with Automated Workspace Isolation.
Generates an isolated timestamped subdirectory per image processing execution thread over an entire directory.
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
    inf_dir = run_dir / "inference" / f"inf_dir_{timestamp}"
    inf_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input and output directory paths
    input_dir_str = kw_args["input_dir"] or config["inference"].get("input_dir")
    output_dir_str = kw_args["output_dir"] or config["inference"].get("output_dir")

    if not input_dir_str:
        raise ValueError("Input directory must be specified via --input_dir or in config.yaml under inference -> input_dir")

    input_dir = Path(input_dir_str)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Target input directory not found or is not a directory: {input_dir}")

    # If an output directory is explicitly provided, we mirror the outputs there within our isolated folder
    if output_dir_str:
        output_images_dir = Path(output_dir_str) / f"outputs_{timestamp}"
    else:
        output_images_dir = inf_dir / "predicted_images"
        
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # 3. Setup Logging Pipeline
    setup_logging(config, run_log_dir=inf_dir, log_name="inference.log")
    logging.info(f"Resolved Root Run Folder: {run_dir}")
    logging.info(f"Created Session Directory: {inf_dir}")
    logging.info(f"Scanning Input Directory: {input_dir}")
    logging.info(f"Saving Generated Images To: {output_images_dir}")
    
    device_str = kw_args["device"] or config["inference"]["device"]
    device = torch.device(device_str)

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

    # Supported image formats
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    img_paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]

    if not img_paths:
        logging.warning(f"No valid image files found in {input_dir}")
        return

    transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])

    processed_records = []

    # 5. Execute Prediction Forward Pass over all discovered images
    logging.info(f"Found {len(img_paths)} images to process. Starting batch execution loop...")
    
    for idx, img_path in enumerate(img_paths, start=1):
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transforms(img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_uint8 = raw_model.generate(img_tensor, is_scaled=True, to_uint8=True)
                pred_array = pred_uint8.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            # Preserve original filename for the output mapping
            output_path = output_images_dir / img_path.name
            output_image = Image.fromarray(pred_array)
            output_image.save(output_path)
            
            logging.info(f"[{idx}/{len(img_paths)}] Successfully processed: {img_path.name} -> {output_path.name}")
            
            processed_records.append({
                "input_file": str(img_path),
                "output_file": str(output_path),
                "status": "SUCCESS"
            })
        except Exception as e:
            logging.error(f"Failed to process image {img_path.name}. Error details: {str(e)}")
            processed_records.append({
                "input_file": str(img_path),
                "status": f"FAILED: {str(e)}"
            })

    # 6. Save Independent Runtime Metadata Summary
    inference_meta = {
        "inference_timestamp": timestamp,
        "input_directory": str(input_dir),
        "output_directory": str(output_images_dir),
        "total_images_found": len(img_paths),
        "successfully_processed": len([r for r in processed_records if "SUCCESS" in r["status"]]),
        "model_checkpoint_used": str(gen_checkpoint),
        "device_used": device_str,
        "detailed_results": processed_records
    }
    
    meta_json_path = inf_dir / "inference_results.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(inference_meta, f, indent=4)
    logging.info(f"Inference execution summary metadata saved to -> {meta_json_path}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Run batch directory inference using a trained Pix2Pix generator model.")
    cli_parser.add_argument("--input_dir", type=str, default=None, help="Directory containing source images for translation.")
    cli_parser.add_argument("--output_dir", type=str, default=None, help="Directory where translated images will be exported.")
    cli_parser.add_argument("--gen_checkpoint", type=str, default=None, help="Path to generator weights file.")
    cli_parser.add_argument("--device", type=str, default=None, help="Execution target device ('cuda' or 'cpu').")
    cli_args = cli_parser.parse_args()
    
    run_inference(kw_args=defaultdict(lambda: None, vars(cli_args)))