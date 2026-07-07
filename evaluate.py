"""
Evaluation Script for Pix2Pix Model Performance.
Generates an isolated, timestamped evaluation folder for historical run tracking.
"""

import argparse
from datetime import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm

from src.dataset import Sentinel
from src.metric import calculate_fid, extract_features, calculate_psnr, calculate_ssim, LPIPMetric
from src.pix2pix import Pix2Pix
from utils.config import Config
from utils.utils import setup_logging


def evaluate(kw_args=defaultdict(lambda: None)):
    # Load global root default config
    config = Config("config.yaml")
    
    # 1. Resolve Parent Run & Checkpoint Directories
    gen_checkpoint = Path(kw_args['gen_checkpoint'] or config["training"]["gen_checkpoint"])
    if not gen_checkpoint.exists():
        raise FileNotFoundError(f"Generator checkpoint file not found: {gen_checkpoint}")

    checkpoint_dir = gen_checkpoint.parent
    run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
    
    # 2. Automatically Inherit the Exact Training Parameters from Checkpoint Directory
    checkpoint_config_path = checkpoint_dir / "config.yaml"
    if checkpoint_config_path.exists():
        run_config = Config(str(checkpoint_config_path))
        print(f"Successfully loaded run configuration automatically from: {checkpoint_config_path}")
    else:
        # Fallback to standard run directory if checkpoints structure differs
        run_config_path = run_dir / "config.yaml"
        if run_config_path.exists():
            run_config = Config(str(run_config_path))
            print(f"Successfully loaded run configuration automatically from: {run_config_path}")
        else:
            run_config = config
            print("Warning: Local run config.yaml not found. Falling back to root configuration.")

    # 3. Extract Ablation Configurations Safely from the Saved Configuration
    is_cgan_bool = run_config["model"].get("is_CGAN", True)
    use_gan_bool = run_config["model"].get("use_gan", True)

    # 4. Create Isolated Timestamped Evaluation Folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = run_dir / "evaluation" / f"eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 5. Initialize Logger inside the unique run folder
    setup_logging(config, run_log_dir=eval_dir, log_name="evaluation.log")
    logging.info(f"Resolved Root Run Directory: {run_dir}")
    logging.info(f"Created Evaluation Session Directory: {eval_dir}")
    logging.info(f"Autodetected Model State -> is_CGAN: {is_cgan_bool} | use_gan (Loss Ablation): {use_gan_bool}")

    device_str = kw_args['device'] or config["training"]["device"]
    device = torch.device(device_str)
    logging.info(f"Running evaluation on device: {device}")

    # 6. Initialize Evaluation Models
    logging.info("Loading evaluation backbones (InceptionV3 & LPIPS)...")
    inception = models.inception_v3(weights="DEFAULT", transform_input=False).eval().to(device)
    lpips_evaluator = LPIPMetric(net='alex', device=device)

    inception_transform = v2.Compose([
        v2.Resize(342),
        v2.CenterCrop(299),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Determine which data splits to execute metrics across
    split_arg = kw_args['split'] or "both"
    splits_to_evaluate = ["val", "test"] if split_arg == "both" else [split_arg]

    for split_type in splits_to_evaluate:
        logging.info(f"\n--- Starting Evaluation for Split: [{split_type.upper()}] ---")

        # 7. Setup Dataset Split Dynamically
        dataset = Sentinel(
            root_dir=kw_args['root_dir'] or config["dataset"]["root_dir"],
            split_type=split_type,
            input_transform=base_transform,
            target_transform=base_transform,
            split_mode=config["dataset"]["split_mode"],
            split_ratio=config["dataset"]["split_ratio"],
            seed=config["dataset"]["seed"],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
        )

        # 8. Initialize Pix2Pix Model with Automatically Extracted Flags
        model = Pix2Pix(
            c_in=config["model"]["c_in"],
            c_out=config["model"]["c_out"],
            netD=config["model"].get("netD"),
            is_CGAN=is_cgan_bool,
            use_upsampling=config["model"]["use_upsampling"],
            mode=config["model"]["mode"],
        ).to(device).eval()

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_model(gen_path=str(gen_checkpoint))
        logging.info(f"Weights loaded successfully for {split_type} from: {gen_checkpoint}")

        # 9. Evaluation Loop
        target_features = []
        fake_features = []
        running_psnr, running_ssim, running_lpips = 0.0, 0.0, 0.0
        batch_count = 0

        logging.info(f"Starting evaluation metrics collection on {split_type} split...")
        with torch.no_grad():
            for real_images, target_images in tqdm(dataloader, desc=f"Evaluating {split_type}"):
                real_images, target_images = real_images.to(device), target_images.to(device)

                fake_images_uint8 = raw_model.generate(real_images, is_scaled=True, to_uint8=True).to(device)
                target_images_uint8 = (target_images * 255).to(dtype=torch.uint8)
                
                fake_images_float = fake_images_uint8.to(dtype=torch.float32) / 255.0
                target_images_float = target_images

                # Metrics Computation
                running_psnr += calculate_psnr(fake_images_float, target_images_float)
                running_ssim += calculate_ssim(fake_images_float, target_images_float)
                running_lpips += lpips_evaluator(fake_images_float, target_images_float)
                batch_count += 1

                # Inception Feature Extraction
                target_feats = extract_features(inception_transform(target_images_uint8), inception)
                target_features.append(target_feats.cpu().numpy())

                fake_feats = extract_features(inception_transform(fake_images_uint8), inception)
                fake_features.append(fake_feats.cpu().numpy())

        # 10. Compute Final Statistics
        logging.info("Processing aggregated metrics...")
        real_features = np.concatenate(target_features, axis=0)
        generated_features = np.concatenate(fake_features, axis=0)
        fid_score = calculate_fid(real_features, generated_features)

        avg_psnr = running_psnr / batch_count
        avg_ssim = running_ssim / batch_count
        avg_lpips = running_lpips / batch_count

        # 11. Print & Log Summary Block
        logging.info("\n" + "="*50 + "\n"
                     f"          EVALUATION COMPLETE [{split_type.upper()}]          \n"
                     + "="*50 + "\n"
                     f" Inherited Model Setup: is_CGAN={is_cgan_bool}, use_gan={use_gan_bool}\n"
                     f" FID Score            : {fid_score:.4f}\n"
                     f" LPIPS Dist           : {avg_lpips:.4f}\n"
                     f" SSIM Index           : {avg_ssim:.4f}\n"
                     f" PSNR Ratio           : {avg_psnr:.2f} dB\n"
                     + "="*50)

        # 12. Export Metrics JSON containing specific tracked ablation tags
        metrics_dict = {
            "evaluation_timestamp": timestamp,
            "split_evaluated": split_type,
            "ablation_is_CGAN": is_cgan_bool,
            "ablation_use_gan": use_gan_bool,
            "lambda_L1": config["model"].get("lambda_L1", 100.0),
            "fid": float(fid_score),
            "lpips": float(avg_lpips),
            "ssim": float(avg_ssim),
            "psnr": float(avg_psnr),
            "batch_size": config["training"]["batch_size"],
            "checkpoint_evaluated": str(gen_checkpoint)
        }
        
        results_json_path = eval_dir / f"{split_type}_evaluation_metrics.json"
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)
        logging.info(f"Isolated evaluation metrics stored at -> {results_json_path}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Run automated Pix2Pix evaluation suite.")
    cli_parser.add_argument("--root_dir", type=str, default=None, help="Path to dataset root directory folder.")
    cli_parser.add_argument("--device", type=str, default=None, help="Hardware compute device target ('cpu'/'cuda').")
    cli_parser.add_argument("--gen_checkpoint", type=str, default=None, help="Explicit path to generator weight .pth file.")
    cli_parser.add_argument("--split", type=str, default="both", choices=["val", "test", "both"], 
                            help="Choose 'val', 'test', or 'both' dataset splits for metrics evaluation loops.")
    cli_args = cli_parser.parse_args()
    
    evaluate(kw_args=defaultdict(lambda: None, vars(cli_args)))