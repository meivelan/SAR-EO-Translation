"""
Evaluation Script
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2

from src.dataset import Sentinel
from src.metric import (
    LPIPMetric,
    calculate_fid,
    calculate_psnr,
    calculate_ssim,
    extract_features,
)
from src.pix2pix import Pix2Pix
from utils.config import Config


def main():
    # Load configuration
    config = Config("config.yaml")
    # Set device
    device = torch.device(config["training"]["device"])

    # Initialize InceptionV3 for FID
    inception = (
        models.inception_v3(weights="DEFAULT", transform_input=False).eval().to(device)
    )

    # Initialize LPIPS metric (loaded once before the loop)
    lpips_metric = LPIPMetric(net="alex", device=device)

    # Transforms from inception_v3 documentation for FID
    transform = v2.Compose(
        [
            v2.Resize(342),
            v2.CenterCrop(299),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = Sentinel(
        root_dir=config["dataset"]["root_dir"],
        split_type="test",
        split_mode=config["dataset"]["split_mode"],
        split_ratio=config["dataset"]["split_ratio"],
        split_file=config["dataset"]["split_file"],
        seed=config["dataset"]["seed"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["dataset"]["shuffle"],
        num_workers=config["training"]["num_workers"],
    )

    # Create model
    model = (
        Pix2Pix(
            c_in=config["model"]["c_in"],
            c_out=config["model"]["c_out"],
            is_train=False,
            use_upsampling=config["model"]["use_upsampling"],
            mode=config["model"]["mode"],
        )
        .to(device)
        .eval()
    )

    gen_checkpoint = Path(config["training"]["gen_checkpoint"])

    if not gen_checkpoint.exists():
        raise FileNotFoundError(
            f"Generator checkpoint file not found: {gen_checkpoint}\nPlease check config.yaml"
        )

    model.load_model(gen_path=gen_checkpoint)

    # Data structures for metric accumulation
    target_features = []
    fake_features = []
    
    total_samples = 0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    print("Starting evaluation loop...")
    for real_images, target_images in dataloader:
        real_images = real_images.to(device)
        target_images = target_images.to(device)
        
        batch_size = real_images.size(0)
        total_samples += batch_size

        # Pix2Pix.generate() gets a scaled tensor ([0,1]) returns a uint8 tensor ([0,255])
        fake_images_uint8 = model.generate(real_images, is_scaled=True, to_uint8=True)
        target_images_uint8 = (target_images * 255).to(dtype=torch.uint8)

        
        fake_images_01 = fake_images_uint8.to(dtype=torch.float32) / 255.0
        target_images_01 = target_images_uint8.to(dtype=torch.float32) / 255.0

        batch_psnr = calculate_psnr(target_images_01, fake_images_01)
        batch_ssim = calculate_ssim(target_images_01, fake_images_01)
        batch_lpips = lpips_metric(target_images_01, fake_images_01)

        # Weight by batch size to ensure accurate averages over the entire dataset
        total_psnr += batch_psnr * batch_size
        total_ssim += batch_ssim * batch_size
        total_lpips += batch_lpips * batch_size

        target_fid_input = transform(target_images_uint8)
        target_feats = extract_features(target_fid_input, inception)
        target_features.append(target_feats.cpu().numpy())

        fake_fid_input = transform(fake_images_uint8)
        fake_feats = extract_features(fake_fid_input, inception)
        fake_features.append(fake_feats.cpu().numpy())

    # Compute final averaged metrics
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    avg_lpips = total_lpips / total_samples

    # Compute FID score
    real_features = np.concatenate(target_features, axis=0)
    generated_features = np.concatenate(fake_features, axis=0)
    fid_score = calculate_fid(real_features, generated_features)

    # Print Evaluation Results
    print("\n" + "=" * 40)
    print("           EVALUATION RESULTS           ")
    print("=" * 40)
    print(f"Total Samples Evaluated : {total_samples}")
    print(f"PSNR                    : {avg_psnr:.4f} dB")
    print(f"SSIM                    : {avg_ssim:.4f}")
    print(f"LPIPS                   : {avg_lpips:.4f}")
    print(f"FID                     : {fid_score:.4f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
