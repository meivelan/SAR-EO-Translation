# SAR-to-EO Image Translation using Conditional GANs

This repository implements a paired, supervised cross-domain translation framework designed to map single-channel **Synthetic Aperture Radar (SAR)** imagery (Sentinel-1 VV band) to multi-channel **Electro-Optical (EO)** optical imagery (Sentinel-2 RGB bands). 

Because SAR imagery is immune to weather and cloud cover but visually complex due to speckle noise and geometric distortions, translating it into clear optical data is highly valuable for remote sensing pipelines. This project addresses the translation task using a **Pix2Pix (Conditional GAN)** framework optimized for high structural fidelity and realistic texture synthesis under tight compute constraints.

---

## Environment Setup

This repository is optimized to run seamlessly within a single-GPU cloud container environment (such as Kaggle Kernels with standard memory limits).

### Recommended Cloud Deployment (Kaggle)
1. Create a new notebook using the following dataset as your root environment workspace: [Sentinel-1 & 2 Image Pairs Segregated by Terrain](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain).
2. Under the notebook configuration panel, ensure **Persistence Memory** is enabled, and your accelerator is active (**GPU T4 x2** or **GPU P100**).
3. Execute the initial configuration cell to clone the environment and eliminate active disk I/O latency by loading data partitions directly into scratch memory:

```python
import os

# Clone the repository framework
!git clone [https://github.com/meivelan/SAR-EO-Translation.git](https://github.com/meivelan/SAR-EO-Translation.git)

# Move into the executable project directory context
os.chdir('SAR-EO-Translation')

# Transfer raw data volumes to local runtime instances to circumvent disk I/O overhead
from utils.utils import copy_dataset_to_kaggle_memory
copy_dataset_to_kaggle_memory()
```

### Local/Custom Server Deployment
For local workflows or headless Linux nodes, clone the repository and adjust the local path configurations within `config.yaml` to match your local scratch directories.

---

## Dataset Structure

The framework processes the terrain-agnostic and terrain-segregated configurations derived from the Sentinel paired dataset. The directory architecture mirrors the following setup:

```text
/SAR2Optical/data
└───v_2
    ├───agri
    │   ├───s1/      # Sentinel-1 SAR Backscatter Patches (VV)
    │   └───s2/      # Sentinel-2 optical Ground Truth Patches (RGB)
    ├───barrenland
    │   ├───s1/
    │   └───s2/
    ├───grassland
    │   ├───s1/
    │   └───s2/
    └───urban
        ├───s1/
        └───s2/
```

* **Footprint:** ~2 GB total containing 16,000 highly aligned image pairs.
* **Resolution:** 256 x 256 pixels, uniformly formatted in lossless PNG structures.
* **Ablation Setup:** Models are trained using a **terrain-agnostic** profile to enforce global terrain generalization across a random 70/15/15% split for training, validation, and testing.

---

## Training Pipelines

### 1. Initialize Training from Scratch
To run the default profile (PatchGAN architecture, lambda_L1 = 100$, Adam optimizer), execute:
```bash
python train.py
```

### 2. Resume an Interrupted Run / Fine-Tuning
The framework includes parameter-safe checkpointing. To resume a training thread (e.g., restarting at Epoch 31 after a systematic rollback or environment timeout), update `config.yaml` to reflect `resume: true` and specify `resume_epoch: 31`, then pass your saved weights:
```bash
python train.py --resume_epoch 31 \
  --gen_checkpoint /kaggle/working/runs/experiments/run_20260707_132855/checkpoints/generator_epoch_30.pth \
  --disc_checkpoint /kaggle/working/runs/experiments/run_20260707_132855/checkpoints/discriminator_epoch_30.pth
```

---

## Inference & Batch Execution

Process a directory of unlabelled target SAR patches using a trained generator model. The inference engine handles automated workspace tracking, creating a clean, timestamped folder for each run to protect historical outputs:

```bash
python inference.py --device cuda \
  --input_dir /path/to/input/sar_samples \
  --output_dir /path/to/output/predictions \
  --gen_checkpoint /kaggle/working/runs/experiments/run_20260707_132855/checkpoints/generator_epoch_50.pth
```

---

## Evaluation Routine

Run quantitative testing against your test split across all evaluation metrics (L1 pixel loss, Structural Similarity Index, Peak Signal-to-Noise Ratio, Learned Perceptual Image Patch Similarity, and Fréchet Inception Distance):

```bash
python evaluate.py \
  --gen_checkpoint /kaggle/working/runs/experiments/run_20260707_132855/checkpoints/generator_epoch_50.pth
```

---

## Ablation Studies & Structural Framework

This project includes a controlled architectural ablation study. We isolate how the discriminator's **receptive field scale** affects structural fidelity and textural realism, holding data partitions, augmentations, and optimization schedules completely fixed (50 Epochs each).

1. **U-Net + PatchGAN Baseline (`netD: "patch"`):** Utilizes a 3-layer discriminator with a localized receptive field of **70 x 70 pixels**. This allows the network to evaluate local high-frequency textures, sharpness, and boundaries without being constrained by global geometric layout.
2. **U-Net + PixelGAN Ablation (`netD: "pixel"`):** Redefines the discriminator's receptive field down to a **1 x 1 pixel** window. This forces the model to maximize radiometric color accuracy and channel combinations, relying entirely on the generator's global L_1 loss to enforce structural consistency.

---

## Experimental Results

### Quantitative Performance Metrics

The variants were evaluated across structural, radiometric, and perceptual indices on held-out data:

| Model Discriminator Variant | MAE (L1 down) | PSNR (up) | SSIM (up) | LPIPS (down) | FID (down) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PatchGAN (70 x 70 Patch)** | **0.0381** | **23.14 dB** | **0.8142** | **0.1421** | **34.82** |
| **PixelGAN (1 x 1 Pixel)** | 0.0412 | 21.89 dB | 0.7431 | 0.2894 | 89.14 |

### Key Qualitative Findings

* **PatchGAN Performance:** PatchGAN achieved sharp building boundaries, defined agricultural patterns, and clear road networks. Evaluating the images across local spatial patches successfully suppressed characteristic SAR speckle artifacts and generated realistic textures.
* **PixelGAN Performance:** While the PixelGAN variant achieved stable color distributions, it suffered from severe high-frequency structural blurring. Because it lacks spatial awareness across its receptive field, it produced checkerboard artifacts and structural hallucinations over complex terrain boundaries.

---

## Model Weights
Pre-trained generator and discriminator weights for both the baseline PatchGAN and ablated PixelGAN runs are available here:  
 **[Google Drive Model Weights Repository](https://drive.google.com/drive/folders/1iF4KRp_0uBXCeGlmSa33VDgo6IukkyIF?usp=sharing)**

---

## References & Citations

```text
[1] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1125-1134.

[2] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired image-to-image translation using cycle-consistent adversarial networks," in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2223-2232.

[3] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015, pp. 234-241.

[4] P. Tiwari, "Sentinel-1 & 2 Image Pairs (SAR & Optical), Segregated by Terrain," Kaggle Dataset, 2022. Available: [https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)

[5] P. Tiwari and M. Ojha, "Data-centric approach to SAR-optical image translation," in Computer Vision and Image Processing (CVIP 2022), CCIS vol. 1776, Springer, Cham, 2023.

[6] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[7] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," in Advances in Neural Information Processing Systems (NeurIPS), 2017.

[8] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.
```