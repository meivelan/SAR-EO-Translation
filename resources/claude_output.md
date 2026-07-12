# SAR-to-EO Translation — GalaxEye Assessment Execution Plan

**Deadline:** 9 Jul 2026 (7 days from today, 3 Jul 2026)
**Candidate:** Meivelan Murugesan

## How this plan is calibrated to you

I read your resume, so I've skipped over things your **Binary Change Detection on EO-SAR Image Pairs** project already proves you know: PyTorch, U-Net + segmentation-models-pytorch, RasterIO, handling paired SAR/optical data, and dealing with class imbalance via loss weighting (Dice+Focal). That project is a real asset here — you've already looked at co-registered SAR/optical patches and built a training pipeline around them.

What's genuinely new territory based on your resume is the **adversarial (GAN) side** — you haven't listed GAN work, so pix2pix's generator/discriminator dynamics get a proper explanation below. Your ShopUp work (building eval harnesses, pass@k metrics, reusable test fixtures) is directly transferable to the Evaluation section — you already think like someone who builds rigorous eval pipelines; here it's just new metrics (LPIPS/FID) instead of field-level JSON accuracy.

---

## Assumptions

- **Datasets** (from the assignment's own table): SEN1-2 = 282,384 pairs, single-pol VV SAR + RGB, ~44GB, CC-BY 4.0. SEN12MS = 180,662 triplets, VV+VH SAR + 13-band optical + MODIS land cover, ~510GB, CC-BY. Kaggle terrain-segregated set = SAR+optical pairs split into 4 land-cover classes, size/exact channel count not published in the listing — **must be audited on Day 1**, not assumed.
- **Resources:** solo candidate, free-tier cloud GPU only (≤16GB VRAM, matching the inference contract), realistic total hands-on time across 7 days is evenings/full-days mixed with report writing — not 7×24h of compute.
- **Scope:** a working, honest MVP (pix2pix-class model, one clean ablation, full I/O-contract-compliant `infer.py`, complete report) beats a half-finished ambitious system. The assignment says this explicitly — modest numbers with real justification outrank strong numbers with none.

---

## 1. Introduction — concepts you need for this specific system

Skipping general CV/DL basics you already have. Flagging what's specific to *this* problem:

- **SAR carries no color/spectral information** — it measures radar backscatter (surface roughness, geometry, dielectric properties), not reflected sunlight. This is why the mapping to RGB is fundamentally **one-to-many and ill-posed**: many plausible optical colorings are consistent with one SAR return. This is the crux of the whole assignment.
- **Speckle noise** is multiplicative, not additive — it's a coherent-imaging artifact, not sensor noise you can denoise away without losing signal. Don't treat it like Gaussian noise.
- **SAR amplitude has a huge dynamic range** and is conventionally viewed in **dB (log) scale** — this is why the inference contract mandates dB-scaled, min-max-normalized 8-bit input. This preprocessing choice isn't cosmetic; it's specified for you.
- **pix2pix mechanics** (new relative to your resume): a conditional GAN with two networks trained adversarially — a **generator** (here, a U-Net: SAR in, RGB out) trying to fool a **discriminator** (a PatchGAN that classifies *local patches* of the output as real/fake, conditioned on the input SAR). The generator loss combines pixel-level L1 (accuracy) with adversarial loss (realism).
- **Why this matters for metrics:** a generator trained with **pixel loss alone** learns to hedge across all plausible outputs and produces a blurry "average" image — it scores decently on PSNR (which rewards low MSE) but looks fake. Adding adversarial loss pushes the generator toward one sharp, plausible output instead of the blurry average — better perceptual quality, often *worse* PSNR. This is exactly the "pixel-vs-perceptual gap" the assignment tells you to discuss, and it's also your ablation (Section 4).
- **LPIPS** = distance in a pretrained CNN's feature space, correlates with human judgment of "does this look real," not pixel-aligned. **FID** = distance between the distribution of real vs. generated image features (Inception embeddings) over a whole set — measures realism/diversity at the population level, not per-pair fidelity. Neither cares about exact pixel alignment, which is the point.

---

## 2. Literature Survey — is there a paper that matches this exact task?

Yes — closer than you might expect, on two different axes.

**Closest possible match — same dataset, same task:** Tiwari & Ojha, *"Data-Centric Approach to SAR-Optical Image Translation"* (CVIP 2022 / Springer CCIS 1776, 2023). This is written by the same person who published the Kaggle terrain-segregated dataset you're leaning toward — Paritosh Tiwari is both the dataset author and the paper author. The paper's core claim: segregating SAR-optical training data by land-cover/terrain type and training per-terrain models outperforms a single generic model, because each terrain has distinct backscatter-to-appearance statistics (water, urban, vegetation, barren land all "translate" differently). **This is directly relevant to a decision you have to make** (Section 3) — and it also creates a real tension: the paper's winning strategy requires terrain labels, but GalaxEye's blind evaluation set explicitly does not provide them. I address this below rather than just citing the paper and moving on.

**Same dataset family, task demonstrated but not the focus:** Schmitt, Hughes & Zhu, *"The SEN1-2 Dataset for Deep Learning in SAR-Optical Data Fusion"* (2018, the paper that introduced SEN1-2). The dataset's own creators ran a preliminary pix2pix experiment generating "artificial optical images from SAR input" as one of several proof-of-concept applications (alongside SAR colorization and SAR-optical patch matching) — it's illustrative, not a rigorously benchmarked study, but it establishes that pix2pix-on-SEN1-2 is a well-trodden, sane starting point rather than a novel risk.

**Quantitative reference point:** a 2022 IJCAI paper (*"SAR-to-Optical Image Translation via Neural Partial Differential Equations"*) benchmarks vanilla pix2pix on SEN1-2 test splits at roughly PSNR ≈ 16–17 dB and SSIM ≈ 0.27–0.35. Useful for calibrating your own expectations — if your pix2pix baseline lands in a similar ballpark, that's a sane result, not a failure.

**Broader field context (for your report's literature review, not a task-match):** a systematic review (*"Generative models for SAR–optical image translation,"* ScienceDirect, late 2025) benchmarks 8 representative methods spanning CGAN/pix2pix, CycleGAN variants, and diffusion models, and explicitly frames the same pixel-vs-perceptual tension the assignment raises. A very recent diffusion-based paper (*OSCAR*, Jan 2026) works on SEN12MS specifically and shows the field's frontier is moving toward diffusion — worth one sentence in your report's "Future Work," not something to attempt this week (diffusion training/sampling cost doesn't fit a 7-day free-tier budget).

**Bottom line for your report's literature survey:** cite Tiwari & Ojha as the closest work (same exact dataset), Schmitt et al. as the dataset-origin baseline, the IJCAI paper for quantitative calibration, and the 2025 review + OSCAR for field context and forward-looking framing.

---

## 3. Dataset

### Decision — Cloud platform

- **Recommendation:** Kaggle Notebooks as primary compute, with Lightning AI as a secondary pool for early/cheap experiments only.
- **Why:** the Kaggle terrain-segregated dataset is itself a *Kaggle Dataset* — attaching it to a Kaggle Notebook costs zero download time, versus fetching it externally on Lightning AI. More importantly, GalaxEye's blind evaluation explicitly targets "Colab/Kaggle free tier, ≤16GB VRAM" — developing on Kaggle (P100/T4, 16GB) gives you dev/eval environment parity, so `infer.py` behavior you validate locally is what GalaxEye will actually see. Kaggle's quota (~30 GPU-hours/week, resets weekly, 9–12h session cap) is predictable.
- **Tradeoff:** Lightning AI's free tier gives more raw hours (80 GPU-hours/month, available immediately rather than metered weekly) and persistent Studios (no re-setup per session, unlike Kaggle's session-based environment). If you burn through Kaggle's weekly cap mid-week, Lightning AI is a good overflow, not a replacement.
- **Best choice under constraints?** Yes — use Kaggle for the pipeline you'll actually submit against (dev/prod parity matters more than raw hours here, since your dataset is modest and a pix2pix-256 model is cheap to train), and treat Lightning AI purely as backup headroom if you need it Day 4–6.

### Decision — Dataset choice: single vs. combination

- **Recommendation:** Kaggle terrain-segregated set alone, as you were already leaning toward.
- **Why:** it's the smallest, fastest to iterate on, has a directly-matching prior paper (Section 2), and — this is the strongest practical reason — the assignment's phrasing ("if trained on additional channels, e.g. VV+VH from SEN12MS...") strongly implies SEN1-2-family data (which the Kaggle set derives from) is **single-channel VV**, exactly matching the inference contract's input format. If confirmed on Day 1, this means **zero channel-adaptation logic needed in `infer.py`** — one less thing that can silently break during GalaxEye's blind evaluation. Combining datasets (SEN1-2 44GB, SEN12MS 510GB) adds real engineering cost: harmonizing channel counts, differing normalization statistics, and non-trivial download/storage time you don't have.
- **Tradeoff:** a combined dataset would very likely generalize better to unseen geographies (the actual thing being scored most heavily) — more scenes, more seasons, more diversity. You're trading some generalization headroom for a much lower risk of running out of time.
- **Best choice under constraints?** Yes, as primary strategy. But add this cheap, high-value idea: **use a small, untrained-on subset of SEN1-2 purely as an extra zero-shot validation check** — since it's a different (but related) distribution you never trained on, decent performance there is a real signal about how you'll do on GalaxEye's true blind set. This costs almost nothing (no training, just running your eval script on a few hundred external pairs) and speaks directly to the "Generalisation" criterion the rubric weights heavily. Do this only if Day 5–6 has slack.

### Decision — Terrain-aware vs. terrain-agnostic training

- **Recommendation:** terrain-agnostic training on a terrain-*balanced* pool (stratified sampling across all 4 land-cover classes), not separate per-terrain models.
- **Why:** Tiwari & Ojha's own paper shows terrain-specialized models win — but that requires knowing the terrain at inference time to route to the right specialist, and GalaxEye's blind set gives you SAR only, no terrain label. A terrain-specialist approach you can't invoke at inference time is a dead end for the graded evaluation, however good it looks on your own held-out set. What you *can* borrow from their insight without needing labels: make sure training data is balanced across terrains (not dominated by whichever class has the most patches), so the single model sees the full diversity of backscatter-to-color relationships rather than overfitting to one dominant terrain.
- **Tradeoff:** a single pooled model will likely underperform Tiwari & Ojha's per-terrain numbers on your own validation set. That's fine — it's the right tradeoff for a system that has to work on unlabeled blind data.
- **Best choice under constraints?** Yes. Explicitly write this reasoning into your report — it directly demonstrates the "trade-offs" and "research thinking" the rubric asks for, using a real paper as the counterpoint rather than just asserting it.

### Decision — Modality-specific normalization/preprocessing

- **Recommendation:** SAR → dB-scale the amplitude, then min-max normalize to [0, 255] as **8-bit PNG**, matching the inference contract exactly. Optical → standard 8-bit RGB, normalized to [-1, 1] at load time for a tanh-output generator.
- **Why:** the inference contract *is* your preprocessing spec — inputs arrive already dB-scaled and min-max normalized to [0,255]. If your training-time preprocessing doesn't produce the same distribution, you'll have a train/inference mismatch that tanks blind-set performance for reasons that have nothing to do with model quality. Store the canonical dB+minmax 8-bit PNG as your on-disk training artifact (so it's identical in nature to what `infer.py` will receive), and do any further network-specific scaling (e.g., to [-1,1]) as a transform applied identically in both `train.py` and `infer.py`. SAR and optical are normalized *separately* because they have unrelated dynamic ranges and noise statistics — sharing normalization stats across them makes no physical sense.
- **Tradeoff:** none real here — this isn't a stylistic choice, it's dictated by the I/O contract. The only "choice" is discipline: keep the exact dB formula and min-max bounds identical everywhere.
- **Best choice under constraints?** Yes, and non-negotiable — get this exactly right on Day 1 before writing any model code, since every downstream step depends on it.

### Decision — Train/val/test split

- **Recommendation:** scene/tile-aware split (not patch-level random shuffle), stratified so each split gets proportional representation from all 4 terrain classes.
- **Why:** the assignment explicitly warns that adjacent patches can be near-duplicates and that naive random splitting leaks information and inflates your numbers. On Day 1, check whether filenames encode a source scene/tile ID — if yes, group by that ID before splitting. If the Kaggle set doesn't expose this, fall back to splitting in contiguous blocks per terrain folder (assuming patches within a folder are stored in spatially coherent order) and **say so explicitly in the report** as a documented limitation — the assignment rewards honesty about this exact kind of constraint far more than it penalizes an imperfect-but-justified split.
- **Tradeoff:** a perfect scene-aware split may leave you with fewer usable groups than patches, especially for FID, which wants a reasonably large test set to be stable. Roughly 80/10/10 is a sane starting ratio; adjust once you see actual counts on Day 1.
- **Best choice under constraints?** Yes.

### Decision — Augmentation and feature engineering

- **Recommendation:** horizontal/vertical flips + 90° rotations, applied identically to the SAR/optical pair (same transform, same parameters). No color jitter (would break the physical SAR→color correspondence), no elastic/arbitrary-angle rotation (introduces alignment error into already co-registered pairs), no hand-crafted feature engineering (GLCM texture, edge maps, etc.).
- **Why:** flips/90° rotations are geometry-preserving and terrain-orientation is not physically meaningful, so they're free label-preserving augmentation. A CNN generator learns its own features end-to-end; hand-engineering texture features for a pix2pix baseline is extra engineering time for uncertain payoff under a 7-day budget.
- **Tradeoff:** you give up any accuracy edge that heavier augmentation or engineered features might buy — acceptable, since the bigger risk this week is running out of time on report/eval, not undertraining the generator.
- **Best choice under constraints?** Yes for the MVP. Note "feed an edge/gradient map as an auxiliary structural input channel" as a specific, concrete Future Work item (this mirrors published feature-guided S2O approaches) — cheap to write, shows awareness, correctly deferred.

---

## 4. Model

### Decision — Architecture

- **Recommendation:** pix2pix, using the official `junyanz/pytorch-CycleGAN-and-pix2pix` implementation (U-Net-256 generator, 70×70 PatchGAN discriminator), adapted to `--input_nc 1 --output_nc 3`.
- **Why:** you explicitly don't want to implement from scratch, and this repo is the canonical, most heavily used and documented pix2pix implementation, directly supporting single-channel input via a flag rather than a rewrite. It matches what the assignment itself suggests and what nearly every SAR-optical translation paper you'll cite uses as their baseline — so your methodology section can lean on well-established defaults instead of defending an untested setup.
- **Tradeoff vs. alternatives:** CycleGAN (unpaired) is unnecessary — you have paired, co-registered data, so supervised pix2pix is strictly more informative. Conditional diffusion is the field's current frontier (per the 2025 review and OSCAR) but has materially higher training/sampling cost, unstable behavior with under a week of tuning time, and inference-time cost that risks breaking the ≤16GB VRAM / no-internet inference constraint if not carefully engineered. The assignment text itself calls diffusion "a stretch, not a requirement."
- **Best choice under constraints?** Yes — a well-executed pix2pix baseline with careful preprocessing and one clean ablation is exactly the "modest numbers, real justification" outcome the assignment says it values over an ambitious, under-baked system.

### Decision — Training recipe (loss + optimizer)

- **Recommendation:** the pix2pix paper's standard recipe as your starting point — combined L1 + adversarial loss (λ_L1 = 100), vanilla GAN loss mode, Adam optimizer (lr = 2e-4, β1 = 0.5, β2 = 0.999).
- **Why:** these are the most battle-tested defaults in the literature (used across nearly every pix2pix-on-SAR paper found in Section 2) — starting from a well-validated recipe minimizes the risk of wasting your limited compute budget chasing an unstable configuration.
- **Tradeoff:** LSGAN loss mode is sometimes more stable than vanilla and is a one-flag change in this repo if vanilla training looks unstable (oscillating losses, mode collapse) — keep it as your Plan B, not your starting point, since the default is the better-documented choice.
- **Best choice under constraints?** Yes — start with defaults, only deviate if you observe a concrete problem (this also gives you something interesting and honest to write in "what you tried and why you moved on").

### Decision — Ablation

- **Recommendation:** L1-only vs. L1 + adversarial loss (disable the discriminator term by zeroing its weight for the ablation run, keep everything else — architecture, data, epochs — identical).
- **Why:** this is the cheapest ablation to implement (it's a config flag, not new code), and it directly *generates the evidence* for the pixel-vs-perceptual discussion the assignment explicitly wants in your Results section. You get two required deliverables (a controlled ablation + the perceptual-gap discussion) from one experiment: expect the L1-only run to show better PSNR/SSIM but visibly blurrier, worse LPIPS/FID output — a clean, expected, well-understood result you can explain with confidence rather than a confusing outcome you have to hand-wave around.
- **Tradeoff vs. alternatives:** "with/without perceptual (VGG) loss" is also a good ablation but requires implementing and tuning an extra loss term and its weight — more engineering time for a less direct connection to the assignment's specific framing. "Generator capacity" (e.g., filter count) is trivial to run but scientifically less interesting.
- **Best choice under constraints?** Yes.

### Decision — Hyperparameter tuning

- **Recommendation:** a small manual/grid "smoke test" — 3–4 configs varying learning rate and λ_L1, run for a handful of epochs on a data subset, purely to catch instability before committing to the full run. No automated Bayesian sweep (Optuna, W&B Sweeps).
- **Why:** automated search tools earn their keep when you can afford dozens of trials; with a 7-day, single-GPU, free-tier budget you can realistically afford a handful of short runs, at which point manual/grid search gets you the same practical benefit (avoid an obviously bad config) without the setup overhead of a sweep orchestrator.
- **Tradeoff:** you likely leave some performance on the table versus a properly tuned model — acceptable, since the rubric explicitly rewards justified, well-reasoned modest results over unexplained strong ones.
- **Best choice under constraints?** Yes.
- **Tool if you want one anyway:** Weights & Biases free tier is fine for the smoke-test phase (nice dashboards, low setup cost) — but don't make your *mandatory* deliverable depend on it. The assignment requires local CSV/JSON + a saved loss-curve plot in the repo regardless; build that logging directly into `train.py` (see blueprint below) so it always works even if W&B has an auth hiccup mid-week. Mirror to W&B optionally, on top of that.

---

## 5. Evaluation

**Metrics** (report on both val and test splits, per the assignment):
- **LPIPS** (primary) — `pip install lpips`, AlexNet backbone (`lpips.LPIPS(net='alex')`), the standard default.
- **FID** (primary) — `pytorch-fid` for simplicity, or `clean-fid` if you have time (its standardized resizing reduces the well-known implementation-variance in FID scores across libraries). Compute over the *full* test set, not per-batch, since FID needs a distribution.
- **SSIM / PSNR** (secondary) — `torchmetrics` gives you both with a consistent, GPU-batched API alongside your other metrics; `skimage.metrics` is an equally valid lighter-weight alternative if you'd rather stay in a library you already know.

**Setup:** wrap all four in a single `eval.py` invocable as `python eval.py --pred_dir <eo> --gt_dir <gt>`, matching the README requirement, and have it emit one metrics table per split (val, test) plus your ablation run — so Results-section tables come directly from script output, not manual transcription.

**Qualitative:** select 5+ SAR→generated→ground-truth triplets deliberately, not randomly — include at least one clear success and one clear failure. Based on the literature surveyed above and the assignment's own hints, the likely failure modes worth specifically looking for are **dense urban/built-up areas** (structural edges and fine geometry are hard to hallucinate correctly from SAR) and **water bodies** (specular SAR return gives little texture signal, so color/boundary can be ambiguous) — go looking for these categories in your test set rather than picking triplets at random, and comment on *why* each failure happens, not just that it does.

---

## 6. Implementation Blueprint

```
sar2eo/
├── data/
│   ├── raw/                     # downloaded Kaggle subset
│   ├── processed/                # dB-scaled/minmax 8-bit SAR PNGs + normalized RGB PNGs
│   └── splits/                   # train.txt / val.txt / test.txt (scene-grouped)
├── src/
│   ├── preprocessing.py
│   ├── datasets.py
│   ├── transforms.py
│   ├── models/
│   │   ├── generator.py          # U-Net-256, in_channels=1, out_channels=3
│   │   ├── discriminator.py      # 70x70 PatchGAN
│   │   └── losses.py
│   ├── train.py
│   ├── eval.py
│   ├── infer.py                  # exact I/O contract CLI
│   └── utils/
│       ├── logging_utils.py
│       └── metrics.py
├── configs/
│   └── config.yaml
├── outputs/
│   ├── checkpoints/
│   ├── loss_curve.png
│   ├── loss_log.csv
│   └── qualitative/
├── notebooks/                     # Day-1 data exploration
├── requirements.txt
└── README.md
```

| Module | Function | Purpose |
|---|---|---|
| `preprocessing.py` | `sar_to_db(amplitude)` | Convert raw SAR amplitude to dB scale |
| | `minmax_normalize(img, out_range=(0,255))` | Match inference contract exactly; used identically in train and infer |
| | `preprocess_pair(sar_path, optical_path)` | Full pipeline for one training pair, writes canonical 8-bit PNGs to `data/processed/` |
| `datasets.py` | `SAR2EODataset(Dataset)` | Loads preprocessed pairs, applies transforms, returns tensors |
| | `build_scene_split(root, val_frac, test_frac, seed)` | Scene/tile-grouped, terrain-stratified split; writes `splits/*.txt` |
| `transforms.py` | `joint_transform(sar, optical)` | Applies identical flip/rotation to both images in a pair |
| `models/generator.py` | `UNetGenerator(in_channels=1, out_channels=3, ngf=64)` | pix2pix generator, single-channel input adapted |
| `models/discriminator.py` | `PatchGANDiscriminator(in_channels=4)` | Input = SAR (1ch) concatenated with RGB (3ch) |
| `models/losses.py` | `l1_loss`, `gan_loss(mode='vanilla')`, `combined_loss(lambda_l1=100)` | Loss composition; `lambda_gan=0` toggles the ablation run |
| `train.py` | `train_one_epoch()`, `validate()`, `main(config)` | Training loop; calls logging every epoch |
| `utils/logging_utils.py` | `log_epoch(epoch, train_loss, val_loss, csv_path)` | Mandatory per-epoch CSV/JSON logging |
| | `plot_loss_curve(csv_path, out_png)` | Saves `outputs/loss_curve.png` |
| `utils/metrics.py` | `compute_lpips`, `compute_fid`, `compute_ssim`, `compute_psnr` | Wraps external libraries with a consistent interface |
| `eval.py` | CLI: `--pred_dir --gt_dir` | Runs all four metrics, prints/saves a results table |
| `infer.py` | CLI: `--input_dir --output_dir --weights` | Loads model with no internet access, runs inference, writes matching filenames |

---

## 7-Day Timeline

| Day | Date | Focus | Key outputs |
|---|---|---|---|
| 1 | Fri Jul 3 | **Data audit + setup.** Confirm Kaggle dataset channel count, size, per-terrain counts, filename/scene structure. Set up Kaggle notebook + GitHub repo skeleton. Build & verify `preprocessing.py` against the provided format-sample pack. Quick literature pass (the papers above + a scan for anything newer). | Verified preprocessing pipeline, repo skeleton, data audit notes for the report |
| 2 | Sat Jul 4 | **Pipeline build.** `datasets.py`, scene-aware split, augmentations, clone/adapt pix2pix repo for 1-channel input. Dry-run: overfit a tiny subset to confirm the pipeline is correct end-to-end. Measure per-epoch time on real hardware (needed for the resource log anyway). | Working dataloader + sanity-checked training loop |
| 3 | Sun Jul 5 | **Kick off baseline training** (L1+adversarial) — start it early, let it run in the background/overnight. While it runs: write Introduction + Literature Survey sections of the report, build `eval.py` and `infer.py` skeletons. | Baseline training underway, report sections drafted, eval/infer scaffolding |
| 4 | Mon Jul 6 | Finish baseline training. Start the **ablation run** (L1-only). Implement `infer.py` fully, test it against the format-sample pack for exact I/O-contract compliance. | Baseline weights, ablation running, infer.py verified |
| 5 | Tue Jul 7 | Finish ablation. Run `eval.py` on val/test for both runs. Generate qualitative triplets (success + failure cases, targeting urban/water areas specifically). If time: the zero-shot SEN1-2 generalization check (Section 3). Draft Methodology + Results. | Full metrics table, qualitative figures, Methodology/Results drafted |
| 6 | Wed Jul 8 | **Buffer day.** Fix anything metrics revealed; retrain only if clearly broken. Upload final weights to a public link. Finalize README against the exact spec. Fresh-clone test of the repo. Write Future Work + Conclusion + Time/Resource log. | Public weights link, reproducible repo, complete report draft |
| 7 | Thu Jul 9 | **Submission day.** Proofread report. Verify: repo is public, weights link has no access gate, `infer.py` runs clean on a fresh clone, ZIP is correctly named and not corrupted. Submit via the form. | Submission complete, with buffer for last-minute issues |

Training runs are scheduled to start Day 3 deliberately — that gives two full days of buffer (6–7) before the deadline in case a run needs to be restarted, and lets you write report sections *while* the GPU is busy instead of sitting idle.

---

## Reporting Checklist (mapped to the assignment's own rubric)

**README.md** (Section 5.1.1): title/description, requirements + pinned `requirements.txt`, environment setup commands, dataset structure, training command, inference command (exact contract), evaluation command, public weights link, results table + loss curve, citations (Kaggle dataset **and** its underlying Sentinel-data attribution, since the license note says "cite source").

**Report sections in order** (Section 5.2): Abstract → Literature Survey → Methodology (architecture, loss, preprocessing, split strategy, ablation, and *why* for each) → Results (LPIPS/FID/SSIM/PSNR on both splits, ablation table, ≥5 qualitative triplets, pixel-vs-perceptual discussion, error profile, loss curves with convergence commentary) → Future Work (be specific — combined-dataset fine-tuning, an SAR-texture-based terrain classifier as an internal router reviving Tiwari & Ojha's approach without needing labels at inference, structural auxiliary input channels, diffusion as a longer-term direction) → Conclusion (honest limitations: modest dataset size, imperfect split if scene IDs weren't available, FID variance at your test-set size).

**Time & Resource Log** (Section 5.3): time by activity, machine/GPU/VRAM used, per-epoch and total training time (measure this Day 2, don't estimate it Day 7), constraints and how they shaped decisions.

**Submission mechanics:** public GitHub repo (not just "unlisted"), `config.yaml` with every hyperparameter, per-epoch loss CSV/JSON + `loss_curve.png` committed to the repo, public weights link with no access-request gate, ZIP named exactly `FirstName_LastName_GalaxEye.zip`, weights **excluded** from the ZIP (link only) — the assignment says any of these being wrong means the submission isn't evaluated at all, so this is worth a dedicated 30 minutes on Day 7, not a rushed last step.