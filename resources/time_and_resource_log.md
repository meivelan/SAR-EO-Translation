# Time and Resource Log

### Day 1 — 02/07/2026 (Thursday)
* **Activities:** Completed an end-to-end reading of the project documentation. Established a project roadmap and schedule. Documented key constraints, unknown terminology, and critical design suggestions. Conducted initial background research and cited relevant foundational literature. Reviewed all candidate datasets and drafted a standardized report template for final submission.
* **Resources Utilized:** Project Documentation, Academic Search Engines (IEEE/Google Scholar), Markdown Editor.

### Day 2 — 03/07/2026 (Friday)
* **Activities:** Conducted deep-dive research into Generative Adversarial Networks ($c\text{GANs}$) and specific algorithmic implementations. Prioritized architectural simplicity and made key design choices:
    * **Model Selection:** Opted for **Pix2Pix** (Conditional GAN) due to the supervised, paired nature of the target task, eliminating the need for unpaired frameworks like CycleGAN.
    * **Dataset Selection:** Selected the *Sentinel-1 & Sentinel-2 Image Pairs (SAR & Optical)* dataset from Kaggle. Chose the terrain-agnostic subset (~2GB, 16,000 paired samples, $256 \times 256$ resolution in PNG format) due to its clean engineering footprint compared to larger, unstructured repositories.
* **Engineering Actions:** Evaluated the official `junyanz/pytorch-CycleGAN-and-pix2pix` repository. Forked the codebase to adapt it to this specific remote sensing use case, identifying a critical bug during setup and contributing a patch back to the open-source repository.
* **Resources Utilized:** Kaggle Dataset Registry, GitHub, PyTorch Pix2Pix Source Code, Technical Lecture Media.

### Day 3 — 04/07/2026 (Saturday)
* **Activities:** Discovered a highly optimized, domain-specific repository directly targeting this exact dataset: `yuuIind/SAR2Optical`. Shifted development focus to integrate custom training objectives, data augmentation pipelines, and rigorous evaluation metrics.
* **Engineering Actions:** Structured data augmentation routines and estimated compute-time requirements. Chose **Kaggle Kernels** as the primary compute platform. Generated a downscaled sample dataset to run pipeline diagnostics and verify framework integrity.
* **Resources Utilized:** GitHub Ecosystem, Kaggle Cloud Compute (GPU Environment).

### Day 4 — 05/07/2026 (Sunday)
* **Activities:** Substantially upgraded the project repository infrastructure. Decoupled parameters into a centralized `config.yaml` file, implemented structured command-line argument parsing (`argparse`), added automated plotting utilities, and configured independent workspace isolation for distinct execution runs.
* **Engineering Actions:** Overhauled `train.py` and `inference.py` to support multi-metric tracking, automated loss histories, and multi-panel visual inferences (plotting 5 validation outputs simultaneously). Completed comprehensive dry runs to ensure training-to-evaluation pipeline stability. Allocated the subsequent two days strictly to compute execution.
* **Resources Utilized:** Python Core Libraries (`argparse`, `pathlib`), Matplotlib, TensorBoard.

### Day 5 — 06/07/2026 (Monday)
* **Activities:** Focused heavily on full-scale model optimization and tracking integration, encountering major codebase blockers during validation metrics loop testing.
* **Debugging Notes:** Encountered a critical copy-paste bug within the logging routine. The validation loop logic correctly called `validation_step`, but reused the variable references from the training sequence. This logic conflict overrode the historical training logs. Halted execution, refactored the metric-tracking arrays, and added isolated dataframe tracking for both histories.
* **Resources Utilized:** Kaggle GPU Runtime, Pandas Logging Utilities.

### Day 6 — 08/07/2026 (Tuesday)
* **Activities:** Initialized full-scale model optimization loops. Handled unexpected runtime interruptions and environment disconnections during long-running execution threads. Parallelized compute waiting windows by compiling technical project documentation and writing the repository `README.md`.
* **Resources Utilized:** Kaggle GPU Cloud Pipelining, Technical Documentation Software.

### Day 7 — 09/07/2026 (Wednesday)
* **Activities:** Conducted a visual inspection of sample translation outputs at varying checkpoints, revealing an architectural anomaly: the 30th-epoch generation was significantly crisper and structurally cleaner than the 50th-epoch output.
* **Root Cause Analysis:** Discovered a synchronization oversight; changes modifying the discriminator configuration to PatchGAN (`netD: 'patch'`) had not been pulled down to the remote Kaggle instance prior to launching the run. Consequently, epochs 1–30 trained correctly on PatchGAN ($70 \times 70$ receptive field), while epochs 31–50 defaulted to a PixelGAN ($1 \times 1$) architecture.
* **Remediation:** Leveraged the checkpointing framework to roll back the model state to Epoch 30, resolved configuration discrepancies, and successfully resumed proper training.
but couldnt train before deadline, so I forced to use 30th epoch checkpoint. and also I completely consumed kaggle weekly free tier resource
* **Resources Utilized:** PyTorch Checkpoint Loading API, Visual Verification Scripts.

---

## Resource Management Reflection

> Overall, hardware resource allocation and compute budgeting were managed efficiently. However, subtle software configuration nuances and logging bugs introduced unexpected overhead, extending the timeline beyond initial estimations. The implementation of robust checkpointing proved critical in mitigating these issues, allowing recovery without losing early compute progress.

<!-- # Day 1 - 02/07/2026 Thu:
    read the document from top to bottom
    made a schedule
    noted
     - key points
     - questions
     - learned unknown words
     - suggestions
    cited some papers
    read about all the dataset mentioned
    made a report template for submission

# Day 2 - 03/07/2026 Fri:
    learned some about GANs
    researched a bit more for specific implementation
    made some decisions in regards with simplicity
    made some design choices
    model - pix2pix, a CGAN, because a supervised learning, there's no need for CycleGAN
    dataset - Sentinel-1&2 Image Pairs (SAR & Optical) kaggle because it already had a good feature seggregated by terrain and its significantly smaller tha other datasets. but I will go with terrain agnostic, only ~2GB with 16000 pair samples are there with 256x256 in PNG format
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix - decided to use this repo pix2pix implementation, actively maintaining, intended to 
    watched https://www.youtube.com/watch?v=RAa55G-oEuk&t=11s
    decided to fork that repo for this specific usecase
    uhmm actually, I found fixed it and contributed to open-source as well in the repo

# Day 3 - 04/07/2026 Sat:
    found gold repo exactly worked on this task with dataset I have decided to use - https://github.com/yuuIind/SAR2Optical
    there are more things to implement like objective and evaluation metrics proper metrics
    todo - augmentation proper scripts
    made some implementations
    estimated the time require for training
    chose the platform - kaggle 
    made some sample runs to check to current working
    created sample dataset

# Day 4 - 05/07/2026 Sun:
    made some improvements to existing like config.yaml, made commandline arguments, add plotting, separate directories for each runs
    improved inference.py, training.py, addedd missing metrics, addedd loss history, addedd visual inference
    made some dry runs on training and evaluating to check proper running
    added plotting 5 outputs.
    Planning to do next two days only for training and last day being correctly format the deliverables

# Day 5 - 06/07/2026 Mon:
    mostly implementation, but there were some blockers, i thought i did the implementation fully, and started training only to found out that there was, bug, while implementing traking and logging training loss and validation loss, i implemented training loss, since both are lossess, i copy pasted the training loss changed function call, but forgot to change the variables name, so it over wrote all training loss history, i had to re-implement made some tweaks

# Day 6 - 08/07/2026 Tue:
    started training
    there were some unexpected stops while executing
    parally preparing the document, and README.md files
# Day 7 - 09/07/2026 Wed:
    when I visually inspected sample output, PatchGAN varient, 30th epoch was much more cleaner than 50th epoch, and found out i didnt pulled the code in kaggle after changing descriminator as 'patch' so first 30 epoch trained on patchGAN and remaining 20 trained on pixel.

    So I had to re-run from 30th epoch

    overall, I managed resource very well, but there were some mistakes and nuances led to took time longer than I expected -->