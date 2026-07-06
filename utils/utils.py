import logging
from pathlib import Path
from utils.config import Config
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import os


def setup_logging(config: Config, run_log_dir: Path = None, log_name: str = "training.log"):
    """Setup logging configuration safely matching the isolated run system."""
    if run_log_dir is not None:
        # Use the custom log_name inside the specific run/evaluation/inference folder
        log_file = Path(run_log_dir) / log_name
    else:
        base_dir = Path(config["logging"]["local"].get("base_dir", "kaggle/working/runs/experiments"))
        log_file = base_dir / f"global_{log_name}"

    log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging initialized. Writing logs to: {log_file}")


def init_comet(config: Config):
    """Safe fallback wrapper keeping compatibility intact if Comet is fully stripped."""
    return None


def log_metrics(experiment, metrics: dict, step: int, last: int = None):
    """Fallback local terminal metric logger matching the old signature."""
    if last:
        logging.info(f"Epoch {step}/{last} metrics: {metrics}")
    else:
        logging.info(f"Epoch {step} metrics: {metrics}")

def copy_dataset_to_kaggle_memory():
    # create sample dataset and sample run
    terrains = ['agri', 'barrenland', 'grassland', 'urban']
    dataset_path = Path("/kaggle/input/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/v_2")
    new_dataset_path = Path("/kaggle/working/sentinel12-image-pairs-segregated-by-terrain/v_2")

    # create directories and copying files    
    for terrain in terrains:
        for s in ['s1', 's2']:
            files_path_new = new_dataset_path / terrain / s
            os.makedirs(files_path_new, exist_ok=True)
            files_path_old = dataset_path / terrain / s
            for file in sorted(os.listdir(files_path_old)):
                shutil.copy(files_path_old / file,  files_path_new)
    tmp = True
    for terrain in terrains:
        for s in ['s1', 's2']:
            files_path_new = new_dataset_path / terrain / s
            if len(os.listdir(files_path_new)) == 4000:
                continue
            tmp = False
            print(f"{files_path_new} doesn't copied properly, contains only {len(os.listdir(files_path_new))}")
    if tmp:
        print(f"sample dataset created successfully at {new_dataset_path}, each containing {len(os.listdir(new_dataset_path / "agri" / "s1"))} samples.")

def show_sample_outputs(path):
    image = plt.imread(path)
    plt.imshow(image)
    plt.axis('off')