from pathlib import Path
import torch

# Static directories
RAW_DIR = '/content/drive/MyDrive/Research/Speech_Disorder/Speech Exemplars and Evaluation Database/'
ZIP_LOC_DRIVE = '/content/drive/MyDrive/Research/Speech_Disorder/Saved/Speech_Disorder.zip'
ZIP_LOC = '/content/Speech_Disorder.zip'
CACHE_DIR = Path(__file__).parent.parent / 'cache'
DATA_DIR = Path(__file__).parent.parent / 'data'

batch_size = 16
new_sample_rate = 8192  # 16000

load_saved_data = True
copy_files_as_zip = False
run_test = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

log_interval = 100
n_epoch = 40