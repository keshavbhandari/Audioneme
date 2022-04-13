from pathlib import Path
import torch

# Static directories
RAW_DIR = '/content/drive/MyDrive/Research/Speech_Disorder/Speech Exemplars and Evaluation Database/'
ZIP_LOC_DRIVE = '/content/drive/MyDrive/Research/Speech_Disorder/Saved/Speech_Disorder.zip'
ZIP_LOC = '/content/Audioneme/'
PRETRAINED_ResnetSE34V2 = '/content/Audioneme/trained_models/resnetse34v2.pt'
CACHE_DIR = Path(__file__).parent.parent / 'cache'
DATA_DIR = Path('.').resolve() / 'data'

build_speaker_level_dataset = True

batch_size = 16
orig_sample_rate = 22050
new_sample_rate = 16000

model_type = "resnetse34v2"
n_mels = 64

log_interval = 100
n_epoch = 40
early_stopping_rounds = 3

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
