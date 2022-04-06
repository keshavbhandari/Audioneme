import torchaudio
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)

import torch
from functools import partial
import shutil
import os
import zipfile

from src.utils.filesystem import make_zipfile
from src.utils.filesystem import get_audio_files
from src.utils.data import get_tasks_encoded, get_train_val_test_files
from src.utils.data import SpeechDisorderDataset
from configs.constants import *

new_sample_rate = 8192

train_transform = ComposeMany(
    [
        torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate),
        RandomApply([PolarityInversion()], p=0.8),
        RandomApply([Noise()], p=0.25),
        RandomApply([Gain()], p=0.3),
        RandomApply(
            [Reverb(sample_rate=new_sample_rate)], p=0.6
        ),
    ],
    1
)

test_transform = ComposeMany(
    [
        torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)
    ],
    1
)

def collate_fn(batch, dataset, augmentation):

    # A data tuple has the form:
    # waveform, utterance_type, label

    tensors, targets = [], []
    if dataset == "train" and augmentation == True:
        t = train_transform
    else:
        t = test_transform

    # Gather in lists, and encode labels as indices
    for [waveform, _], label in batch:
        tensors += [t(waveform).squeeze(1)]
        targets += [label]

    # Group the list of tensors into a batched tensor
    # tensors = pad_sequence(tensors)
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)

    return tensors, targets




raw_wav_files = get_audio_files(wd = RAW_DIR)
task_numbered, numbered_task = get_tasks_encoded(raw_wav_files)
train_files, val_files, test_files = get_train_val_test_files(raw_wav_files)

# Add if condition here for whether data exists in drive
if load_saved_data:
    src = ZIP_LOC_DRIVE
    dst = ZIP_LOC
    # target_dst = '/content/data/'
    shutil.copy(src, dst)

    os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
    with zipfile.ZipFile(dst, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    # !rm / content / Speech_Disorder.zip

if run_test:
    train_files = train_files[0:100]
    val_files = val_files[0:10]
    test_files = test_files[0:10]

train_set = SpeechDisorderDataset(files = train_files, encoding_lookup = task_numbered, split = 'train', ext = 'wav', cache_dir = DATA_DIR)
val_set = SpeechDisorderDataset(files = val_files, encoding_lookup = task_numbered, split = 'val', ext = 'wav', cache_dir = DATA_DIR)
test_set = SpeechDisorderDataset(files = test_files, encoding_lookup = task_numbered, split = 'test', ext = 'wav', cache_dir = DATA_DIR)

if copy_files_as_zip:
    make_zipfile(output_filename = 'Speech_Disorder.zip', source_dir = dst)
    shutil.copy(dst, src)


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=partial(collate_fn, dataset="train", augmentation=True),
    num_workers=num_workers,
    pin_memory=pin_memory,
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=partial(collate_fn, dataset="val", augmentation=False),
    num_workers=num_workers,
    pin_memory=pin_memory,
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=partial(collate_fn, dataset="test", augmentation=False),
    num_workers=num_workers,
    pin_memory=pin_memory,
)
