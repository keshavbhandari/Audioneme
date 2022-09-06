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

from functools import partial
import shutil
import os
import zipfile
import pandas as pd

from src.utils.filesystem import make_zipfile
from src.utils.filesystem import get_audio_files
from src.utils.data_v2 import get_tasks_encoded, build_meta_data, build_speaker_data, build_vocab, SpeechDisorderDataset
from configs.dataset_configs import *

train_transform = ComposeMany(
    [
        torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate),
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
        torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate)
    ],
    1
)


def pad_audio_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def pad_text_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch


def collate_fn(batch, dataset, augmentation):
    # A data tuple has the form:
    # waveform, utterance_type, label

    tensors, transcriptions, file_tensor, targets = [], [], [], []
    if dataset == "train" and augmentation == True:
        t = train_transform
    else:
        t = test_transform

    # Gather in lists, and encode labels as indices
    for [waveform, transcription_tokens, encoded_file], label in batch:
        tensors += [t(waveform.unsqueeze(0)).squeeze(1)]
        transcriptions += [transcription_tokens]
        targets += [label]
        file_tensor += [encoded_file]

    # Group the list of tensors into a batched tensor
    tensors = pad_audio_sequence(tensors)
    transcriptions = pad_text_sequence(transcriptions)
    # tensors = torch.stack(tensors)
    targets = torch.stack(targets)
    file_tensor = torch.stack(file_tensor)

    return (tensors, transcriptions, file_tensor), targets


# Load saved zip file if set to True
src = ZIP_LOC_DRIVE
dst = ZIP_LOC
transcription_path = TRANSCRIPTIONS_FILEPATH


if load_saved_data:
    data_dir_exists = os.path.isdir(DATA_DIR)
    if data_dir_exists == False:
        shutil.copy(src, dst)
        os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
        with zipfile.ZipFile(dst + 'Speech_Disorder.zip', 'r') as zip_ref:
            zip_ref.extractall(dst)
        # !rm / content / Speech_Disorder.zip
    train_files, val_files, test_files = [], [], []
    task_numbered, numbered_task = None, None
    id2text, text2id = None, None
    train_speakers, validate_speakers, test_speakers = None, None, None
else:
    transcription_data = pd.read_csv(transcription_path, encoding="ISO-8859-1")
    transcription_data['site_task_code'] = transcription_data['Site'].str.lower() + '_' + transcription_data[
        'Task'].str.lower() + '_' + transcription_data['Item Number'].astype(str).str.lower()
    transcription_data['Transcription'] = transcription_data['Transcription'].str.lower()
    print(transcription_data.head())
    id2text, text2id = build_vocab(transcription_data)
    transcriptions = dict(zip(transcription_data.site_task_code, transcription_data.Transcription))

    raw_wav_files = get_audio_files(wd=RAW_DIR)
    task_numbered, numbered_task = get_tasks_encoded(raw_wav_files)
    transcribed_files = build_meta_data(raw_wav_files, transcriptions)
    train_speakers, validate_speakers, test_speakers = build_speaker_data(transcribed_files, split_ratio={'train': 0.6, 'val': 0.1})

    if run_test:
        train_speakers = train_speakers.iloc[0:500]
        validate_speakers = validate_speakers.iloc[0:50]
        test_speakers = test_speakers.iloc[0:50]


train_set = SpeechDisorderDataset(meta_data=train_speakers, encoding_lookup=[task_numbered, numbered_task], transcription_tokenizer=[id2text, text2id], split='train',
                                  sample_rate=orig_sample_rate, ext='wav', cache_dir=DATA_DIR)
val_set = SpeechDisorderDataset(meta_data=validate_speakers, encoding_lookup=[task_numbered, numbered_task], transcription_tokenizer=[id2text, text2id], split='val',
                                sample_rate=orig_sample_rate, ext='wav', cache_dir=DATA_DIR)
test_set = SpeechDisorderDataset(meta_data=test_speakers, encoding_lookup=[task_numbered, numbered_task], transcription_tokenizer=[id2text, text2id], split='test',
                                 sample_rate=orig_sample_rate, ext='wav', cache_dir=DATA_DIR)

if copy_files_as_zip:
    make_zipfile(output_filename='Speech_Disorder.zip', source_dir=DATA_DIR)
    shutil.copy(dst + 'Speech_Disorder.zip', src)

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