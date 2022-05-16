import random
import torch
import os
from os import path
from pathlib import Path
from tqdm import tqdm
import librosa as li
import math

from src.utils.filesystem import ensure_dir


def get_utterance_files(files):
    random.shuffle(files)
    train_size, val_size = int(0.8 * len(files)), int(0.1 * len(files))

    train = files[0: train_size]
    validate = files[train_size: train_size + val_size]
    test = files[train_size + val_size:]

    print(f'Train Size: {len(train)}, Val Size: {len(validate)}, Test Size: {len(test)}')

    return train, validate, test


def get_digits(string):
    digits = []
    for e in string:
        if e.isdigit():
            digits.append(e)
        else:
            break

    return int(''.join(digits))


def get_speaker_files(files, split_ratio={'train': 0.8, 'val': 0.1}, external_files=[]):
    utterance_dict = dict()

    for n, i in enumerate(files):
        try:
            # a = i.lower().split('/')[-1].strip('.wav').split('-')[1]
            a = i.lower().split('/')[-1].split('-')[0:2]
            a = a[0] + '_' + str(get_digits(a[1]))

            utterance_dict[i] = a #get_digits(a)
        except:
            print("Files skipped", n, i)

    speakers = list(set(utterance_dict.values()))
    random.shuffle(speakers)
    train_size, val_size = int(split_ratio['train'] * len(speakers)), int(split_ratio['val'] * len(speakers))

    train_speakers = speakers[0: train_size]
    validate_speakers = speakers[train_size: train_size + val_size]
    test_speakers = speakers[train_size + val_size:]

    train, validate, test = [], [], []

    for key, val in utterance_dict.items():
        if val in train_speakers:
            train.append(key)
        elif val in validate_speakers:
            validate.append(key)
        else:
            test.append(key)

    print(f'Train Size: {len(train)}, Val Size: {len(validate)}, Test Size: {len(test)}')

    if len(external_files) > 0:
        train = train + external_files
        print(f'Train Size After Adding External Files: {len(train)}, Val Size: {len(validate)}, Test Size: {len(test)}')

    return train, validate, test


def get_task_stringmatch(string):
    # s = string.lower().split('/')[-1].strip('.wav').split('-')[-1].split(' ')[0].split('.')[0].split('_')[-1]
    s = string.lower()
    return s


def get_tasks_encoded(raw_wav_files):
    task_numbered = dict()
    numbered_task = dict()
    counter = 0
    for file in raw_wav_files:
        s = get_task_stringmatch(file)
        if s not in task_numbered.keys():
            task_numbered[s] = counter
            numbered_task[str(counter)] = s
            counter += 1

    return task_numbered, numbered_task


class SpeechDisorderDataset:

    def __init__(self,
                 files: list,
                 encoding_lookup: list = None,
                 split: str = 'test',
                 sample_rate: int = 16000,
                 ext: str = 'wav',
                 cache_dir: str = 'cache_data_dir/',
                 signal_length: float = 5.0,
                 scale: float = 1.0,
                 **kwargs):
        """
    :param files:
    :param split:
    :param sample_rate:
    :param ext:
    :param cache_dir:
    :param signal_length:
    :param scale:
    """

        self.encoding_lookup, self.decoding_lookup = encoding_lookup
        self.sample_rate = sample_rate
        self.duration = signal_length
        self.signal_length = math.floor(signal_length * sample_rate)
        self.scale = scale
        self.ext = ext

        # set directories

        self.cache_dir = path.join(
            os.fspath(cache_dir),
            split
        )

        # create directories if necessary
        ensure_dir(self.cache_dir)
        self.audio_list = files

        # check for dataset cached in tensor form
        cache_list = sorted(list(Path(self.cache_dir).rglob('*.pt')))

        if len(cache_list) > 0:
            self.tx = torch.load(path.join(self.cache_dir, 'tx.pt'))
            self.ty = torch.load(path.join(self.cache_dir, 'ty.pt'))
            self.tz = torch.load(path.join(self.cache_dir, 'tz.pt'))
            tokenizer = torch.load(path.join(self.cache_dir, 'tokenizer.pt'))
            self.encoding_lookup = tokenizer['encoder']
            self.decoding_lookup = tokenizer['decoder']

        else:
            [self.tx, self.tz], self.ty = self._build_cache()

        self.n_classes = 2 #torch.unique(self.ty).shape[-1]

    def _build_cache(self):

        # cache dataset in tensor form
        # tx = torch.zeros((len(self.audio_list), 1, self.signal_length))
        # ty = torch.zeros(len(self.audio_list), dtype=torch.long)
        # tz = torch.zeros(len(self.audio_list), dtype=torch.long)
        tx, ty, tz = [], [], []

        pbar = tqdm(self.audio_list,
                    total=len(self.audio_list))
        for i, audio_fn in enumerate(pbar):
            pbar.set_description(
                f'Loading Speech Disorder Dataset ({path.basename(audio_fn)})')
            waveform, _ = li.load(audio_fn,
                                  mono=True,
                                  sr=self.sample_rate,
                                  duration=None) #self.duration

            if len(waveform) > self.signal_length:
                waveform = waveform[0:self.signal_length]
            waveform = torch.from_numpy(waveform)

            # tx[i, :, :waveform.shape[-1]] = waveform
            tx.append(waveform)

            if 'non-disordered' in str(audio_fn).lower():
                # ty[i] = 0
                ty.append(torch.zeros(1))
            else:
                # ty[i] = 1
                ty.append(torch.ones(1))

            s = get_task_stringmatch(str(audio_fn))
            if s in self.encoding_lookup.keys():
                # tz[i] = self.encoding_lookup[s]
                tz.append(torch.tensor(self.encoding_lookup[s]))

        # apply scale
        # tx *= self.scale

        torch.save(tx, path.join(self.cache_dir, 'tx.pt'))
        torch.save(tz, path.join(self.cache_dir, 'tz.pt'))
        torch.save(ty, path.join(self.cache_dir, 'ty.pt'))
        torch.save({'encoder': self.encoding_lookup, 'decoder': self.decoding_lookup}, path.join(self.cache_dir, 'tokenizer.pt'))

        return [tx, tz], ty

    def __len__(self):
        # return self.tx.shape[0]
        return len(self.tx)

    def __getitem__(self, idx):
        return [self.tx[idx], self.tz[idx]], self.ty[idx]
