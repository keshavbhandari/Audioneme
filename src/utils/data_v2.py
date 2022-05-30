import random
import torch
import os
import pandas as pd
from os import path
from pathlib import Path
from tqdm import tqdm
import librosa as li
import math

from src.utils.filesystem import ensure_dir


def get_digits(string):
    digits = []
    for e in string:
        if e.isdigit():
            digits.append(e)
        else:
            break

    return int(''.join(digits))


def get_site_task_number(file):
    file = file.lower().split('/')[-1].strip('.wav')
    site = file.split('-')[0]
    site_code = 'uc_' if 'uc' in site else 'au_'
    task_number = file.split('-')[-1].split(' ')[0].split('.')[0].split('_')[-1]
    try:
        speaker = file.split('-')[0:2]
        speaker = speaker[0] + '_' + str(get_digits(speaker[1]))
    except:
        speaker = "Error"
        print(file)

    string = []
    digits = []
    for e in task_number:
        if e.isdigit():
            digits.append(e)
        else:
            string.append(e)

    if len(digits):
        digits = [str(int(''.join(digits)))]
        string_digit = string + ['_'] + digits
        output = site_code + ''.join(string_digit)
    else:
        print(file)
        output = site_code
    return output, speaker


def build_meta_data(files: list, transcriptions: dict):
    filepath = []
    speaker_number = []
    site_task_number = []
    transcription = []

    for file in files:
        filepath.append(file)
        stn, sp = get_site_task_number(file)
        site_task_number.append(stn)
        speaker_number.append(sp)
        if stn in transcriptions.keys():
            transcription.append(transcriptions[stn])
        else:
            transcription.append('')

    transcribed_files = pd.DataFrame({
        'filepath': filepath,
        'speaker_number': speaker_number,
        'site_task_number': site_task_number,
        'transcription': transcription
    })
    print(len(transcribed_files))

    transcribed_files = transcribed_files[transcribed_files['speaker_number'] != "Error"]
    print("Length after removing erroneous speaker numbers", len(transcribed_files))

    return transcribed_files


def build_speaker_data(transcribed_files, split_ratio={'train': 0.6, 'val': 0.1}):

    all_speakers = list(set(transcribed_files['speaker_number'].values.tolist()))
    random.shuffle(all_speakers)
    train_size, val_size = int(split_ratio['train'] * len(all_speakers)), int(split_ratio['val'] * len(all_speakers))

    train = transcribed_files[transcribed_files['speaker_number'].isin(all_speakers[0: train_size])]
    validate = transcribed_files[transcribed_files['speaker_number'].isin(all_speakers[train_size: train_size + val_size])]
    test = transcribed_files[transcribed_files['speaker_number'].isin(all_speakers[train_size + val_size:])]

    print(f'Train Size: {len(train)}, Val Size: {len(validate)}, Test Size: {len(test)}')

    return train, validate, test


def build_vocab(transcription_data):
    vocab = set(transcription_data['Transcription'].sum())
    id2text = dict()
    text2id = dict()
    counter = 0
    for n, char in enumerate(vocab):
        id2text[n+1] = char
        text2id[char] = n+1

    id2text[0], text2id['UNK'] = 'UNK', 0
    return id2text, text2id


def encoder(text2id, string):
    tokens = []
    for char in string:
        token = text2id[char]
        tokens.append(token)

    # tokens = torch.tensor(tokens)
    return tokens


def decoder(id2text, tokens):
    string = []
    for token in tokens:
        char = id2text[token]
        string.append(char)

    return ''.join(string)


def get_tasks_encoded(raw_wav_files):
    task_numbered = dict()
    numbered_task = dict()
    counter = 0
    for file in raw_wav_files:
        s = file.lower()
        if s not in task_numbered.keys():
            task_numbered[s] = counter
            numbered_task[str(counter)] = s
            counter += 1

    return task_numbered, numbered_task


class SpeechDisorderDataset:

    def __init__(self,
                 meta_data,
                 encoding_lookup: list = None,
                 transcription_tokenizer: list = None,
                 split: str = 'test',
                 sample_rate: int = 16000,
                 ext: str = 'wav',
                 cache_dir: str = 'cache_data_dir/',
                 signal_duration: float = 5.0,
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
        self.transcription_decoding_lookup, self.transcription_encoding_lookup = transcription_tokenizer
        self.sample_rate = sample_rate
        self.duration = signal_duration
        self.signal_length = math.floor(signal_duration * sample_rate)
        self.scale = scale
        self.ext = ext

        # set directories

        self.cache_dir = path.join(
            os.fspath(cache_dir),
            split
        )

        # create directories if necessary
        ensure_dir(self.cache_dir)
        # self.audio_list = files
        self.meta_data = meta_data

        # check for dataset cached in tensor form
        cache_list = sorted(list(Path(self.cache_dir).rglob('*.pt')))

        if len(cache_list) > 0:
            self.tx = torch.load(path.join(self.cache_dir, 'tx.pt'))
            self.ty = torch.load(path.join(self.cache_dir, 'ty.pt'))
            self.tz = torch.load(path.join(self.cache_dir, 'tz.pt'))
            self.tt = torch.load(path.join(self.cache_dir, 'tt.pt'))

            tokenizer = torch.load(path.join(self.cache_dir, 'tokenizer.pt'))
            self.encoding_lookup = tokenizer['encoder']
            self.decoding_lookup = tokenizer['decoder']

            self.transcription_tokenizer = torch.load(path.join(self.cache_dir, 'transcription_tokenizer.pt'))
            self.transcription_encoding_lookup = transcription_tokenizer['encoder']
            self.transcription_decoding_lookup = transcription_tokenizer['decoder']
        else:
            [self.tx, self.tt, self.tz], self.ty = self._build_cache()

        self.n_classes = 2

    def _build_cache(self):

        tx, tt, ty, tz = [], [], [], []
        pbar = tqdm(self.meta_data.iterrows(),
                    total=len(self.meta_data))
        for i, row in enumerate(pbar):
            row = row[1]
            audio_fn = row['filepath']
            transcription_text = row['transcription']
            # pbar.set_description(f'Loading Speech Disorder Dataset ({path.basename(audio_fn)})')
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
                ty.append(torch.zeros(1))
            else:
                ty.append(torch.ones(1))

            s = str(audio_fn).lower()
            tz.append(torch.tensor(self.encoding_lookup[s]))

            transcription_token = encoder(self.transcription_encoding_lookup, transcription_text)
            tt.append(torch.tensor(transcription_token))

        # apply scale
        # tx *= self.scale

        torch.save(tx, path.join(self.cache_dir, 'tx.pt'))
        torch.save(tz, path.join(self.cache_dir, 'tz.pt'))
        torch.save(ty, path.join(self.cache_dir, 'ty.pt'))
        torch.save(ty, path.join(self.cache_dir, 'ty.pt'))
        torch.save({'encoder': self.encoding_lookup, 'decoder': self.decoding_lookup}, path.join(self.cache_dir, 'tokenizer.pt'))
        torch.save({'encoder': self.transcription_encoding_lookup, 'decoder': self.transcription_decoding_lookup}, path.join(self.cache_dir, 'transcription_tokenizer.pt'))

        return [tx, tt, tz], ty

    def __len__(self):
        # return self.tx.shape[0]
        return len(self.tx)

    def __getitem__(self, idx):
        return [self.tx[idx], self.tt[idx], self.tz[idx]], self.ty[idx]
