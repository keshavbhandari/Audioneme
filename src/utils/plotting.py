import io
import torch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
from IPython.core.display import display

from PIL import Image
from torchvision.transforms import ToTensor

from typing import Union


def tensor_to_np(x: torch.Tensor):
    return x.clone().detach().cpu().numpy()


def play_audio(x: torch.Tensor, sample_rate: int = 16000):
    display(ipd.Audio(tensor_to_np(x).flatten(), rate=sample_rate))


def plot_waveform(x: torch.Tensor, scale: Union[int, float] = 1.0):
    """
    Given single audio waveform, return plot as image
    """
    try:
        assert len(x.shape) == 1 or x.shape[0] == 1
    except AssertionError:
        raise ValueError('Audio input must be single waveform')

    # waveform plot
    fig, ax = plt.subplots(figsize=(8,8))
    fig.subplots_adjust(bottom=0.2)
    plt.xticks(
        #rotation=90
    )
    ax.plot(tensor_to_np(x).flatten(), color='k')
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Waveform Amplitude")
    plt.axis((None, None, -scale, scale))  # set y-axis range

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot as image
    return ToTensor()(np.array(img))


def plot_spectrogram(x: torch.Tensor):
    """
    Given single audio waveform, return spectrogram plot as image
    """
    try:
        assert len(x.shape) == 1 or x.shape[0] == 1
    except AssertionError:
        raise ValueError('Audio input must be single waveform')

    x = x.clone().detach()

    # spectrogram plot
    spec = torch.stft(x.reshape(1, -1),
                      n_fft=512,
                      win_length=512,
                      hop_length=256,
                      window=torch.hann_window(
                          window_length=512
                      ).to(x.device),
                      return_complex=True,
                      center=False
                      )
    spec = torch.squeeze(
        torch.abs(spec) / (torch.max(torch.abs(spec)))
    )  # normalize spectrogram by maximum absolute value

    # save plot to buffer
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pcolormesh(tensor_to_np(torch.log(spec + 1)), vmin=0, vmax=.31)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot image as tensor
    return ToTensor()(np.array(img))