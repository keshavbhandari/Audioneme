# Audioneme
AI model for child speech disorder detection.
We finetune wav2vec2 for this binary classification task.

<h2 id="install">Installation</h2>

1. Clone the repository:

```
git clone https://github.com/keshavbhandari/Audioneme.git
```

2. We recommend working from a clean environment, e.g. using `conda`:

```
conda create --name audioneme python=3.9
source activate audioneme 
```

3. Install dependencies :

```
cd Audioneme
pip install -r requirements.txt
pip install -e .
```

<h2 id="usage">Usage</h2>

<h3 id="usage-data">Loading Data</h3>

Unfortunately, data cannot be shared as it is not public yet. However, this framework can be used on any speech dataset. The data loader should spit out audio signal, encoded transcription of the audio, encoded filename (used for further analysis) and a binary target.

```python
from scripts.data_loader import train_loader, val_loader, test_loader

# Audio Data, Transcription, Filename, Binary Target
for batch_idx, (data, target) in enumerate(train_loader):
    print(data[0].shape, data[1].shape, data[2].shape, target.shape)
    break
```

<h3 id="usage-data">Train Model</h3>

Train the speech recognition model on wav2vec2. Check configs first to ensure parameters and model type is correct.

```commandline
python scripts.train.py
```
