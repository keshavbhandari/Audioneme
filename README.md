# Audioneme
AI model for speech disorder detection

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

We provide simple dataset wrappers in `src.data`. After [downloading](#install), datasets can be used just as any typical PyTorch Dataset:

```python
from scripts.data_loader import train_loader, val_loader, test_loader

for batch_idx, (data, target) in enumerate(train_loader):
    print(data.shape, target.shape)
    break
```
