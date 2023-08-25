# learn-transformer

Learning Transformer following Attention is All You Need

## Environment

I use `miniconda` to manage my environments. Note: my default channel is `conda-forge`.

```bash
# create virtual environment
conda create -n learn-transformer python=3.10 -y
conda activate learn-transformer
# install pytorch
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
# install tokenizer
pip install tokenizers
# install HuggingFace datasets
pip install datasets
# install dev tools
conda install jupyter matplotlib -y
```

## Dataset

Using [WMT 2014 English-German dataset](https://huggingface.co/datasets/wmt14).

By default, the dataset is downloaded at `~/.cache/huggingface/datasets/`.
