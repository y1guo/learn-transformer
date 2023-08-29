# learn-transformer

Learning Transformer following Attention is All You Need

## Environment

I use `miniconda` to manage my environments. My default channel is `conda-forge`.

```bash
# create virtual environment
conda create -n learn-transformer python=3.10 -y
conda activate learn-transformer
# install pytorch (Apple)
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
# or install pytorch (Nvidia)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install tokenizer
pip install tokenizers
# install HuggingFace datasets
pip install datasets
# install sacrebleu for evaluating with BLEU score
pip install sacrebleu
# install dev tools
conda install jupyter matplotlib colorama -y
```

## Dataset

Using [WMT 2014 English-German dataset](https://huggingface.co/datasets/wmt14).

By default, the dataset is downloaded at `~/.cache/huggingface/datasets/`. In the code, I've turned off dataset caching
to avoid disk explosion :)

## Source Files

-   prototype/prototype.ipynb

    Built Transformer Base using `torch.nn.Transformer` for prototyping.

-   transformer_from_scratch/transformer_from_scratch.ipynb

    Built Transformer from scratch. Successful de-en translation. BLEU score 7.7
