# Goal

Learn the Transformer Model following ["Attention is All You Need" by Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

## File Structure

-   `prototype/`

    Built Transformer Base Model using `torch.nn.Transformer` for prototyping the model and the training/evaluating process.

-   `transformer_from_scratch/`

    Following the steps from the paper, built Transformer Base Model with basic PyTorch modules. Trained on WMT14 de-en translation dataset for 10 epochs, which was 30 hours on an RTX 4090. The final BLEU score is 10.4

## Dataset

Using [WMT 2014 English-German dataset](https://huggingface.co/datasets/wmt14).

By default, the dataset is downloaded at `~/.cache/huggingface/datasets/`. In the code, I've turned off dataset caching
to avoid disk explosion :)

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

## Potential Improvements

-   Better data cleaning and tokenization

    I did minimal treatments (simply removing white spaces) to tokenize the data. Could've done better. Might improve the training and the translation performance.

-   Pre-trained embedding layer

    I might help speed up the training if I could use a pre-trained embedding layer from NLP classification tasks. So that the model won't need to learn both languages from scratch from the translation dataset.