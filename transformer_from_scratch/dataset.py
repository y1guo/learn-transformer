import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, disable_caching
from datasets.arrow_dataset import Dataset as ArrowDataset
from tokenizers import Tokenizer
from typing import cast
from utils import NUM_PROC, round_power_two


class Dataset:
    def __init__(self, name: str, language: str, percentage: int = 100):
        """Load a dataset from HuggingFace Datasets.

        Parameters
        ----------
        name : str
            Name of the dataset. Example: "wmt14".
        language : str
            Language of the dataset. Example: "de-en".
        percentage : int, optional
            Percentage of the training set to load. Default: 100.

        Returns
        -------
        dict
            Dictionary with the train, validation and test sets.
        """

        disable_caching()  # avoid disk explosion

        # Note: the dataset is downloaded at ~/.cache/huggingface/datasets
        self.dataset: dict[str, ArrowDataset] = {}
        for split in ["train", "validation", "test"]:
            ext = f"[:{percentage}%]" if split == "train" else ""
            dataset = load_dataset(name, language, split=split + ext)
            if isinstance(dataset, ArrowDataset):
                self.dataset[split] = dataset
            else:
                raise TypeError(f"Dataset {name}/{split} is not of type datasets.arrow_dataset.Dataset")
        self.languages = language.split("-")

    def tokenize(self, tokenizer: Tokenizer):
        def encode(example):
            tokenized_example = {}
            for lg, name in zip(self.languages, ["src", "tgt"]):
                enc = tokenizer.encode(f"[CLS] {example['translation'][lg]} [SEP]")
                tokenized_example[name] = enc.ids
                tokenized_example[name + "_mask"] = enc.attention_mask
                tokenized_example[name + "_len"] = len(enc.ids)
            tokenized_example["seq_len"] = max(tokenized_example["src_len"], tokenized_example["tgt_len"])
            return tokenized_example

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(encode, num_proc=NUM_PROC)
        self.tokenizer = tokenizer

    def get_dataloader(
        self,
        split: str,
        batch_size: int = 1,
        shuffle: bool = False,
        min_len: int = 1,
        max_len: int = 128,
    ):
        """Get a PyTorch DataLoader for a given split.

        Parameters
        ----------
        split : str
            Split to load. Example: "train".
        batch_size : int, optional
            Batch size. Default: 1.
        shuffle : bool, optional
            Whether to shuffle the data. If False, data would come in order of sequence length. Default: False.
        min_len : int, optional
            Minimum length of the sequences (inclusive). Default: 1.
        max_len : int, optional
            Maximum length of the sequences (inclusive). Default: 128.

        Returns
        -------
        torch.utils.data.DataLoader
            PyTorch DataLoader.
        """

        # Filter sequences by length
        dataset = self.dataset[split].filter(lambda e: min_len <= e["seq_len"] <= max_len)

        # Sort sequences by length
        # Note that if only sort with one column, e.g. seq_len, the sorting would be extremely slow!
        # We don't rely on the ordering in the second column but it speeds things up greatly.
        dataset = dataset.sort(("seq_len", "tgt_len"), reverse=(True, True))

        # Pad sequences
        def pad(example):
            for name in ["src", "tgt"]:
                pad_len = max_len - len(example[name])
                example[name] += [self.tokenizer.token_to_id("[PAD]")] * pad_len
                example[name + "_mask"] += [0] * pad_len
            return example

        dataset = dataset.map(pad, num_proc=NUM_PROC)
        # Create DataLoader
        dataset.set_format(type="torch", columns=["src", "src_mask", "tgt", "tgt_mask"])
        dataloader = DataLoader(cast(TorchDataset, dataset), batch_size=batch_size, shuffle=shuffle)
        return dataloader
