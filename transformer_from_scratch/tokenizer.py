from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset, disable_caching


def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield [_ for sample in dataset[i : i + batch_size]["translation"] for _ in sample.values()]


def get_tokenizer(name: str, language: str, vocab_size: int):
    """Load a tokenizer from HuggingFace Tokenizers.

    Parameters
    ----------
    name : str
        Name of the tokenizer. Example: "wmt14".
    language : str
        Language of the tokenizer. Example: "de-en".

    Returns
    -------
    Tokenizer
        Tokenizer.
    """

    save_file = f"../tokenizer-{name}-{language}.json"
    try:
        tokenizer = Tokenizer.from_file(save_file)
        print(f"Loaded tokenizer from {save_file}")
    except:
        print(f"Creating tokenizer at {save_file}  This may take several minutes.")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        )  # type: ignore
        disable_caching()  # avoid disk explosion
        dataset = load_dataset(name, language, split="train+validation+test")
        tokenizer.pre_tokenizer = Whitespace()  # type: ignore
        tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
        tokenizer.save(save_file)
    return tokenizer
