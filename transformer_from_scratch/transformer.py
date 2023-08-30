import torch, time
from tokenizers import Tokenizer
from utils import DEVICE, log, sec2hms, truncate_sequence, free_memory
from colorama import Fore
from model import TransformerModel
from sacrebleu.metrics.bleu import BLEU
from tqdm import tqdm
from datasets.arrow_dataset import Dataset as ArrowDataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from typing import cast


class Transformer:
    def __init__(
        self,
        model: TransformerModel,
        tokenizer: Tokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id("[PAD]"),
            pad_token="[PAD]",
            length=model.max_seq_len,
        )
        self.tokenizer.enable_truncation(max_length=model.max_seq_len)

    def train(
        self,
        dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        update_interval: int = 100,
        log_file: str | None = None,
    ):
        """Train the model for one epoch.

        Parameters
        ----------
        dataloader : DataLoader
                DataLoader for training data.
        loss_fn : torch.nn.Module
                Loss function. e.g. CrossEntropyLoss
        optimizer : torch.optim.Optimizer
                Optimizer. e.g. Adam
        scheduler : torch.optim.lr_scheduler.LambdaLR
                Learning rate scheduler. e.g. LambdaLR
        update_interval : int, optional
                Number of batches between logging. Default: 100
        log_file : str, optional
                Path to log file. Default: None
        """
        size = len(dataloader.dataset)  # type: ignore
        self.model.train()
        start_time = time.perf_counter()
        acc_loss, acc_correct, acc_examples, acc_tokens = 0, 0, 0, 0
        src_len, tgt_len = 0, 0
        for i_batch, batch in enumerate(dataloader):
            x, x_mask, y, y_mask = batch.values()
            x, x_mask = truncate_sequence(x.to(DEVICE), x_mask.to(DEVICE))
            y, y_mask = truncate_sequence(y.to(DEVICE), y_mask.to(DEVICE))
            if x.size(1) != src_len or y.size(1) != tgt_len:
                src_len, tgt_len = x.size(1), y.size(1)
                free_memory()

            # compute prediction error
            optimizer.zero_grad()
            pred = self.model(x, x_mask, y, y_mask)[:, :-1, :]  # (batch_size, seq_len, vocab_size)
            label = y[:, 1:]  # (batch_size, seq_len)
            label_mask = y_mask[:, 1:] == 1  # (batch_size, seq_len)
            loss = loss_fn(pred[label_mask], label[label_mask])

            # optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            # metrics
            acc_loss += loss.item() * label.shape[0]
            acc_correct += (pred.argmax(-1) == label)[label_mask].float().sum().item()
            acc_examples += label.shape[0]
            acc_tokens += label_mask.sum().item()

            # log
            if (i_batch + 1) % update_interval == 0 or i_batch == len(dataloader) - 1:
                elapsed_time = time.perf_counter() - start_time
                remaining_time = elapsed_time * (size - acc_examples) / acc_examples
                correct = 100 * acc_correct / acc_tokens
                loss = acc_loss / acc_examples
                log(
                    f"Accuracy: {correct:>4.1f}%, Avg loss: {loss:>10f}, Lr: {scheduler.get_last_lr()[0]:>10f}"
                    f"  [{acc_examples:>{len(str(size))}d}/{size}]"
                    f"  [{sec2hms(elapsed_time)} < {sec2hms(remaining_time)}]",
                    log_file,
                )
        free_memory()

    def validate(self, dataloader, loss_fn, log_file: str | None = None):
        """Validate the model on the validation set.

        Parameters
        ----------
        dataloader : DataLoader
                DataLoader for validation data.
        loss_fn : torch.nn.Module
                Loss function. e.g. CrossEntropyLoss
        log_file : str, optional
                Path to log file. Default: None
        """
        num_batches = len(dataloader)
        self.model.eval()
        validation_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                x, x_mask, y, y_mask = batch.values()
                x, x_mask, y, y_mask = x.to(DEVICE), x_mask.to(DEVICE), y.to(DEVICE), y_mask.to(DEVICE)
                label_mask = y_mask[:, 1:] == 1
                pred = self.model(x, x_mask, y, y_mask)[:, :-1, :][label_mask]
                label = y[:, 1:][label_mask]
                validation_loss += loss_fn(pred, label).item()
                correct += (pred.argmax(-1) == label).float().sum().item()
                total += label.numel()
        validation_loss /= num_batches
        correct /= total
        log(
            f"Validation: \nAccuracy: {100*correct:>0.1f}%, Avg loss: {validation_loss:>8f}",
            log_file,
        )
        free_memory()

    def predict(self, source: str, target: str):
        """Run the "predict the next token" evaluation. This can be thought to be a guided translation task.

        Parameters
        ----------
        source : str
            Source sentence.
        target : str
            Reference translation.
        """
        enc = self.tokenizer.encode(f"[CLS] {source} [SEP]")
        src = torch.tensor(enc.ids)[None, :].to(DEVICE)  # (batch_size, seq_len)
        src_mask = torch.tensor(enc.attention_mask)[None, :].to(DEVICE)  # (batch_size, seq_len)
        src, src_mask = truncate_sequence(src, src_mask)
        enc = self.tokenizer.encode(f"[CLS] {target} [SEP]")
        tgt = torch.tensor(enc.ids)[None, :].to(DEVICE)  # (batch_size, seq_len)
        tgt_mask = torch.tensor(enc.attention_mask)[None, :].to(DEVICE)  # (batch_size, seq_len)
        tgt, tgt_mask = truncate_sequence(tgt, tgt_mask)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(src, src_mask, tgt, tgt_mask)  # (batch_size, seq_len, vocab_size)
            pred_tokens = pred[0, :-1, :].argmax(-1)  # (seq_len)
            label_tokens = tgt[0, 1:]  # (seq_len)
            label_mask = tgt_mask[0, 1:] == 1  # (seq_len)
            correct = (pred_tokens == label_tokens)[label_mask].float().sum().item() / label_tokens[label_mask].numel()
            print(f"Accuracy: {100*correct:>0.1f}%")
            for i in range(len(label_tokens)):
                if label_tokens[i] == self.tokenizer.token_to_id("[SEP]"):
                    break
                predict = self.tokenizer.decode([pred_tokens[i]])
                label = self.tokenizer.decode([label_tokens[i]])
                color = Fore.GREEN if predict == label else Fore.RED
                history = self.tokenizer.decode(label_tokens[:i].tolist())
                if history:
                    history += " "
                print(f"{history}{color}{predict}{Fore.RESET}")
        free_memory()

    def translate(self, source: str, realtime: bool = False):
        """Translate a sentence.

        Parameters
        ----------
        source : str
            Source sentence.

        Returns
        -------
        str
            Translated sentence.
        """
        max_seq_len = self.model.max_seq_len
        enc = self.tokenizer.encode(f"[CLS] {source} [SEP]")
        src = torch.tensor(enc.ids)[None, :].to(DEVICE)  # (batch_size, seq_len)
        src_mask = torch.tensor(enc.attention_mask)[None, :].to(DEVICE)  # (batch_size, seq_len)
        tgt = torch.full((1, max_seq_len), self.tokenizer.token_to_id("[CLS]")).to(DEVICE)  # (batch_size, seq_len)
        tgt_mask = torch.zeros((1, max_seq_len)).to(DEVICE)  # (batch_size, seq_len)
        self.model.eval()
        with torch.no_grad():
            src, src_mask = truncate_sequence(src, src_mask)
            for i in range(max_seq_len - 1):
                tgt_mask[:, i] = 1
                pred = self.model(src, src_mask, tgt, tgt_mask)  # (batch_size, seq_len, vocab_size)
                pred_token = pred.argmax(-1)[0, i]
                tgt[0, i + 1] = pred_token
                if realtime:
                    print(self.tokenizer.decode([pred_token]), end=" ")
                if pred_token == self.tokenizer.token_to_id("[SEP]"):
                    break
            if realtime:
                print()
        translation = cast(str, self.tokenizer.decode(tgt[0].tolist()))
        free_memory()
        return translation

    def evaluate_bleu(self, dataloader: DataLoader):
        """Evaluate BLEU score on a dataset.

        Parameters
        ----------
        dataloader : DataLoader

        Returns
        -------
        result
            BLEU score object. result.score is the BLEU score.
        references : list[str]
            List of reference translations.
        hypotheses : list[str]
            List of predicted translations.
        """
        max_seq_len = self.model.max_seq_len
        self.model.eval()
        with torch.no_grad():
            references, hypotheses = [], []
            for batch in tqdm(dataloader):
                x, x_mask, y, y_mask = batch.values()
                x, x_mask = x.to(DEVICE), x_mask.to(DEVICE)
                batch_size = len(x)
                ref_list = [list(label[mask]) for label, mask in zip(y, y_mask == 1)]
                tgt = torch.full((batch_size, max_seq_len), self.tokenizer.token_to_id("[CLS]")).to(DEVICE)
                tgt_mask = torch.zeros((batch_size, max_seq_len)).to(DEVICE)
                tgt_list = [[] for _ in range(batch_size)]
                finished = [False] * batch_size
                for i in range(max_seq_len - 1):
                    # tgt_mask is a batchsize x seq_len tensor where the first i tokens are 1 and the rest are 0
                    tgt_mask[:, i] = 1
                    pred = self.model(x, x_mask, tgt, tgt_mask)
                    pred_tokens = pred.argmax(-1)[:, i]
                    for j in range(batch_size):
                        if not finished[j]:
                            tgt[j, i + 1] = pred_tokens[j]
                            tgt_list[j].append(pred_tokens[j].item())
                            if pred_tokens[j] == self.tokenizer.token_to_id("[SEP]"):
                                finished[j] = True
                    if all(finished):
                        break
                references.extend([self.tokenizer.decode(ref_list[i]) for i in range(batch_size)])
                hypotheses.extend([self.tokenizer.decode(tgt_list[i]) for i in range(batch_size)])
        result = BLEU().corpus_score(hypotheses, [references])
        free_memory()
        return result, references, hypotheses

    # def evaluate_bleu(self, dataset: ArrowDataset, batch_size: int = 32):
    #     """Evaluate BLEU score on a dataset.

    #     Parameters
    #     ----------
    #     dataset : datasets.arrow_dataset.Dataset

    #     Returns
    #     -------
    #     result
    #         BLEU score object. result.score is the BLEU score.
    #     references : list[str]
    #         List of reference translations.
    #     hypotheses : list[str]
    #         List of predicted translations.
    #     """
    #     # Sort sequences by length to increase efficiency
    #     dataset = dataset.sort("tgt_len")

    #     # Filter sequences by length
    #     dataset = dataset.filter(lambda e: max(len(e["src"]), len(e["tgt"])) <= MAX_SEQ_LEN)

    #     # Pad sequences
    #     def pad(example):
    #         for name in ["src", "tgt"]:
    #             pad_len = MAX_SEQ_LEN - len(example[name])
    #             example[name] += [self.tokenizer.token_to_id("[PAD]")] * pad_len
    #             example[name + "_mask"] += [0] * pad_len
    #         return example

    #     dataset = dataset.map(pad)
    #     # Create DataLoader
    #     dataset.set_format(type="torch", columns=["src", "src_mask", "tgt", "tgt_mask"])
    #     dataloader = DataLoader(cast(TorchDataset, dataset), batch_size=batch_size, shuffle=False)

    #     self.model.eval()
    #     with torch.no_grad():
    #         hypotheses = []
    #         for batch in tqdm(dataloader):
    #             x, x_mask, _, _ = batch.values()
    #             x, x_mask = x.to(DEVICE), x_mask.to(DEVICE)
    #             tgt = torch.full((batch_size, MAX_SEQ_LEN), self.tokenizer.token_to_id("[CLS]")).to(DEVICE)
    #             tgt_list = [[] for _ in range(batch_size)]
    #             finished = [False] * batch_size
    #             for i in range(MAX_SEQ_LEN - 1):
    #                 # tgt_mask is a batchsize x seq_len tensor where the first i tokens are 1 and the rest are 0
    #                 tgt_mask = (
    #                     torch.tensor([1] * (i + 1) + [0] * (MAX_SEQ_LEN - i - 1)).expand(batch_size, -1).to(DEVICE)
    #                 )
    #                 pred = self.model(x, x_mask, tgt, tgt_mask)
    #                 pred_tokens = pred.argmax(-1)[:, i]
    #                 for j in range(batch_size):
    #                     if not finished[j]:
    #                         tgt[j, i + 1] = pred_tokens[j]
    #                         tgt_list[j].append(pred_tokens[j].item())
    #                         if pred_tokens[j] == self.tokenizer.token_to_id("[SEP]"):
    #                             finished[j] = True
    #                 if all(finished):
    #                     break
    #             hypotheses.extend([self.tokenizer.decode(tgt_list[i]) for i in range(batch_size)])
    #     references = [example["translation"]["tgt"] for example in dataset]
    #     result = BLEU().corpus_score(hypotheses, [references])
    #     return result, references, hypotheses

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
