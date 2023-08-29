import torch, time
from tokenizers import Tokenizer
from utils import DEVICE, log, sec2hms
from colorama import Fore
from model import TransformerModel
from sacrebleu.metrics.bleu import BLEU
from tqdm import tqdm
from datasets.arrow_dataset import Dataset as ArrowDataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from typing import cast


MAX_SEQ_LEN = 128  # need fix


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
            length=MAX_SEQ_LEN,
        )
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)

    def train_epoch(self, dataloader, loss_fn, optimizer, scheduler):
        size = len(dataloader.dataset)
        self.model.train()
        start_time = time.perf_counter()
        for i_batch, batch in enumerate(dataloader):
            x, x_mask, y, y_mask = batch.values()
            x, x_mask, y, y_mask = (
                x.to(DEVICE),
                x_mask.to(DEVICE),
                y.to(DEVICE),
                y_mask.to(DEVICE),
            )

            # Compute prediction error
            optimizer.zero_grad()
            pred = self.model(x, x_mask, y, y_mask)[:, :-1, :]  # (batch_size, seq_len, vocab_size)
            label = y[:, 1:]  # (batch_size, seq_len)
            label_mask = y_mask[:, 1:] == 1  # (batch_size, seq_len)
            loss = loss_fn(pred[label_mask], label[label_mask])

            # Optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i_batch % 100 == 0:
                loss, current = loss.item(), (i_batch + 1) * len(x)
                correct = (pred.argmax(-1) == label)[label_mask].float().sum().item() / label[label_mask].numel()
                elapsed_time = time.perf_counter() - start_time
                remaining_time = elapsed_time * (size - current) / current
                log(
                    f"Accuracy: {100*correct:>4.1f}%, Avg loss: {loss:>10f}, Lr: {scheduler.get_last_lr()[0]:>10f}"
                    f"  [{current:>{len(str(size))}d}/{size}]"
                    f"  [{sec2hms(elapsed_time)} < {sec2hms(remaining_time)}]",
                    "train.log",
                )

    def validate(self, dataloader, loss_fn):
        num_batches = len(dataloader)
        self.model.eval()
        validation_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                x, x_mask, y, y_mask = batch.values()
                x, x_mask, y, y_mask = (
                    x.to(DEVICE),
                    x_mask.to(DEVICE),
                    y.to(DEVICE),
                    y_mask.to(DEVICE),
                )
                label_mask = y_mask[:, 1:] == 1
                pred = self.model(x, x_mask, y, y_mask)[:, :-1, :][label_mask]
                label = y[:, 1:][label_mask]
                validation_loss += loss_fn(pred, label).item()
                correct += (pred.argmax(-1) == label).float().sum().item()
                total += label.numel()
        validation_loss /= num_batches
        correct /= total
        log(
            f"Validation Error: \n Accuracy: {100*correct:>0.1f}%, Avg loss: {validation_loss:>8f} \n",
            "train.log",
        )

    def train(self, dataloader, loss_fn, optimizer, scheduler, epochs=1):
        for i in range(epochs):
            log("-------------------------------", "train.log")
            log(f"Epoch {i+1}/{epochs}", "train.log")
            self.train_epoch(dataloader["train"], loss_fn, optimizer, scheduler)
            self.validate(dataloader["validation"], loss_fn)
            bleu, _, _ = self.evaluate_bleu(dataloader["validation"])
            log(f"BLEU score: {bleu.score}", "train.log")
            torch.save(self.model.state_dict(), f"last_train.pth")
        log("Done!", "train.log")

    def predict(self, source: str, target: str):
        enc = self.tokenizer.encode(f"[CLS] {source} [SEP]")
        src = torch.tensor(enc.ids)[None, :].to(DEVICE)  # (batch_size, seq_len)
        src_mask = torch.tensor(enc.attention_mask)[None, :].to(DEVICE)  # (batch_size, seq_len)
        enc = self.tokenizer.encode(f"[CLS] {target} [SEP]")
        tgt = torch.tensor(enc.ids)[None, :].to(DEVICE)  # (batch_size, seq_len)
        tgt_mask = torch.tensor(enc.attention_mask)[None, :].to(DEVICE)  # (batch_size, seq_len)
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

    def translate(self, source: str):
        enc = self.tokenizer.encode(f"[CLS] {source} [SEP]")
        src = torch.tensor(enc.ids)[None, :].to(DEVICE)  # (batch_size, seq_len)
        src_mask = torch.tensor(enc.attention_mask)[None, :].to(DEVICE)  # (batch_size, seq_len)
        tgt = torch.full((1, MAX_SEQ_LEN), self.tokenizer.token_to_id("[CLS]")).to(DEVICE)  # (batch_size, seq_len)
        tgt_mask = torch.triu(torch.full((MAX_SEQ_LEN, MAX_SEQ_LEN), 1)).T.to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            for i in range(MAX_SEQ_LEN - 1):
                pred = self.model(src, src_mask, tgt, tgt_mask[i : i + 1, :])  # (batch_size, seq_len, vocab_size)
                pred_token = pred.argmax(-1)[0, i]
                tgt[0, i + 1] = pred_token
                if pred_token == self.tokenizer.token_to_id("[SEP]"):
                    break
        return self.tokenizer.decode(tgt[0].tolist())

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
        self.model.eval()
        with torch.no_grad():
            references, hypotheses = [], []
            for batch in tqdm(dataloader):
                x, x_mask, y, y_mask = batch.values()
                x, x_mask = x.to(DEVICE), x_mask.to(DEVICE)
                batch_size = len(x)
                ref_list = [list(label[mask]) for label, mask in zip(y, y_mask == 1)]
                tgt = torch.full((batch_size, MAX_SEQ_LEN), self.tokenizer.token_to_id("[CLS]")).to(DEVICE)
                tgt_list = [[] for _ in range(batch_size)]
                finished = [False] * batch_size
                for i in range(MAX_SEQ_LEN - 1):
                    # tgt_mask is a batchsize x seq_len tensor where the first i tokens are 1 and the rest are 0
                    tgt_mask = (
                        torch.tensor([1] * (i + 1) + [0] * (MAX_SEQ_LEN - i - 1)).expand(batch_size, -1).to(DEVICE)
                    )
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
