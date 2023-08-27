import torch, time
from tokenizers import Tokenizer
from utils import DEVICE, log, sec2hms
from colorama import Fore
from model import TransformerModel


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

    def train_epoch(self, dataloader, model, loss_fn, optimizer, scheduler):
        size = len(dataloader.dataset)
        model.train()
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
            pred = model(x, x_mask, y, y_mask)[
                :, :-1, :
            ]  # (batch_size, seq_len, vocab_size)
            label = y[:, 1:]  # (batch_size, seq_len)
            label_mask = y_mask[:, 1:] == False  # (batch_size, seq_len)
            loss = loss_fn(pred[label_mask], label[label_mask])

            # Optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i_batch % 100 == 0:
                loss, current = loss.item(), (i_batch + 1) * len(x)
                correct = (pred.argmax(-1) == label)[
                    label_mask
                ].float().sum().item() / label[label_mask].numel()
                elapsed_time = time.perf_counter() - start_time
                remaining_time = elapsed_time * (size - current) / current
                log(
                    f"Accuracy: {100*correct:>0.1f}%, Avg loss: {loss:>7f}  [{current:>{len(str(size))}d}/{size}]"
                    f"  [{sec2hms(elapsed_time)} < {sec2hms(remaining_time)}]",
                    "train.log",
                )

    def validate(self, dataloader, model, loss_fn):
        num_batches = len(dataloader)
        # model.eval() # Note: this triggers some optimized behavior which causes trouble with our setup
        model.train()
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
                pred = model(x, x_mask, y, y_mask)[:, :-1, :]
                label = y[:, 1:]
                label_mask = y_mask[:, 1:] == False
                validation_loss += loss_fn(pred[label_mask], label[label_mask]).item()
                correct += (pred.argmax(-1) == label)[label_mask].float().sum().item()
                total += label[label_mask].numel()
        validation_loss /= num_batches
        correct /= total
        log(
            f"Validation Error: \n Accuracy: {100*correct:>0.1f}%, Avg loss: {validation_loss:>8f} \n",
            "train.log",
        )

    def train(self, dataloader, model, loss_fn, optimizer, scheduler, epochs=1):
        for i in range(epochs):
            log("-------------------------------", "train.log")
            log(f"Epoch {i+1}/{epochs}", "train.log")
            self.train_epoch(dataloader["train"], model, loss_fn, optimizer, scheduler)
            self.validate(dataloader["validation"], model, loss_fn)
            torch.save(model.state_dict(), f"model_{i+1}.pth")
        log("Done!", "train.log")

    def predict(self, source: str, target: str):
        enc = self.tokenizer.encode("[CLS]" + source + "[SEP]")
        src = torch.tensor(enc.ids)[None, :].to(DEVICE)  # (batch_size, seq_len)
        src_mask = (
            torch.tensor(enc.attention_mask)[None, :].to(DEVICE) == 0
        )  # (batch_size, seq_len)
        enc = self.tokenizer.encode("[CLS]" + target + "[SEP]")
        tgt = torch.tensor(enc.ids)[None, :].to(DEVICE)  # (batch_size, seq_len)
        tgt_mask = (
            torch.tensor(enc.attention_mask)[None, :].to(DEVICE) == 0
        )  # (batch_size, seq_len)
        with torch.no_grad():
            pred = self.model(
                src, src_mask, tgt, tgt_mask
            )  # (batch_size, seq_len, vocab_size)
            pred_tokens = pred[0, :-1, :].argmax(-1)  # (seq_len)
            label_tokens = tgt[0, 1:]  # (seq_len)
            label_mask = tgt_mask[0, 1:] == False  # (seq_len)
            correct = (pred_tokens == label_tokens)[
                label_mask
            ].float().sum().item() / label_tokens[label_mask].numel()
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
        enc = self.tokenizer.encode("[CLS]" + source + "[SEP]")
        src = torch.tensor(enc.ids)[None, :].to(DEVICE)  # (batch_size, seq_len)
        src_mask = (
            torch.tensor(enc.attention_mask)[None, :].to(DEVICE) == 0
        )  # (batch_size, seq_len)
        tgt = torch.full((1, MAX_SEQ_LEN), self.tokenizer.token_to_id("[CLS]")).to(
            DEVICE
        )  # (batch_size, seq_len)
        tgt_mask = torch.triu(
            torch.full((MAX_SEQ_LEN, MAX_SEQ_LEN), True), diagonal=1
        ).to(DEVICE)
        with torch.no_grad():
            for i in range(MAX_SEQ_LEN - 1):
                pred = self.model(
                    src, src_mask, tgt, tgt_mask[i : i + 1, :]
                )  # (batch_size, seq_len, vocab_size)
                pred_token = pred.argmax(-1)[0, i]
                tgt[0, i + 1] = pred_token
                if pred_token == self.tokenizer.token_to_id("[SEP]"):
                    break
        return self.tokenizer.decode(tgt[0].tolist())
