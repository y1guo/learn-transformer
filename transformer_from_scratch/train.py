import torch, json, os
from torch import nn
from dataset import Dataset
from tokenizer import get_tokenizer
from utils import NUM_PROC, DEVICE, log
from model import *
from transformer import Transformer


# load config
with open("config.json") as f:
    config = json.load(f)
    DATASET = config["dataset"]
    LANGUAGE = config["language"]
    VOCAB_SIZE = config["vocab_size"]
    BATCH_SIZE = config["batch_size"]
    MAX_SEQ_LEN = config["max_seq_len"]
    MODEL = config["model"]
    DIM_MODEL = config["models"][MODEL]["dim_model"]
    NUM_HEADS = config["models"][MODEL]["num_heads"]
    NUM_ENCODER_LAYERS = config["models"][MODEL]["num_encoder_layers"]
    NUM_DECODER_LAYERS = config["models"][MODEL]["num_decoder_layers"]
    DIM_FEEDFORWARD = config["models"][MODEL]["dim_feedforward"]
    DROPOUT_RATE = config["models"][MODEL]["dropout_rate"]
    BETA1 = config["beta1"]
    BETA2 = config["beta2"]
    EPSILON = config["epsilon"]
    WARMUP_STEPS = config["warmup_steps"]
    LABEL_SMOOTHING = config["label_smoothing"]
    NUM_LOG_PER_EPOCH = config["num_log_per_epoch"]
print("Number of CPU: ", NUM_PROC, ",\tDevice: ", DEVICE)

# get tokenizer
tokenizer = get_tokenizer(name=DATASET, language=LANGUAGE, vocab_size=VOCAB_SIZE)

# get dataset
print("Loading dataset...")
dataset = Dataset(name=DATASET, language=LANGUAGE, percentage=1)
dataset.tokenize(tokenizer)
dataloader = {}
for split in ["train", "validation"]:
    dataloader[split] = dataset.get_dataloader(split=split, batch_size=BATCH_SIZE, shuffle=False, max_len=MAX_SEQ_LEN)

# create model
model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    d_model=DIM_MODEL,
    nhead=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT_RATE,
    max_seq_len=MAX_SEQ_LEN,
).to(DEVICE)
transformer = Transformer(model, tokenizer)

# load model
prefix = f"{MODEL}_{DATASET}_{LANGUAGE}"
epoch = 0
num_steps_trained = 0
for file in os.listdir("checkpoints"):
    if file.startswith(prefix):
        epoch = max(epoch, int(os.path.splitext(file)[0].split(prefix + "_")[1]))
if epoch == 0:
    print("No checkpoint found. Training from scratch.")
    transformer.save(f"checkpoints/{prefix}_00.pth")
else:
    print(f"Checkpoint found. Resuming from epoch {epoch}.")
    transformer.load(f"checkpoints/{prefix}_{epoch:02d}.pth")
    num_steps_trained = epoch * len(dataloader["train"])

# train
optimizer = torch.optim.Adam(model.parameters(), lr=DIM_MODEL**-0.5, betas=(BETA1, BETA2), eps=EPSILON)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda nstep: min((nstep + num_steps_trained + 1) ** -0.5, (nstep + num_steps_trained + 1) * WARMUP_STEPS**-1.5),
)
loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
update_interval = (len(dataloader["train"]) - 1) // NUM_LOG_PER_EPOCH + 1  # equivalent to ceil
log_file = prefix + ".log"
while True:
    epoch += 1
    log("-------------------------------", log_file)
    log(f"Epoch {epoch}", log_file)
    transformer.train(dataloader["train"], loss_fn, optimizer, scheduler, update_interval, log_file)
    transformer.validate(dataloader["validation"], loss_fn)
    bleu, _, _ = transformer.evaluate_bleu(dataloader["validation"])
    log(f"BLEU score: {bleu.score}", log_file)
    transformer.save(f"checkpoints/{prefix}_{epoch:02d}.pth")
