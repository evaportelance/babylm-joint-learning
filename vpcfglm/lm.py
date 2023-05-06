# LM objective

from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
from rouge import Rouge
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel
import datasets
import torch
from datetime import datetime


def tokenization(example):
    return tokenizer(example["text"], truncation=True, max_length=pps.max_length)


def load_datasets(mode):
    # load datasets
    if mode == 'train':
        with open(pps.train_data) as f:
            lines = f.readlines()
            if pps.mode == 'test':
                lines = lines[0:200]
        train_dict = {'text': [sent.strip() for sent in lines if sent.strip()!='']}
        train_data = datasets.Dataset.from_dict(train_dict)
        train_data = train_data.map(tokenization, batched=True)
        TrainDataset = CustomDataset(train_data['input_ids'])
        torch.save(TrainDataset, './TrainDataset.pt')

    if mode == 'valid':
        with open(pps.valid_data) as f:
            lines = f.readlines()
        valid_dict = {'text': [sent.strip() for sent in lines if sent.strip()!='']}
        valid_data = datasets.Dataset.from_dict(valid_dict)
        valid_data = valid_data.map(tokenization, batched=True)
        ValidDataset = CustomDataset(valid_data['input_ids'])
        torch.save(ValidDataset, './ValidDataset.pt')


class CustomDataset(Dataset):
    def __init__(self, pt_data_dict):
        self.input = pt_data_dict

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        source = self.input[index][0:-1]
        target = self.input[index][1:]

        return source, target


def collate_fn(batch):

    sent_lens = [len(x[0]) for x in batch]  # x0 = source
    source = []
    target = []
    max_input_len = max(sent_lens)
    for idx in range(len(batch)):
        input_len = len(batch[idx][0])
        x_source = list(batch[idx][0]) + [2] * (max_input_len - input_len) # eos/pad = 2
        x_target = list(batch[idx][1]) + [2] * (max_input_len - input_len) # eos/pad = 2
        source.append(x_source)
        target.append(x_target)

    return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class LanguageModel(PreTrainedModel):
    def __init__(self, config, model, V=50265, hdim=64):
        super().__init__(config)
        self.model = model
        self.model.transformer.wte = torch.nn.Embedding(V, hdim)

    def forward(self, source, target):

        inputs = {"input_ids": source}
        outputs = self.model(**inputs, labels=target)

        return outputs


def rouge(not_ignore, shift_labels, preds):
    main_rouge = Rouge()
    true_length = [w.sum() for w in not_ignore.float()]
    rouge_labels = []
    rouge_predicts = []
    for idx, tmp_len in enumerate(true_length):
        tmp_labels = shift_labels[idx][:int(tmp_len)]
        rouge_labels.append(" ".join([str(w) for w in tmp_labels.tolist()]))
        tmp_pred = preds[idx][:int(tmp_len)]
        rouge_predicts.append(" ".join([str(w) for w in tmp_pred.tolist()]))
    rouge_score = main_rouge.get_scores(rouge_predicts, rouge_labels, avg=True)
    return rouge_score


def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=2, reduction='sum') # value = tokenizer.eos_token_id = 2
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(2)  # value = tokenizer.pad_token_id = 2
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    rouge_score = rouge(not_ignore, shift_labels, preds)
    return loss, accuracy, rouge_score


def train(model, dataloader):
    num_training_steps = pps.epochs * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=pps.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=pps.warmup_steps,
        num_training_steps=num_training_steps
    )
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    batch_steps = 0
    for epoch in range(pps.epochs):
        for batch in dataloader:
            batch_steps += 1
            source, target = batch
            source = source.to(device)
            target = target.to(device)
            outputs = model(source, target)
            loss, acc, rouge_score = calculate_loss_and_accuracy(
                outputs, source.to(device), device)
            # print(loss, acc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), pps.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if batch_steps % pps.log_steps == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}, rouge-1 {}, rouge-2 {}, rouge-l {}".format(
                    epoch, pps.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc,
                    rouge_score["rouge-1"]['f'],
                    rouge_score["rouge-2"]["f"],
                    rouge_score["rouge-l"]["f"]))


class Parameters():

    def __init__(self) -> None:

        self.vocab_size = 8192
        self.batch_size = 64
        self.embed_dim = 256 # 400 in the paper == h_dim
        self.h_dim = 256 # 1150 in the paper
        self.n_ctx = 64
        self.max_length = 128
        self.n_head = 8
        self.n_layer = 3
        self.dropout_rate = 0.2
        self.tie_weights = True
        self.lr = 1e-4
        self.epochs = 10
        self.warmup_steps = 2000
        self.max_grad_norm = 1.0
        self.log_steps = 2000
        self.mode = 'train'#'train' #'test'
        self.train_data = '/network/scratch/x/xuanda.chen/babylm_data/train.txt' 
        self.valid_data = '/network/scratch/x/xuanda.chen/babylm_data/dev.txt'
        self.tokenizer_path = '/network/scratch/x/xuanda.chen/baby_model'
        self.save_model_path = './models' 


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    # load hyperparameters
    pps = Parameters()
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pps.tokenizer_path)
    # tokenizer.pad_token = tokenizer.eos_token

    # load model
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=pps.vocab_size,
        n_ctx=pps.n_ctx,
        bos_token_id=0,
        eos_token_id=2,
        n_embd=pps.embed_dim,
        n_head=pps.n_head,
        n_layer=pps.n_layer,
        n_positions=pps.max_length,
    )

    gpt = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in gpt.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    # customize model
    lm = LanguageModel(config, gpt, V=pps.vocab_size, hdim=pps.h_dim)
    print(lm)
    
    # load a save the dataset for reuse
    # load_datasets(mode='train')

    TrainDataset = torch.load('./TrainDataset.pt')
    TrainLoader = DataLoader(
    TrainDataset, batch_size=pps.batch_size, shuffle=True, collate_fn=collate_fn)
    print('Train data loaded ...')

    # import sys
    # sys.exit(1)
    # ValidDataset = torch.load('./ValidDataset.pt')
    # ValidLoader = DataLoader(
    # ValidDataset, batch_size=pps.batch_size, shuffle=False, collate_fn=collate_fn)
    # print('Valid data loaded ...')


    # Training Procedure
    num_training_steps = pps.epochs * len(TrainLoader)
    optimizer = AdamW(lm.parameters(), lr=pps.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=pps.warmup_steps,
        num_training_steps=num_training_steps
    )

    lm.to(device)
    lm.train()
    batch_steps = 0
    for epoch in range(pps.epochs):
        for batch in TrainLoader:
            batch_steps += 1
            source, target = batch
            source = source.to(device)
            target = target.to(device)
            outputs = lm(source, target)
            loss, acc, rouge_score = calculate_loss_and_accuracy(
                outputs, source.to(device), device)
            # print(loss, acc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                lm.parameters(), pps.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if batch_steps % pps.log_steps == 0:
                current_time = datetime.now()
                print("Current Time:", current_time)
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}, rouge-1 {}, rouge-2 {}, rouge-l {}".format(
                    epoch, pps.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc,
                    rouge_score["rouge-1"]['f'],
                    rouge_score["rouge-2"]["f"],
                    rouge_score["rouge-l"]["f"]))

    model_to_save = lm.module if hasattr(lm, 'module') else lm
    model_to_save.save_pretrained(pps.save_model_path)
