import os
import sys
import time, pickle, argparse, logging
import numpy as np
import torch
from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm.auto import tqdm
import datasets
from torch.utils.data import DataLoader
from vpcfg.model import VGCPCFGs


def load_datasets(opt, tokenizer):    
    def tokenization(batch):
        return tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')        
    with open(opt.train_data) as f:
        lines = f.readlines()
        if opt.tiny :
            lines = lines[0:200]
    train_dict = {'text': [sent.strip() for sent in lines if len(sent.strip()) > 1]}
    train_data = datasets.Dataset.from_dict(train_dict)
    train_loader = DataLoader(train_data['text'], batch_size=opt.batch_size, shuffle = True, collate_fn=tokenization)
    with open(opt.val_data) as f:
        lines = f.readlines()
    if opt.tiny :
            lines = lines[0:200]
    val_dict = {'text': [sent.strip() for sent in lines if len(sent.strip()) > 1]}
    val_data = datasets.Dataset.from_dict(val_dict)
    val_loader = DataLoader(val_data['text'], batch_size=opt.batch_size, shuffle = True, collate_fn=tokenization)
    return train_loader, val_loader


def get_val_accuracy(model, val_dataloader, device, opt):
    model.eval()
    total_correct = 0
    total_targets = 0
    for i, batch in enumerate(val_dataloader):
        if i > opt.val_step:
            break
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels=input_ids
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(1)  # value = tokenizer.pad_token_id = 1
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum()
        total_correct += correct
        total_targets += num_targets
    return total_correct/total_targets
    
def train(model, train_dataloader, val_dataloader, logger):
    num_training_steps = opt.num_epochs * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=opt.lr)
    loss_fct = CrossEntropyLoss(ignore_index=3) # value = tokenizer.eos_token_id = 3
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=opt.warmup_steps,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    prev_val_accuracy = 0
    for epoch in range(opt.num_epochs):
        model.train()
        total_correct = 0
        total_targets = 0
        total_loss = 0
        for i, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels=input_ids
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            _, preds = shift_logits.max(dim=-1)
            not_ignore = shift_labels.ne(1)  # value = tokenizer.pad_token_id = 1
            num_targets = not_ignore.long().sum().item()
            correct = (shift_labels == preds) & not_ignore
            correct = correct.float().sum()
            total_correct += correct
            total_targets += num_targets
            total_loss += float(loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if i % opt.log_step == 0:
                batch_accuracy = correct / num_targets
                epoch_accuracy = total_correct/total_targets
                val_accuracy = get_val_accuracy(model, val_dataloader, device, opt)
                logger.info("train epoch {}/{}, batch {}/{}, batch loss {}, batch accuracy {}, epoch accuracy {}, val accuracy {}".format(
                    epoch, opt.num_epochs,
                    i,
                    num_training_steps,
                    loss, batch_accuracy,
                    epoch_accuracy,
                    val_accuracy))
                if val_accuracy > prev_val_accuracy:
                    prev_val_accuracy = val_accuracy
                    filename = os.path.join(opt.save_model_path, "lm_model_best.pth.tar")
                    state = {'epoch': epoch,
                        'batch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': val_accuracy,
                        'train_accuracy': epoch_accuracy,
                        'opt': opt}
                    torch.save(state, filename)
        epoch_accuracy = total_correct/total_targets
        average_loss = total_loss/num_training_steps
        val_accuracy = get_val_accuracy(model, val_dataloader, device, opt)
        logger.info("END OF EPOCH {}: average loss {} - train accuracy {} - val accuracy {}".format(
                    epoch,
                    average_loss,
                    epoch_accuracy,
                    val_accuracy))
        checkpoint_name = str(epoch) + "_lm_checkpoint.pth.tar"
        filename = os.path.join(opt.save_model_path, checkpoint_name)
        state = {'epoch': epoch,
            'batch': -1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'train_accuracy': epoch_accuracy,
            'opt': opt}
        torch.save(state, filename)
        if val_accuracy > prev_val_accuracy:
            prev_val_accuracy = val_accuracy
            filename = os.path.join(opt.save_model_path, "lm_model_best.pth.tar")
            state = {'epoch': epoch,
                'batch':-1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_accuracy': epoch_accuracy,
                'opt': opt}
            torch.save(state, filename)
        

if __name__ == '__main__':
    # hyper parameters
    parser = argparse.ArgumentParser()

    # Parser: Generative model parameters
    parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
    parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
    parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
    parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
    # Parser: Inference network parameters
    parser.add_argument('--h_dim', default=768, type=int, help='hidden dim for variational LSTM')
    parser.add_argument('--w_dim', default=768, type=int, help='embedding dim for variational LSTM')
    parser.add_argument('--gpu', default=1, type=int, help='which gpu to use')
    parser.add_argument('--sem_dim', default=768, type=int, help='semantic rep. dim')
    parser.add_argument('--syn_dim', default=768, type=int, help='syntactic rep. dim')
    parser.add_argument('--word_dim', default=768, type=int,
                        help='dimensionality of the word embedding')
    parser.add_argument('--lstm_dim', default=768, type=int,
                        help='dimensionality of the lstm hidden embedding')
    #
    parser.add_argument('--prefix', default="all", type=str, help='prefix')
    parser.add_argument('--visual_mode', default=False, type=bool, help='run visual model')
    parser.add_argument('--tiny', action='store_true', help='if testing will create tiny dataloaders')
    parser.add_argument('--shuffle', action='store_true', help='shuffle training data')
    parser.add_argument('--seed', default=1213, type=int, help='random seed')
    parser.add_argument('--model_init', default='../../babylm-models/as_graminduct/outputs/model_best.pth.tar', type=str, help='random seed')
    parser.add_argument('--max_length', default=128, type=int, help='vocab name')
    parser.add_argument('--vocab_size', default=10000, type=int,
                        help='tokenizer/vocabulary size')
    #
    parser.add_argument('--val_data', default='../../data/all_dev_data.txt', help='path to validation data file')
    parser.add_argument('--train_data', default='../../data/noas_all_train_data.txt', help='path to train data file')
    parser.add_argument('--tokenizer_path', default='../../babylm-models/test/', help='path to pretrained tokenizer')
    parser.add_argument('--lm_config_path', default='../../babylm-models/test/', help='path to config file for lm')
    parser.add_argument('--save_model_path', default='../../babylm-models/as_graminduct/', help='path to directory with model configs')
    parser.add_argument('--logger_name', default='../../babylm-models/as_graminduct/', help='location for model outputs and logfiles to be saved')
    #
    parser.add_argument('--margin', default=0.2, type=float,
                        help='rank loss margin')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='number of training epochs')
    parser.add_argument('--batch_size', default=5, type=int,
                        help='size of a training mini-batch')
    parser.add_argument('--grad_clip', default=3., type=float,
                        help='gradient clipping threshold')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=float,
                        help='warmup steps for scheduler')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='max grad norm for gradient clipping')
#    parser.add_argument('--workers', default=0, type=int, help='number of data loader workers')
    #
    parser.add_argument('--log_step', default=2000, type=int,
                        help='number of steps to print and record the log')
    parser.add_argument('--val_step', default=5000, type=int,
                        help='number of steps to run validation')
    #
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='dimensionality of the image embedding')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    #
#    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer, can be Adam, SGD, etc.')
    parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
    #
    parser.add_argument('--vse_mt_alpha', type=float, default=0.01)
    parser.add_argument('--vse_lm_alpha', type=float, default=1.0)

    opt = parser.parse_args()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # setup logger
    if os.path.exists(opt.logger_name):
        print(f'Warning: the folder {opt.logger_name} exists.')
    else:
        print('Creating {}'.format(opt.logger_name))
        os.mkdir(opt.logger_name)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(opt.logger_name, 'train.log'), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info('cuda:{}@{}'.format(opt.gpu, os.uname().nodename))
    logger.info(opt)
    
    # Initialize LM and Tokenizer
    config = AutoConfig.from_pretrained(os.path.join(opt.lm_config_path,"config.json"))
    lm = AutoModelForCausalLM.from_config(config)
    subword_embeds = lm.model.decoder.embed_tokens
    
    tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer_path)
    tokenizer.pad_token = '[PAD]'
    
    # Get pretrained grammar induction model embeddings
    model = VGCPCFGs(opt, logger, subword_embeds, tokenizer)
    checkpoint = torch.load(opt.model_init)
    parser_params = checkpoint['model'][VGCPCFGs.NS_PARSER]
    model.parser.load_state_dict(parser_params)
    pretrained_embed = model.parser.enc_emb
    #lm.model.decoder.embed_tokens = pretrained_embed
    
    # Load data and create data loaders
    train_loader, val_loader = load_datasets(opt, tokenizer)
    
    torch.cuda.empty_cache()
    # Train language model
    train(lm, train_loader, val_loader, logger)
