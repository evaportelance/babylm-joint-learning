import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerFast
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--lm_config_path', default="../babylm-models/test/", type=str)
parser.add_argument('--lm_model_path', default="../babylm-models/baby_small_graminduct527/", type=str)
parser.add_argument('--save_path', default="../babylm-models/submission/", type=str)
parser.add_argument('--checkpoint_name', default="lm_model_best.pth.tar", type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(os.path.join(args.lm_config_path,"config.json"))
    lm = AutoModelForCausalLM.from_config(config)
    checkpoint = torch.load(os.path.join(args.lm_model_path, args.checkpoint_name))
    print("epoch: " + str(checkpoint['epoch']))
    print("batch: " + str(checkpoint['batch']))
    lm.load_state_dict(checkpoint['model_state_dict'])
    lm.save_pretrained(args.lm_model_path)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.lm_config_path, "tokenizer.json"))
    tokenizer.pad_token = '[PAD]'
    tokenizer.eos_token = '[CLS]'
    tokenizer.bos_token = '[SEP]'
    tokenizer.unk_token = '[UNK]'
    tokenizer.save_pretrained(save_path)
