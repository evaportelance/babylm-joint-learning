import time
import transformers
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from eva_metric import calculate_loss_and_accuracy, get_lm_info


class LanguageModel(transformers.PreTrainedModel):
    def __init__(self, config, lm):
        super().__init__(config)
        
        self.model = lm

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.model.decoder.embed_tokens.weight.requires_grad = True


    def forward(self, source, target):
        inputs = {"input_ids": source, "labels": target}
        outputs = self.model(**inputs)

        return outputs
    
# "babylm/opt-125m-strict"

class JointModel(torch.nn.Module):
    def __init__(self, plm, cpcfg):
        super().__init__()
        self.plm = plm
        self.cpcfg = cpcfg

    def forward(self, image, caption, lengths, spans, epoch, lm_source, lm_target, mapping, device, mode='joint'):
        # To device
        if torch.cuda.is_available():
            image = image.cuda()
            caption = caption.cuda()
            lengths = lengths.cuda()
            spans = spans.cuda()
            lm_source = lm_source.cuda()
            lm_target = lm_target.cuda()

        # LM objective
        if mode == 'lm':
            outputs = self.plm(lm_source, lm_target)

            lm_loss, lm_acc, rouge_score = calculate_loss_and_accuracy(outputs, lm_source, device)
            lm_info = get_lm_info(epoch, lm_loss, lm_acc, rouge_score)
            gi_loss, gi_info = 0, 0

        # GI objective
        if mode == 'gi':
            # lm_source = lm_source[:,1:]
            gi_loss, gi_info = self.cpcfg(image, caption, lengths, spans, epoch, lm_source, mapping)
            lm_loss, lm_info = 0, 0

        if mode == 'joint':
            outputs = self.plm(lm_source, lm_target)

            lm_loss, lm_acc, rouge_score = calculate_loss_and_accuracy(outputs, lm_source, device)
            lm_info = get_lm_info(epoch, lm_loss, lm_acc, rouge_score) 

            gi_loss, gi_info = self.cpcfg(image, caption, lengths, spans, epoch, lm_source, mapping)           
  
        return lm_loss, gi_loss, lm_info, gi_info


