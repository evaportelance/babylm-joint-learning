import time
import numpy as np
from collections import OrderedDict
import torch
import Utils
from rouge import Rouge

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

def validate_parser(opt, data_loader, model, vocab, logger, visual_mode):
    # if visual_mode:
        # import eval_vis
        # return eval_vis.validate(opt, data_loader, model, vocab, logger)
    batch_time = AverageMeter()
    val_logger = LogCollector()

    model.eval()
    end = time.time()
    nbatch = len(data_loader)

    n_word, n_sent = 0, 0
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    total_ll, total_kl = 0., 0.

    for i, (images, captions, lengths, spans, lm_source, lm_target, mapping) in enumerate(data_loader):
        model.logger = val_logger
        # if torch.cuda.is_available():
        #     if isinstance(lengths, list):
        #         lengths = torch.tensor(lengths).long()
        if torch.cuda.is_available():
            captions = captions.cuda()
            lengths = lengths.cuda()
            lm_source = lm_source.cuda()
        bsize = captions.size(0) 

        nll, kl, argmax_spans, trees, lprobs = model.forward_parser(captions, lengths, lm_source, mapping)

        batch_time.update(time.time() - end)
        end = time.time()

        total_ll += nll.sum().item()
        total_kl += kl.sum().item()
        n_word += (lengths + 1).sum().item()
        n_sent += bsize

        for b in range(bsize):
            max_len = lengths[b].item() 
            pred = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
            pred_set = set(pred[:-1])
            gold = [(spans[b][i][0].item(), spans[b][i][1].item()) for i in range(max_len - 1)] 
            gold_set = set(gold[:-1])

            tp, fp, fn = Utils.get_stats(pred_set, gold_set) 
            corpus_f1[0] += tp
            corpus_f1[1] += fp
            corpus_f1[2] += fn
            
            overlap = pred_set.intersection(gold_set)
            prec = float(len(overlap)) / (len(pred_set) + 1e-8)
            reca = float(len(overlap)) / (len(gold_set) + 1e-8)
            
            if len(gold_set) == 0:
                reca = 1. 
                if len(pred_set) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            sent_f1.append(f1)

        if i % model.log_steps == 0:
            logger.info(
                'Test: [{0}/{1}]\t{e_log}\t'
                .format(
                    i, nbatch, e_log=str(model.logger)
                )
            )
        del captions, lengths, spans


    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word) 
    kl = total_kl / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, corpus_f1 * 100, sent_f1 * 100
    )
    logger.info(info)
    return ppl_elbo 


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
    # shift_logits = logits[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous().to(device)

    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous().to(device)

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=1, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(1) # value = tokenizer.pad_token_id
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    rouge_score = rouge(not_ignore, shift_labels, preds)
    return loss, accuracy, rouge_score


def get_lm_info(epoch, loss, acc, rouge_score):
    return "train epoch {}/{}, batch {}/{}, loss {}, accuracy {}, rouge-1 {}, rouge-2 {}, rouge-l {}".format(
            epoch, 10,
            0,
            0,
            loss, acc,
            rouge_score["rouge-1"]['f'],
            rouge_score["rouge-2"]["f"],
            rouge_score["rouge-l"]["f"])