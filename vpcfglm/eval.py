from .eva_metric import AverageMeter, LogCollector
import time
import numpy as np
import argparse
from collections import defaultdict

import torch
from torch_struct import SentCFG

from . import Utils
from .Utils import Vocabulary
from .Modules import CompoundCFG


def build_parse(spans, caption, vocab):
    tree = [(i, vocab.idx2word[int(word)]) for i, word in enumerate(caption)]
    tree = dict(tree)
    for l, r, A in spans:
        if l != r:
            span = '({} {})'.format(tree[l], tree[r])
            tree[r] = tree[l] = span
    return tree[0]


def make_model(best_model, args):
    model = CompoundCFG(
        args.vocab_size, args.nt_states, args.t_states,
        h_dim=args.h_dim,
        w_dim=args.w_dim,
        z_dim=args.z_dim,
        s_dim=args.state_dim
    )
    best_model = best_model['parser']
    model.load_state_dict(best_model)
    return model

'''
def eval_trees(args):
    checkpoint = torch.load(args.model, map_location='cpu')
    opt = checkpoint['opt']
    use_mean = True
    # load vocabulary used by the model
    data_path = args.data_path
    # data_path = getattr(opt, "data_path", args.data_path)
    vocab_name = getattr(opt, "vocab_name", args.vocab_name)
    vocab = pickle.load(open(os.path.join(data_path, vocab_name), 'rb'))
    checkpoint['word2idx'] = vocab.word2idx
    pps['vocab_size'] = len(vocab)

    parser = checkpoint['model']
    parser = make_model(parser, opt)
    parser.cuda()
    parser.eval()

    batch_size = 5
    prefix = args.prefix
    print('Loading dataset', data_path + prefix + args.split)
    data_loader = data.eval_data_iter(
        data_path, prefix + args.split, vocab, batch_size=batch_size)

    # stats
    trees = list()
    n_word, n_sent = 0, 0
    per_label_f1 = defaultdict(list)
    by_length_f1 = defaultdict(list)
    sent_f1, corpus_f1 = [], [0., 0., 0.]
    total_ll, total_kl, total_bc, total_h = 0., 0., 0., 0.

    pred_out = open(args.out_file, "w")

    for i, (captions, lengths, spans, labels, tags, ids) in enumerate(data_loader):
        lengths = torch.tensor(lengths).long() if isinstance(
            lengths, list) else lengths
        if torch.cuda.is_available():
            lengths = lengths.cuda()
            captions = captions.cuda()

        params, kl = parser(captions, lengths, use_mean=use_mean)
        dist = SentCFG(params, lengths=lengths)

        arg_spans = dist.argmax[-1]
        argmax_spans, _, _ = Utils.extract_parses(
            arg_spans, lengths.tolist(), inc=0)

        candidate_trees = list()
        bsize = captions.shape[0]
        n_word += (lengths + 1).sum().item()
        n_sent += bsize

        for b in range(bsize):
            max_len = lengths[b].item()
            pred = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
            pred_set = set(pred[:-1])
            gold = [(l, r) for l, r in spans[b] if l != r]
            gold_set = set(gold[:-1])

            ccaption = captions[b].tolist()[:max_len]
            sent = [vocab.idx2word[int(word)]
                    for _, word in enumerate(ccaption)]
            iitem = (sent, gold, labels, pred)
            json.dump(iitem, pred_out)
            pred_out.write("\n")

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

            word_tree = build_parse(
                argmax_spans[b], captions[b].tolist(), vocab)
            candidate_trees.append(word_tree)

            for j, gold_span in enumerate(gold[:-1]):
                label = labels[b][j]
                label = re.split("=|-", label)[0]
                per_label_f1.setdefault(label, [0., 0.])
                per_label_f1[label][0] += 1

                lspan = gold_span[1] - gold_span[0] + 1
                by_length_f1.setdefault(lspan, [0., 0.])
                by_length_f1[lspan][0] += 1

                if gold_span in pred_set:
                    per_label_f1[label][1] += 1
                    by_length_f1[lspan][1] += 1

        appended_trees = ['' for _ in range(len(ids))]
        for j in range(len(ids)):
            tree = candidate_trees[j]
            appended_trees[ids[j] - min(ids)] = tree
        for tree in appended_trees:
            # print(tree)
            pass
        trees.extend(appended_trees)
        # if i == 50: break

    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + \
        recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word)
    kl = total_kl / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, corpus_f1 * 100, sent_f1 * 100
    )
    print(info)

    f1_ids = ["CF1", "SF1", "NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]

    f1s = {"CF1": corpus_f1, "SF1": sent_f1}

    print("\nPER-LABEL-F1 (label, acc)\n")
    for k, v in per_label_f1.items():
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        f1s[k] = v[1] / v[0]

    f1s = ['{:.2f}'.format(float(f1s[x]) * 100) for x in f1_ids]
    print("\t".join(f1_ids))
    print("\t".join(f1s))

    acc = []

    print("\nPER-LENGTH-F1 (length, acc)\n")
    xx = sorted(list(by_length_f1.items()), key=lambda x: x[0])
    for k, v in xx:
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        if v[0] >= 5:
            acc.append((str(k), '{:.2f}'.format(v[1] / v[0])))
    k = [x for x, _ in acc]
    v = [x for _, x in acc]
    print("\t".join(k))
    print("\t".join(v))

    pred_out.close()
    return trees
'''


def encode_data(model, data_loader, log_step=10, logging=print, vocab=None):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    model.eval()
    end = time.time()

    n_word, n_sent = 0, 0
    total_ll, total_kl = 0., 0.
    sent_f1, corpus_f1 = [], [0., 0., 0.]

    img_embs = None
    cap_embs = None
    for i, (images, captions, lengths, ids, spans) in enumerate(data_loader):
        model.logger = val_logger
        lengths = torch.tensor(lengths).long() if isinstance(
            lengths, list) else lengths

        bsize = captions.size(0)
        img_emb, cap_span_features, nll, kl, span_margs, argmax_spans, trees, lprobs = \
            model.forward_encoder(
                images, captions, lengths, spans, require_grad=False
            )
        mstep = (lengths * (lengths - 1) / 2).int()  # (b, NT, dim)
        cap_feats = torch.cat(
            [cap_span_features[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0
        )
        span_marg = torch.softmax(
            torch.cat([span_margs[j][k - 1].unsqueeze(0)
                      for j, k in enumerate(mstep)], dim=0), -1
        )
        cap_emb = torch.bmm(span_marg.unsqueeze(-2),  cap_feats).squeeze(-2)
        cap_emb = Utils.l2norm(cap_emb)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        total_ll += nll.sum().item()
        total_kl += kl.sum().item()
        n_word += (lengths + 1).sum().item()
        n_sent += bsize

        bsize = img_emb.shape[0]
        for b in range(bsize):
            max_len = lengths[b].item()
            pred = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
            pred_set = set(pred[:-1])
            gold = [(spans[b][i][0].item(), spans[b][i][1].item())
                    for i in range(max_len - 1)]
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

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions
        # if i >= 50: break

    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + \
        recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word)
    kl = total_kl / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, corpus_f1 * 100, sent_f1 * 100
    )
    logging(info)
    return img_embs, cap_embs, ppl_elbo, sent_f1 * 100


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # compute scores
        d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def validate(pps, val_loader, model, vocab, logger):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, val_ppl, val_f1 = encode_data(
        model, val_loader, pps['log_step'], logger.info, vocab)
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure='cosine')
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, measure='cosine')
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    return val_ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--split', type=str, default='test_gold')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--data_path', type=str, default='./mscoco')
    parser.add_argument('--vocab_name', type=str, default='coco.dict.pkl')
    parser.add_argument('--out_file', default='pred-parse.txt')
    parser.add_argument('--gold_out_file', default='gold-parse.txt')
    args = parser.parse_args()
    # eval_trees(args)
