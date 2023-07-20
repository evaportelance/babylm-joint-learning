import os, re, json
import numpy as np
import random
import torch
import torch.utils.data as data
from operator import itemgetter

TXT_IMG_DIVISOR=1
TXT_MAX_LENGTH=45

def set_constant(visual_mode, max_length):
    global TXT_IMG_DIVISOR, TXT_MAX_LENGTH
    TXT_IMG_DIVISOR = 1 if not visual_mode else 5
    TXT_MAX_LENGTH = max_length
    #print(TXT_IMG_DIVISOR, TXT_MAX_LENGTH)

def set_rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_number(w):
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w

class SortedBlockSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        all_sample = len(self.data_source)
        batch_size = data_source.batch_size
        nblock = all_sample // batch_size 
        residue = all_sample % batch_size
        nsample = all_sample - residue
        # https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        # it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
        self.groups = np.array_split(range(nsample), nblock)
        self.strip_last = False
        if residue > 0:
            self.strip_last = True
            block = np.array(range(nsample, all_sample))
            self.groups.append(block)

    def __iter__(self):
        self.data_source._shuffle()
        end = -1 if self.strip_last else len(self.groups)
        groups = self.groups[:end]
        #random.shuffle(groups) 
        indice = torch.randperm(len(groups)).tolist() 
        groups = [groups[k] for k in indice]
        if self.strip_last:
            groups.append(self.groups[-1])
        indice = list()
        for i, group in enumerate(groups):
            indice.extend(group)
            #print(i, group)
        assert len(indice) == len(self.data_source)
        return iter(indice)

    def __len__(self):
        return len(self.data_source)

class SortedRandomSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)

class SortedSequentialSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class DataLoader(data.Dataset):
    def __init__(self, data_path, data_split, tokenizer,
                 load_img=True, img_dim=2048, batch_size=1, tiny=False):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.ids_captions_spans = list()
        max_length = TXT_MAX_LENGTH
        indexes, removed, idx = list(), list(), -1
        try:
            with open(os.path.join(data_path, f'{data_split}_caps.json'), 'r') as f1, open(os.path.join(data_path, f'{data_split}.id'), 'r') as f2:
                for line, img_id in zip(f1.readlines(), f2.readlines()):
                    if tiny and idx > 32 :
                        break
                    idx += 1
                    (caption, span) = json.loads(line)
                    caption = [clean_number(w) for w in caption.strip().lower().split()]
                    caption_new, span_new, input_ids = self.ajust_tokens_spans(caption, span)
                    if TXT_MAX_LENGTH < 1000 and (len(caption_new) < 2 or len(caption_new) > max_length):
                        removed.append((idx, len(caption_new)))
                        continue
                    self.ids_captions_spans.append({'img_id':int(img_id), 'caption': caption_new, 'span':span_new, 'input_ids': input_ids})
                    indexes.append(idx)
        except:
            with open(os.path.join(data_path, f'{data_split}_caps.json'), 'r') as f1:
                for line in f1.readlines():
                    img_id = 0
                    if tiny and idx > 32 :
                        break
                    idx += 1
                    (caption, span) = json.loads(line)
                    caption = [clean_number(w) for w in caption.strip().lower().split()]
                    caption_new, span_new, input_ids = self.ajust_tokens_spans(caption, span)
                    if TXT_MAX_LENGTH < 1000 and (len(caption_new) < 2 or len(caption_new) > max_length):
                        removed.append((idx, len(caption_new)))
                        continue
                    self.ids_captions_spans.append({'img_id':int(img_id), 'caption': caption_new, 'span':span_new, 'input_ids': input_ids})
                    indexes.append(idx)
        self.length = len(self.ids_captions_spans)
        self.im_div = TXT_IMG_DIVISOR
        print("removed idx: ")
        print(removed)

        if load_img:
            self.images = np.load(os.path.join(data_path, 'all_resn-152.npy'))
        else:
            self.images = np.zeros((10020, img_dim))

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist()
        indice = sorted(indice, key=lambda k: len(self.ids_captions_spans[k]))
        self.ids_captions_spans = [self.ids_captions_spans[k] for k in indice]

    def __getitem__(self, index):
        img_id = self.ids_captions_spans[index]['img_id']
        image = torch.tensor(self.images[img_id])
        input_ids = self.ids_captions_spans[index]['input_ids']
        input_ids = torch.tensor(input_ids)
        span = self.ids_captions_spans[index]['span']
        span = torch.tensor(span)
        return image, input_ids, index, img_id, span

    def __len__(self):
        return self.length
    
    def ajust_tokens_spans(self, caption, span):
        tok_cap = self.tokenizer(caption, add_special_tokens= False, is_split_into_words=True, return_offsets_mapping = True)
        sent = self.tokenizer.convert_ids_to_tokens(tok_cap['input_ids'])
        if len(caption) != len(tok_cap['input_ids']):
            #is_subword = np.array(tok_cap['offset_mapping'])[:,0] != 0
            is_subword = np.array(["##" in token for token in sent])
            subword_spans = []
            end = -1
            for i in reversed(range(len(is_subword))):
                if end > 0:
                    subword_spans.append([i, end])
                if is_subword[i] and end == -1:
                    end = i
                if not is_subword[i]:
                    end = -1
            subword_spans = sorted(subword_spans, key=itemgetter(1))
            mapping = [[x,x] for x in range(len(caption))]
            inc = 0
            diff = 0
            for j in range(len(is_subword)):
                if is_subword[j]:
                    inc += 1
                    diff += 1  
                else:
                    for m in range((j-diff),len(mapping)):
                        mapping[m][1] += inc
                    inc = 0          
            for [b,e] in span:
                subword_spans.append([mapping[b][1], mapping[e][1]])
            span = subword_spans
        caption = self.tokenizer.convert_ids_to_tokens(tok_cap['input_ids'])
        return caption, span, tok_cap['input_ids']

def collate_fun(data):
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, ids, img_ids, spans = zipped_data
    images = torch.stack(images, 0)
    max_len = max([len(caption) for caption in captions])
    targets = torch.zeros(len(captions), max_len).long()
    lengths = [len(cap) for cap in captions]
    indices = torch.zeros(len(captions), max_len, 2).long()
    for i, cap in enumerate(captions):
        cap_len = len(cap)
        targets[i, : cap_len] = cap[: cap_len]
        indices[i, : cap_len - 1, :] = spans[i]
    return images, targets, lengths, ids, indices

def get_data_loader(data_path, data_split, tokenizer,
                    batch_size=128,
                    shuffle=True,
                    nworker=2,
                    load_img=True,
                    img_dim=2048,
                    tiny = False,
                    sampler=True):
    dset = DataLoader(data_path, data_split, tokenizer, load_img, img_dim, batch_size, tiny)
    if sampler:
        model = SortedRandomSampler
        if not isinstance(sampler, bool) and issubclass(sampler, data.Sampler):
            model = sampler
        #sampler = SortedRandomSampler(dset)
        sampler = model(dset)
    data_loader = torch.utils.data.DataLoader(
                    dataset=dset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    pin_memory=True,
                    collate_fn=collate_fun
    )
    return data_loader

def get_data_iters(data_path, prefix, tokenizer, batch_size, nworker, shuffle=False, sampler=True, load_img=True, tiny=False, split='train'):
    if split == 'test':
        shuffle=False
        sampler=None
        load_img=False
    data_loader = get_data_loader(data_path, prefix, tokenizer, batch_size=batch_size, shuffle=shuffle, sampler=sampler, nworker=nworker, load_img=load_img, tiny=tiny)
    return data_loader
