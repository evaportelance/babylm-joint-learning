import os
import re
import pickle
import json
import numpy as np
import random
import torch
import torch.utils.data
from Utils import clean_number
from torch.nn.utils.rnn import pad_sequence

TXT_IMG_DIVISOR = 1
TXT_MAX_LENGTH = 45

class AbstractScene(torch.utils.data.Dataset):
    def __init__(self, data_path, data_split, vocab, load_img=True, img_dim=2048, batch_size=1, is_test=False, min_length=1, max_length=100):

        self.vocab = vocab

        dict_path = os.path.join(data_path, 'train_dict.pickle')
        imgs_path = os.path.join(data_path, 'all_resn-152.npy')

        with open(dict_path, 'rb') as f: # load data dicts
            data = pickle.load(f)

        self.subwords = data[data_split]['subwords']
        self.lm_source = data[data_split]['lm_source']
        self.lm_target = data[data_split]['lm_target']
        self.caption = data[data_split]['caption']
        self.span = data[data_split]['span']
        self.id = data[data_split]['id']
        self.label = data[data_split]['label']
        self.tag = data[data_split]['tag']
        self.mapping = data[data_split]['mapping']

        if load_img:
            self.images = np.load(imgs_path)
        else:
            self.images = np.zeros((10020, img_dim))

        if is_test == True:
            self.subwords = self.subwords[0:32]
            self.lm_source = self.lm_source[0:32]
            self.lm_target = self.lm_target[0:32]
            self.caption = self.caption[0:32]
            self.span = self.span[0:32]
            self.id = self.id[0:32]
            self.label = self.label[0:32]
            self.tag = self.tag[0:32]
            self.images = self.images[0:32]
            self.mapping = self.mapping[0:32]

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist()
        indice = sorted(indice, key=lambda k: len(self.caption[k]))
        self.caption = [self.caption[k] for k in indice]
        self.span = [self.span[k] for k in indice]

    def __getitem__(self, index):

        subwords = self.subwords[index]
        lm_source = self.lm_source[index]
        lm_target = self.lm_target[index]
        idx = self.id[index]
        span = self.span[index]
        image = self.images[idx]
        caption = self.caption[index]
        lengths = len(caption)
        mapping = self.mapping[index]

        return torch.tensor(image), torch.LongTensor(caption), lengths, torch.LongTensor(span), torch.LongTensor(lm_source), torch.LongTensor(lm_target), mapping

    def __len__(self):
        return len(self.caption)


def collate_train_fun(data):
    '''
    Padding to the longest in a batch 
    '''

    images, captions, lengths, spans, lm_source, lm_target, mapping = list(zip(*data))

    images = torch.stack(images, dim=0)
    lengths = torch.LongTensor(lengths)
    captions = pad_sequence(captions, batch_first=True, padding_value=1)
    spans = pad_sequence(spans, batch_first=True, padding_value=1)
    lm_source = pad_sequence(lm_source, batch_first=True, padding_value=1)
    lm_target = pad_sequence(lm_target, batch_first=True, padding_value=1)

    return images, captions, lengths, spans, lm_source, lm_target, mapping



class SortedBlockSampler(torch.utils.data.Sampler):
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
        indice = torch.randperm(len(groups)).tolist()
        groups = [groups[k] for k in indice]
        if self.strip_last:
            groups.append(self.groups[-1])
        indice = []
        for i, group in enumerate(groups):
            indice.extend(group)
        assert len(indice) == len(self.data_source)
        return iter(indice)

    def __len__(self):
        return len(self.data_source)


class SortedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


class SortedSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


def align_tokens(tokens1, tokens2):
    '''
    token1: word level; list
    token2: sub-word tokens; list
    pad value == 1
    '''
    
    i = 0
    j = 0
    alignment = []
    group_idx = []

    while j < len(tokens2):
        if tokens1[i] == tokens2[j]:
            group_idx.append(0)
            alignment.append((i, j))
            i += 1
            j += 1
            
        elif tokens1[i].startswith(tokens2[j]):
            group_idx.append(1)
            alignment.append((i, j))
            j += 1
            
        elif tokens1[i].endswith(tokens2[j]):
            group_idx.append(1)
            alignment.append((i, j))
            i += 1
            j += 1
        else:
            # no alignment found, move both pointers
            group_idx.append(1)
            alignment.append((i, j))
            j += 1

    # inform when to merge
    nested_grouped_tokens = [] # list of list
    grouped_tokens = [] # one list

    for idx, item in enumerate(group_idx):
        
        if item == 0 and grouped_tokens == []:
            nested_grouped_tokens.append([idx])
        elif item == 0 and grouped_tokens != []:
            nested_grouped_tokens.append(grouped_tokens)
            grouped_tokens = []
            nested_grouped_tokens.append([idx])
        elif item != 0:
            grouped_tokens.append(idx)
             
    return alignment, group_idx, nested_grouped_tokens
    

def get_data_dicts():
    '''
    Get the data dicts from json files'''
        
    train_dict = {'caption':[], 
                 'span':[], 
                 'id':[], 
                 'label':[], 
                 'tag':[],
                 'subwords':[],
                 'lm_source':[],
                 'lm_target':[],
                 'mapping':[],
                 }
    
    valid_dict = {'caption':[], 
                 'span':[], 
                 'id':[], 
                 'label':[], 
                 'tag':[],
                 'subwords':[],
                 'lm_source':[],
                 'lm_target':[],
                 'mapping':[],
                 }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("babylm/opt-125m-strict")

    with open('/home/mila/x/xuanda.chen/lm/preprocessed-data/abstractscenes/word_vocab.pickle', 'rb') as f:
        word_vocab_tokenizer = pickle.load(f)
    
    train_caps = os.path.join(data_path, 'all_caps.json')
    train_ids = os.path.join(data_path, 'all.id')
    eval_caps = os.path.join(data_path, 'all_gold_caps.json')

    with open(train_caps, 'r') as tc, open(train_ids, 'r') as ti:    
        for line, img_id in zip(tc.readlines(), ti.readlines()):
            # load a subset of data for testing
            (caption, span) = json.loads(line)
            subtokens = tokenizer(caption.strip())['input_ids']
            subwords = [i.strip('Ġ') for i in tokenizer.tokenize(caption.strip())]
            wordtkns = [w for w in caption.strip().split(' ')]
            _, _, nested_mapping = align_tokens(wordtkns, subwords)
            # tokenizer does not add eos token
            lm_source = subtokens
            lm_target = subtokens[1:] + [0]
            caption = [word_vocab_tokenizer[w]
                    for w in caption.strip().split(' ')]
            train_dict['mapping'].append(nested_mapping)
            train_dict['subwords'].append(subwords)
            train_dict['lm_source'].append(lm_source)
            train_dict['lm_target'].append(lm_target)
            train_dict['caption'].append(caption)
            train_dict['span'].append(span)
            train_dict['id'].append(int(img_id))
            train_dict['label'].append(0) # placeholder
            train_dict['tag'].append(0)

    with open(eval_caps, 'r') as ec:
        for line in ec:
            (caption, span, label, tag) = json.loads(line)
            caption = [clean_number(w) for w in caption.strip().lower().split()]
            train_dict['mapping'].append(0)
            train_dict['subwords'].append(0)
            train_dict['lm_source'].append(0)
            train_dict['lm_target'].append(0)
            valid_dict['caption'].append(caption)
            valid_dict['span'].append(span)
            valid_dict['id'].append(0) # placeholder
            valid_dict['label'].append(label)
            valid_dict['tag'].append(tag)
    
    data_dict = {'train': train_dict, 'val': valid_dict}
    
    # Save the dictionary to a file using pickle
    with open('/home/mila/x/xuanda.chen/lm/preprocessed-data/abstractscenes/train_dict.pickle', 'wb') as f:
        pickle.dump(data_dict, f)

    # with open('/home/mila/x/xuanda.chen/lm/preprocessed-data/abstractscenes/train_dict.pickle', 'rb') as f:
    #     my_dict = pickle.load(f)


def get_word_dict():
    '''
    input json file
    '''
    train_caps = os.path.join(data_path, 'all_caps.json')
    with open(train_caps, 'r') as tc:
        lc = tc.readlines()
    
    lc = [i for line in lc for i in json.loads(line)[0].strip().split(' ')]

    vocab = set(lc)

    vcbdt = dict(zip(vocab, range(0, len(vocab))))

    print(len(vocab))

    with open('/home/mila/x/xuanda.chen/lm/preprocessed-data/abstractscenes/word_vocab.pickle', 'wb') as f:
        pickle.dump(vcbdt, f)





if __name__ == '__main__':

    data_path = '/home/mila/x/xuanda.chen/lm/preprocessed-data/abstractscenes'

    # get_data_dicts()
    # get_word_dict()

    with open('/home/mila/x/xuanda.chen/lm/preprocessed-data/abstractscenes/word_vocab.pickle', 'rb') as x:
        pickle.load(x)


    # sampler = None
    # shuffle = False
    # batch_size = 4

    # train_data = AbstractScene(data_path, data_split='train', vocab=1, load_img=False, is_test=True)

    # if sampler:
    #     model = SortedRandomSampler
    #     if not isinstance(sampler, bool) and issubclass(sampler, torch.utils.data.Sampler):
    #         model = sampler
    #     sampler = model(train_data)

    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_data,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    #     sampler=sampler,
    #     pin_memory=True,
    #     collate_fn=collate_train_fun
    # )

    # for i in train_loader:
    #     images, captions, length, spans, lm_source, lm_target, mapping = i
    #     print(captions)
    #     print(length)
    #     print(spans)
    #     print(lm_source)
    #     print(mapping)
    #     break