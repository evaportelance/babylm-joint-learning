import json
import pickle
import os
from pathlib import Path
from .utils import Vocabulary

def add_words(ifile, vocab):
    with ifile.open("r") as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.split("\t")[2]
            line = line.strip().tolower().split(" ")
            for word in line:
              if word in vocab:
                vocab[word] +=1
              else:
                vocab[word] = 1
    return vocab

def write_word_list(ofile):
    with ofile.open("w") as fw:
        json.dump(vocab, fw)

def get_complete_word_list(preprocessed_data_path):
    sentence_files = ["SimpleSentences1_clean.txt", "SimpleSentences2_clean.txt"]
    ofile = preprocessed_data_path / "complete_word_list_counts.json"
    vocab = dict()

    for ifile in sentence_files:
        vocab = add_words(preprocessed_data_path / ifile, vocab)
    write_word_list(ofile)


def create_vocab(word_list_dir, word_list_file, vocab_file, vocab_size = 2000):
    vocab = Vocabulary()
    word_list_file = word_list_dir / word_list_file
    with word_list_file.open("r") as f:
        word_list = json.load(f)
    sorted_words = sorted(word_list.items(), key=lambda x:x[1])
    id = 0
    for word,count in sorted_words:
        if id >= vocab_size:
            break
        vocab.add_word(word)
        id+=1
    vocab_file = word_list_dir / vocab_file
    with vocab_file.open("wb") as fw:
        pickle.dump(vocab, fw)
    return vocab

def get_vocab(preprocessed_dir, vocab_size = 2000):
    word_list_dir = Path(preprocessed_dir)
    word_list_file = 'complete_word_list_counts.json'
    vocab_file = 'vocab_dict.pkl'
    if vocab_file in os.listdir(word_list_dir):
        vocab_file = word_list_dir / vocab_file
        with vocab_file.open("rb") as f:
            vocab = pickle.load(f)
    elif word_list_file in os.listdir(word_list_dir):
        vocab = create_vocab(word_list_dir, word_list_file, vocab_file, vocab_size)
    else :
        get_complete_word_list(preprocessed_dir)
        vocab = create_vocab(word_list_dir, word_list_file, vocab_file, vocab_size)
    return vocab
