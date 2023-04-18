import os, sys, re
import time, json

import spacy, nltk, benepar

from tqdm import tqdm
from collections import Counter, OrderedDict, defaultdict

PARSER_NAME = 'benepar_en3_large'
parser = benepar.Parser(PARSER_NAME)

def get_caps(caps, img_id):
    this_caps = []
    for cap in caps['annotations']:
        if cap['image_id'] == img_id:
            this_caps.append(cap['caption'])
    return this_caps

def mscoco_split_data(portion='train', idx=4):
    maps = {'train': 'train', 'test': 'val', 'dev': 'val'}
    regx = 'COCO_{}2014_(\d+).jpg'.format(maps[portion])
    cap_file = root_caps + 'mscoco/annotations/captions_{}2014.json'.format(maps[portion])

    caps = json.load(open(cap_file, 'r'))

    if portion == 'train':
        ids_file = root_caps + 'mscoco/{}-{}_ids.txt'.format(portion, idx)
        out_file = root_caps + 'mscoco/{}-{}_caps.txt'.format(portion, idx)
    else:
        ids_file = root_caps + 'mscoco/{}_ids.txt'.format(portion)
        out_file = root_caps + 'mscoco/{}_caps.txt'.format(portion)

    nline = 0
    with open(ids_file, 'r') as fr, \
        open(out_file, 'w') as fw:
        for line in tqdm(fr):
            img_id = re.search(regx, line).group(1)
            this_caps = get_caps(caps, int(img_id))

            tmp_caps = []
            for cap in this_caps:
                cap = cap.strip()
                if cap:
                    tmp_caps.append(cap)
            this_caps = tmp_caps[:5]

            if len(this_caps) != 5:
                print(line)
                break # some images have more than 5 captions
            for cap in this_caps:
                # use split() to remove multi-spaces in captions
                cap = ' '.join(cap.split())
                fw.write(cap + '\n')
            nline += 1
        print('total {} samples'.format(nline))

def init_parser():
   # nltk.download('punkt')
    benepar.download(PARSER_NAME)

def write_trees(ibatch, fw, captions, bsize):
    print("--parsing {} x {}".format(bsize, ibatch))
    trees = parser.parse_sents(captions)
    for tree in trees:
        tree_str = " ".join(str(tree).split())
        fw.write(tree_str + "\n")

def parse_batch(ifile, ofile, bsize=500, flickr=False, abstractscenes=False):
    with open(ifile, "r") as fr, open(ofile, "w") as fw:
        cnt = 0
        ibatch = 0
        captions = []
        while True:
            line = fr.readline()
            if not line:
                break
            if cnt == bsize:
                ibatch += 1
                write_trees(ibatch, fw, captions, bsize)
                captions = []
                cnt = 0
            cnt += 1
            if flickr: #
                line = line.split("\t")[1]
            if abstractscenes:
                line = line.split("\t")[2]
            caption = benepar.InputSentence(line.strip().split(" "))
            captions.append(caption)
        if captions:
            write_trees(ibatch, fw, captions, bsize)

def main_parse_mscoco():
    maps = {'train': 'train', 'test': 'test', 'val': 'val'}
    caps_tmp = root_caps + "mscoco/{}_caps.{}"
    for fname in maps.keys():
        ifile = caps_tmp.format(fname, "txt")
        ofile = caps_tmp.format(fname, "parsed")
        print("ifile: {}".format(ifile))
        print("ofile: {}".format(ofile))
        parse_batch(ifile, ofile, bsize=500)

def main_parse_flickr():
    maps = {'results_20130124': 'results_20130124'}
    caps_tmp = root_caps + "flickr/{}.{}"
    for fname in maps.keys():
        ifile = caps_tmp.format(fname, "token")
        ofile = caps_tmp.format(fname, "parsed")
        print("ifile: {}".format(ifile))
        print("ofile: {}".format(ofile))
        parse_batch(ifile, ofile, flickr=True, bsize=50)

def main_parse_abstractscenes():
    files = ['SimpleSentences1_clean', 'SimpleSentences2_clean']
    caps_tmp = root_caps + "abstractscenes/{}.{}"
    for fname in files:
        ifile = caps_tmp.format(fname, "txt")
        ofile = caps_tmp.format(fname, "parsed")
        print("ifile: {}".format(ifile))
        print("ofile: {}".format(ofile))
        parse_batch(ifile, ofile, abstractscenes=True)

def flickr_read_ids(ifile, test=False):
    ids = list()
    with open(ifile, "r") as fr:
        for line in fr:
            line = line.strip()
            if test:
                ext = line.rsplit(".", 1)[-1]
                name = line.split("_", 1)[0]
                line = f"{name}.{ext}"
            ids.append(line)
    return ids

def flickr_read_all_parses(index_file, parse_file):
    data = defaultdict(list)
    with open(index_file, "r") as f1, open(parse_file, "r") as f2:
        for line1, line2 in zip(f1, f2):
            line1 = line1.split("\t")[0]
            image = line1.split("#")[0]
            data[image].append(line2)
    return data

def flickr_write_split(ids, data, ofile):
    with open(ofile, "w") as fw:
        for image in ids:
            assert image in data, f"{image} doesn't exist in the pool."
            for parse in data[image]:
                fw.write(parse)
    pass

def flickr_make_split():
    caps_tmp = root_caps + "flickr/{}_{}"
    index_file = f"{root_caps}/flickr/results_20130124.token"
    parse_file = f"{root_caps}/flickr/results_20130124.parsed"
    parses = flickr_read_all_parses(index_file, parse_file)

    maps = {'train': 'train', 'test': 'test', 'val': 'val'}
    for fname in maps.keys():
        ifile = caps_tmp.format(fname, "ids.txt")
        ofile = caps_tmp.format(fname, "caps.parsed")
        print("ifile: {}".format(ifile))
        print("ofile: {}".format(ofile))
        ids = flickr_read_ids(ifile)
        flickr_write_split(ids, parses, ofile)

def flickr_write_split(ids, data, ofile):
    with open(ofile, "w") as fw:
        for image in ids:
            assert image in data, f"{image} doesn't exist in the pool."
            for parse in data[image]:
                fw.write(parse)
    pass

def abstractscenes_read_all_parses(index_file, parse_file, parses):
    with open(index_file, "r") as f1, open(parse_file, "r") as f2:
        for line1, line2 in zip(f1, f2):
            img_id = line1.split("\t")[0]
            parses.append((img_id,line2))
    return parses

def abstractscenes_write_all():
    caps_tmp = root_caps + "abstractscenes/{}.{}"
    files = ['SimpleSentences1_clean', 'SimpleSentences2_clean']
    parses = list()
    for fname in files:
        index_file = caps_tmp.format(fname, "txt")
        parse_file= caps_tmp.format(fname, "parsed")
        parses = abstractscenes_read_all_parses(index_file, parse_file, parses)
    all_parses_file = caps_tmp.format("all_parses", "parsed")
    all_ids_file = caps_tmp.format("all_parses", "id")
    with open(all_parses_file, "w") as f1, open(all_ids_file, "w") as f2:
        for id, parse in parses:
            f2.write(str(id)+"\n")
            f1.write(parse)


if __name__ == '__main__':
    """ expect directory hierarchy like this:
        vpcfg/
        ├── flickr
        └── mscoco
    """
    root_caps = "../../preprocessed-data/" #

    # download the Benepar, only need to run once
    #init_parser()

    # extract mscoco captions for each split
    #mscoco_split_data(portion='test', idx=3)

    # parse each split, need slight modifications if you have split the train data into small splits
    #main_parse_mscoco()

    # parse the whole set of flickr captions
    #main_parse_flickr()

    # create flickr splits
    #flickr_make_split()

    # parse the whole set of abstractscenes captions
    main_parse_abstractscenes()

    # write all captions and parses into single file
    abstractscenes_write_all()
    pass
