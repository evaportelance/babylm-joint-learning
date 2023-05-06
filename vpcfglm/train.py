import os
import sys
import time
import pickle
import logging
import numpy as np
import torch
from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from Model import LanguageModel, JointModel
from Dataset import AbstractScene, collate_train_fun
from eva_metric import AverageMeter, LogCollector, validate_parser


def load_word_dict(f):
    with open(f, 'rb') as fhandle:
        wd = pickle.load(fhandle)
    return wd


def train(model, dataloader, optimizer, device, init_time):
    '''
    Train VPCFG + LM
    '''
    model.train()

    for image, caption, lengths, spans, lm_source, lm_target, mapping in dataloader:

        optimizer.zero_grad()
        data_time.update(time.time() - init_time)
        model.cpcfg.logger = train_logger
        
        lm_loss, gi_loss, lm_info, gi_info = model(
            image, caption, lengths, spans, epoch, lm_source, lm_target, mapping, device)
        loss = lm_loss + gi_loss
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            model.parameters(), pps.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - init_time)
        init_time = time.time()

        if model.cpcfg.niter % pps.log_steps == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] {e_log} {info}'
                .format(
                    epoch, 1, nbatch, e_log=str(model.cpcfg.logger), info=gi_info
                )
            )
            logger.info(lm_info)


        if model.cpcfg.niter % pps.val_step == 0:
            validate_parser(pps, valid_loader, model.cpcfg, vocab, logger, pps.visual_mode)
            logger.info(lm_info)
    
    validate_parser(pps, valid_loader, model.cpcfg, vocab, logger, pps.visual_mode)
    logger.info(lm_info)


class Parameters():

    def __init__(self) -> None:

        self.vocab_size = 8192
        self.batch_size = 8
        self.embed_dim = 256  # 400 in the paper == h_dim
        self.h_dim = 256  # 1150 in the paper
        self.n_ctx = 64
        self.max_length = 128
        self.n_head = 8
        self.n_layer = 3
        self.dropout_rate = 0.2
        self.tie_weights = True
        self.lr = 1e-4
        self.num_epochs = 10
        self.warmup_steps = 2000
        self.max_grad_norm = 1.0
        self.log_steps = 2000
        self.mode = 'train'
        self.z_dim = 64
        self.t_states = 60
        self.nt_states = 30
        self.state_dim = 256
        self.h_dim = 768
        self.w_dim = 768
        self.gpu = -1
        self.seed = 1213
        self.model_init = None
        self.w2vec_file = None
        self.vocab_name = 'coco.dict.pkl'
        self.prefix = 'all'
        self.parser_type = '2nd'
        self.share_w2vec = False
        self.visual_mode = False
        self.is_test = True
        self.shuffle = False
        self.sem_dim = 768
        self.syn_dim = 768
        self.word_dim = 768
        self.lstm_dim = 768
        self.w2vec_file = None
        self.margin = 0.2
        self.grad_clip = 3.
        self.workers = 0
        self.img_dim = 2048
        self.no_imgnorm = False
        self.optimizer = 'Adam'
        self.beta1 = 0.75
        self.beta2 = 0.999
        self.vse_mt_alpha = 0.01
        self.vse_lm_alpha = 1.0
        self.val_step = float("inf")
        self.train_data = '/network/scratch/x/xuanda.chen/babylm_data/train.txt'
        self.valid_data = '/network/scratch/x/xuanda.chen/babylm_data/dev.txt'
        self.tokenizer_path = '/network/scratch/x/xuanda.chen/baby_model'
        self.save_model_path = '/home/mila/x/xuanda.chen/lm/vc-pcfg/vpcfglm/models'
        self.logger_name = '/home/mila/x/xuanda.chen/lm/vc-pcfg/vpcfglm/output'
        self.data_path = "/home/mila/x/xuanda.chen/lm/preprocessed-data/abstractscenes"
        self.val_model = '/home/mila/x/xuanda.chen/lm/vc-pcfg/vpcfglm/val'


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda'
    torch.manual_seed(42)
    np.random.seed(42)
    pps = Parameters()

    # setup logger
    if os.path.exists(pps.logger_name):
        print('Warning: the folder {} exists.'.format(pps.logger_name))
    else:
        print('Creating {}'.format(pps.logger_name))
        os.mkdir(pps.logger_name)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(
        os.path.join(pps.logger_name, 'train.log'), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info('cuda:{}@{}'.format(pps.gpu, os.uname().nodename))

    # load word-token vocab
    vocab = load_word_dict(os.path.join(pps.data_path, 'word_vocab.pickle'))
    pps.vocab_size = len(vocab)
    logger.info("|vocab|={}".format(len(vocab)))

    # Same dataset for train and eval
    train_data = AbstractScene(
        pps.data_path, data_split='train', vocab=1, load_img=True, is_test=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=pps.batch_size,
        shuffle=True,
        # sampler=True,
        pin_memory=True,
        collate_fn=collate_train_fun
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=pps.batch_size,
        shuffle=False,
        # sampler=False,
        pin_memory=True,
        collate_fn=collate_train_fun
    )

    # 1. Load LMs
    tokenizer = AutoTokenizer.from_pretrained(pps.save_model_path)
    # tokenizer.save_pretrained(pps.save_model_path)
    config = AutoConfig.from_pretrained(pps.save_model_path)
    lm = AutoModelForCausalLM.from_pretrained(
        pps.save_model_path, config=config)
    # lm.save_pretrained(pps.save_model_path)
    plm = LanguageModel(config, lm)

    # 2. Load CPCFG
    from Modules import VGCPCFGs
    embeds = lm.model.decoder.embed_tokens
    cpcfg = VGCPCFGs(pps, vocab, logger, embeds)

    # 3. Load Joint Model
    jm = JointModel(plm, cpcfg)
    jm = jm.to(device)
    # print(jm)

    optimizer = torch.optim.Adam(
        jm.parameters(), lr=pps.lr, betas=(pps.beta1, pps.beta2)
    )

    for epoch in range(pps.num_epochs):

        train_logger = LogCollector()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        nbatch = len(train_loader)
        init_time = time.time()
        jm.cpcfg.n_word = 0
        jm.cpcfg.n_sent = 0
        jm.cpcfg.s_time = init_time
        jm.cpcfg.all_stats = [[0., 0., 0.]]
        train(jm, train_loader, optimizer, device, init_time)

    tokenizer.save_pretrained(pps.val_model)
    jm.plm.save_pretrained(pps.val_model)