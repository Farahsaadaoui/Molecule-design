import sys
from io import StringIO
import argparse
import time
import torch
from torchtext import data

#from Process import *
import torch.nn.functional as F
#from Optim import CosineWithRestarts
#from Batch import create_masks
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import pandas as pd
import pdb
import dill as pickle
import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
#from Models import get_model
#from Beam import beam_search
import torch
import torch.nn as nn
import torch.nn.functional as F
#from Layers import EncoderLayer, DecoderLayer
#from Embed import Embedder, PositionalEncoder
#from Sublayers import Norm
import copy
import numpy as np
from nltk.corpus import wordnet
from torch.autograd import Variable
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import re
import pandas as pd
import numpy as np
import math

#import moses
#from rand_gen import rand_gen_from_data_distribution, tokenlen_gen_from_data_distribution
from torch.autograd import Variable

#batch.py
import torch
from torchtext import data
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    if opt.use_cond2dec == True:
        cond_mask = np.zeros((1, opt.cond_dim, opt.cond_dim))
        cond_mask_upperright = np.ones((1, opt.cond_dim, size))
        cond_mask_upperright[:, :, 0] = 0
        cond_mask_lowerleft = np.zeros((1, size, opt.cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    #if opt.device == 0:
      #np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, cond, opt):
    torch.set_printoptions(profile="full")
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    cond_mask = torch.unsqueeze(cond, -2)
    cond_mask = torch.ones_like(cond_mask, dtype=bool)
    src_mask = torch.cat([cond_mask, src_mask], dim=2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        if opt.use_cond2dec == True:
            trg_mask = torch.cat([cond_mask, trg_mask], dim=2)
        np_mask = nopeak_mask(trg.size(1), opt)
        #if trg.is_cuda:
            #np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask

# patch on Torchtext's batching process that makes it more efficient
# from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

 #beam
import torch
#from Batch import nopeak_mask
import torch.nn.functional as F
import math
import numpy as np


def init_vars(cond, model, SRC, TRG, toklen, opt, z):
    init_tok = TRG.vocab.stoi['<sos>']

    src_mask = (torch.ones(1, 1, toklen) != 0)
    trg_mask = nopeak_mask(1, opt)

    trg_in = torch.LongTensor([[init_tok]])


    if opt.device == 0:
        #trg_in, z, src_mask, trg_mask = trg_in.cuda(), z.cuda(), src_mask.cuda(), trg_mask.cuda()
        trg_in, z, src_mask, trg_mask = trg_in, z, src_mask, trg_mask


    if opt.use_cond2dec == True:
        output_mol = model.out(model.decoder(trg_in, z, cond, src_mask, trg_mask))[:, 3:, :]
    else:
        output_mol = model.out(model.decoder(trg_in, z, cond, src_mask, trg_mask))
    out_mol = F.softmax(output_mol, dim=-1)
    
    probs, ix = out_mol[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_strlen).long()
    if opt.device == 0:
        outputs = outputs#.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(opt.k, z.size(-2), z.size(-1))
    if opt.device == 0:
        e_outputs = e_outputs#.cuda()
    e_outputs[:, :] = z[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(cond, model, SRC, TRG, toklen, opt, z):
    if opt.device == 0:
        #cond = cond.cuda()
        cond = cond
    cond = cond.view(1, -1)

    outputs, e_outputs, log_scores = init_vars(cond, model, SRC, TRG, toklen, opt, z)
    cond = cond.repeat(opt.k, 1)
    src_mask = (torch.ones(1, 1, toklen) != 0)
    src_mask = src_mask.repeat(opt.k, 1, 1)
    #if opt.device == 0:
        #src_mask = src_mask.cuda()
    eos_tok = TRG.vocab.stoi['<eos>']

    ind = None
    for i in range(2, opt.max_strlen):
        trg_mask = nopeak_mask(i, opt)
        trg_mask = trg_mask.repeat(opt.k, 1, 1)

        if opt.use_cond2dec == True:
            output_mol = model.out(model.decoder(outputs[:,:i], e_outputs, cond, src_mask, trg_mask))[:, 3:, :]
        else:
            output_mol = model.out(model.decoder(outputs[:,:i], e_outputs, cond, src_mask, trg_mask))
        out_mol = F.softmax(output_mol, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out_mol, log_scores, i, opt.k)
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long)#.cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        length = (outputs[0]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

#embed.py
import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)
    
#sublayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

#layer.py
import torch
import torch.nn as nn
#from Sublayers import FeedForward, MultiHeadAttention, Norm
import numpy as np


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, opt, d_model, heads, dropout=0.1):
        super().__init__()
        self.use_cond2dec = opt.use_cond2dec
        self.use_cond2lat = opt.use_cond2lat
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, cond_input, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        if self.use_cond2lat == True:
            cond_mask = torch.unsqueeze(cond_input, -2)
            cond_mask = torch.ones_like(cond_mask, dtype=bool)
            src_mask = torch.cat([cond_mask, src_mask], dim=2)

        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

#calcul proprties 
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import pandas as pd


def printProgressBar(i,max,postText):
    n_bar = 20 #size of progress bar
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"  [{'#' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()

def calcProperty(opt):
    data = [opt.src_data, opt.src_data_te]
    for data_kind in data:
        if data_kind == opt.src_data:
            print("Calculating properties for {} train molecules: logP, tPSA, QED".format(len(opt.src_data)))
        if data_kind == opt.src_data_te:
            print("Calculating properties for {} test molecules: logP, tPSA, QED".format(len(opt.src_data_te)))
        count = 0
        mol_list, logP_list, tPSA_list, QED_list = [], [], [], []

        for smi in opt.src_data:
            count += 1
            printProgressBar(int(count / len(opt.src_data) * 100), 100, 'completed!')
            mol = Chem.MolFromSmiles(smi)
            mol_list.append(smi), logP_list.append(Descriptors.MolLogP(mol)), tPSA_list.append(Descriptors.TPSA(mol)), QED_list.append(QED.qed(mol))

        if data_kind == opt.src_data:
            prop_df = pd.DataFrame({'logP': logP_list, 'tPSA': tPSA_list, 'QED': QED_list})
            prop_df.to_csv("/kaggle/working/prop_temp.csv", index=False)
        if data_kind == opt.src_data_te:
            prop_df_te = pd.DataFrame({'logP': logP_list, 'tPSA': tPSA_list, 'QED': QED_list})
            prop_df_te.to_csv("/kaggle/working/prop_temp_te.csv", index=False)

    return prop_df, prop_df_te

#tpkonize
import selfies
from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
import spacy
import re


class moltokenize(object):
    def tokenizer(self, sentence):
        return [tok for tok in atomwise_tokenizer(sentence) if tok != " "]

#process.py
import pandas as pd
import torch
import torchtext
from torchtext import data
#from Tokenize import moltokenize
#from Batch import MyIterator, batch_size_fn
import os
#import dill as pickle
import pickle
import numpy as np


def read_data(opt):
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()

    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

    if opt.src_data_te is not None:
        try:
            opt.src_data_te = open(opt.src_data_te, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_data_te + "' file not found")
            quit()

    if opt.trg_data_te is not None:
        try:
            opt.trg_data_te = open(opt.trg_data_te, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data_te + "' file not found")
            quit()


def create_fields(opt):
    lang_formats = ['SMILES', 'SELFIES']
    if opt.lang_format not in lang_formats:
        print('invalid src language: ' + opt.lang_forma + 'supported languages : ' + lang_formats)

    print("loading molecule tokenizers...")

    t_src = moltokenize()
    t_trg = moltokenize()

    SRC = data.Field(tokenize=t_src.tokenizer)
    TRG = data.Field(tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))

        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()

    return (SRC, TRG)


def create_dataset(opt, SRC, TRG, PROP, tr_te):
    # masking data longer than max_strlen
    if tr_te == "tr":
        print("\n* creating [train] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    if tr_te == "te":
        print("\n* creating [test] dataset and iterator... ")
        raw_data = {'src': [line for line in opt.src_data_te], 'trg': [line for line in opt.trg_data_te]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    df = pd.concat([df, PROP], axis=1)

    # if tr_te == "tr":  #for code test
    #     df = df[:30000]
    # if tr_te == "te":
    #     df = df[:3000]

    if opt.lang_format == 'SMILES':
        mask = (df['src'].str.len() + opt.cond_dim < opt.max_strlen) & (df['trg'].str.len() + opt.cond_dim < opt.max_strlen)
    # if opt.lang_format == 'SELFIES':
    #     mask = (df['src'].str.count('][') + opt.cond_dim < opt.max_strlen) & (df['trg'].str.count('][') + opt.cond_dim < opt.max_strlen)

    df = df.loc[mask]
    if tr_te == "tr":
        print("     - # of training samples:", len(df.index))
        df.to_csv("DB_transformer_temp.csv", index=False)
    if tr_te == "te":
        print("     - # of test samples:", len(df.index))
        df.to_csv("DB_transformer_temp_te.csv", index=False)

    logP = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    tPSA = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    QED = data.Field(use_vocab=False, sequential=False, dtype=torch.float)

    data_fields = [('src', SRC), ('trg', TRG), ('logP', logP), ('tPSA', tPSA), ('QED', QED)]

    if tr_te == "tr":
        toklenList = []
        train = data.TabularDataset('./DB_transformer_temp.csv', format='csv', fields=data_fields, skip_header=True)
        for i in range(len(train)):
            toklenList.append(len(vars(train[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv("toklen_list.csv", index=False)
        if opt.verbose == True:
            print("     - tokenized training sample 0:", vars(train[0]))
    if tr_te == "te":
        train = data.TabularDataset('./DB_transformer_temp_te.csv', format='csv', fields=data_fields, skip_header=True)
        if opt.verbose == True:
            print("     - tokenized testing sample 0:", vars(train[0]))

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg), len(x.logP), len(x.tPSA), len(x.QED)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)
    try:
        os.remove('DB_transformer_temp.csv')
    except:
        pass
    try:
        os.remove('DB_transformer_temp_te.csv')
    except:
        pass

    if tr_te == "tr":
        if opt.load_weights is None:
            print("     - building vocab from train data...")
            SRC.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of SRC: {}\n        -> {}'.format(len(SRC.vocab), SRC.vocab.stoi))
            TRG.build_vocab(train)
            if opt.verbose == True:
                print('     - vocab size of TRG: {}\n        -> {}'.format(len(TRG.vocab), TRG.vocab.stoi))
            if opt.checkpoint > 0:
                try:
                    os.mkdir("weights")
                except:
                    print("weights folder already exists, run program with -load_weights weights to load them")
                    quit()
                pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
                pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']

        opt.train_len = get_len(train_iter)

    if tr_te == "te":
        opt.test_len = get_len(train_iter)

    return train_iter


def get_len(train):
    for i, b in enumerate(train):
        pass
    return i

#embed
import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        #if x.is_cuda:
            #pe.cuda()
        x = x + pe
        return self.dropout(x)

#model
import torch
import torch.nn as nn
import torch.nn.functional as F
#from Layers import EncoderLayer, DecoderLayer
#from Embed import Embedder, PositionalEncoder
#from Sublayers import Norm
import copy
import numpy as np

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, opt, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.cond_dim = opt.cond_dim
        self.d_model = d_model
        self.embed_sentence = Embedder(vocab_size, d_model)
        self.embed_cond2enc = nn.Linear(opt.cond_dim, d_model*opt.cond_dim)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

        self.fc_mu = nn.Linear(d_model, opt.latent_dim)
        self.fc_log_var = nn.Linear(d_model, opt.latent_dim)

    def forward(self, src, cond_input, mask):
        cond2enc = self.embed_cond2enc(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
        x = self.embed_sentence(src)
        x = torch.cat([cond2enc, x], dim=1)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return self.sampling(mu, log_var), mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)
    
class Decoder(nn.Module):
    def __init__(self, opt, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.cond_dim = opt.cond_dim
        self.d_model = d_model
        self.use_cond2dec = opt.use_cond2dec
        self.use_cond2lat = opt.use_cond2lat
        self.embed = Embedder(vocab_size, d_model)
        if self.use_cond2dec == True:
            self.embed_cond2dec = nn.Linear(opt.cond_dim, d_model * opt.cond_dim) #concat to trg_input
        if self.use_cond2lat == True:
            self.embed_cond2lat = nn.Linear(opt.cond_dim, d_model * opt.cond_dim) #concat to trg_input
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.fc_z = nn.Linear(opt.latent_dim, d_model)
        self.layers = get_clones(DecoderLayer(opt, d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, cond_input, src_mask, trg_mask):
        x = self.embed(trg)
        e_outputs = self.fc_z(e_outputs)
        if self.use_cond2dec == True:
            cond2dec = self.embed_cond2dec(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1)
        if self.use_cond2lat == True:
            cond2lat = self.embed_cond2lat(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, cond_input, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, opt, src_vocab, trg_vocab):
        super().__init__()
        self.use_cond2dec = opt.use_cond2dec
        self.use_cond2lat = opt.use_cond2lat
        self.encoder = Encoder(opt, src_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
        self.decoder = Decoder(opt, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
        self.out = nn.Linear(opt.d_model, trg_vocab)
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)
    def forward(self, src, trg, cond, src_mask, trg_mask):
        z, mu, log_var = self.encoder(src, cond, src_mask)
        d_output = self.decoder(trg, z, cond, src_mask, trg_mask)
        output = self.out(d_output)
        if self.use_cond2dec == True:
            output_prop, output_mol = self.prop_fc(output[:, :3, :]), output[:, 3:, :]
        else:
            output_prop, output_mol = torch.zeros(output.size(0), 3, 1), output
        return output_prop, output_mol, mu, log_var, z

def get_model(opt, src_vocab, trg_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(opt, src_vocab, trg_vocab)
    if opt.print_model == True:
        print("model structure:\n", model)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    #if opt.device == 0:
        #model = model.cuda()

    return model

#optimize
import torch
import numpy as np
# code from AllenNLP

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        # step = self.last_epoch + 1
        step = self.last_epoch
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


class WarmUpDefault(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)


#data distribution
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import pandas as pd

def checkdata(fpath):
    # fpath = "data/moses/prop_temp.csv"
    results = pd.read_csv(fpath)

    logP, tPSA, QED = results.iloc[:, 0], results.iloc[:, 1], results.iloc[:, 2]

    figure, ((ax1,ax2,ax3)) = plt.subplots(nrows=1, ncols=3)

    sns.violinplot(y = "logP", data =results, ax=ax1, color=sns.color_palette()[0])
    sns.violinplot(y = "tPSA", data =results, ax=ax2, color=sns.color_palette()[1])
    sns.violinplot(y = "QED", data =results, ax=ax3, color=sns.color_palette()[2])

    ax1.set(xlabel='logP', ylabel='')
    ax2.set(xlabel='tPSA', ylabel='')
    ax3.set(xlabel='QED', ylabel='')

    bound_logP = get_quatiles(logP)
    for i in range(4):
        text = ax1.text(0, bound_logP[i], f'{bound_logP[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    bound_tPSA = get_quatiles(tPSA)
    for i in range(4):
        text = ax2.text(0, bound_tPSA[i], f'{bound_tPSA[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    bound_QED = get_quatiles(QED)
    for i in range(4):
        text = ax3.text(0, bound_QED[i], f'{bound_QED[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    #plt.show()

    logP_max, logP_min = min(bound_logP[0], logP.max()), max(bound_logP[-1], logP.min())
    tPSA_max, tPSA_min = min(bound_tPSA[0], tPSA.max()), max(bound_tPSA[-1], tPSA.min())
    QED_max, QED_min = min(bound_QED[0], QED.max()), max(bound_QED[-1], QED.min())

    return logP_max, logP_min, tPSA_max, tPSA_min, QED_max, QED_min


def get_quatiles(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    UAV = Q3 + 1.5 * IQR
    LAV = Q1 - 1.5 * IQR
    return [UAV, Q3, Q1, LAV]

#randem gene
# Adjusted from https://alpynepyano.github.io/healthyNumerics/posts/sampling_arbitrary_distributions_with_python.html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_distrib1(xc, count_c):
    with plt.style.context('fivethirtyeight'):
        plt.figure(figsize=(17,5))
        plt.plot(xc,count_c, ls='--', lw=1, c='b')
        wi = np.diff(xc)[0]*0.95
        plt.bar (xc, count_c, color='gold', width=wi, alpha=0.7, label='Histogram of data')
        plt.title('Data distribution of tokenlen', fontsize=25, fontweight='bold')
        plt.show()
    return

def plot_line(X,Y,x,y):
    with plt.style.context('fivethirtyeight'):
        fig, ax1 = plt.subplots(figsize=(17,5))
        ax1.plot(X,Y, 'mo-', lw=7, label='discrete CDF', ms=20)
        ax1.legend(loc=6, frameon=False)
        ax2 = ax1.twinx()
        ax2.plot(x,y, 'co-', lw=7, label='discrete PDF', ms=20)
        ax2.legend(loc=7, frameon=False)
        ax1.set_ylabel('CDF-axis');  ax2.set_ylabel('PDF-axis');
        plt.title('Tokenlen: CDF and PDF', fontsize=25, fontweight='bold')
        plt.show()

def plot_distrib3(xc, myPDF, X):
    with plt.style.context('fivethirtyeight'):
        plt.figure(figsize=(17,5))
        width, ms = 0.5, 20
        plt.bar(xc, X, color='blue', width=width, label='resampled PDF')
        plt.plot(xc, np.zeros_like(X) ,color='magenta', ls='-',lw=13, alpha=0.6)
        plt.plot(xc, myPDF, 'co-', lw=7, label='discrete PDF', ms=ms, alpha=0.5)
        plt.title('Tokenlen sampling from data distribution', fontsize=25, fontweight='bold')
        plt.legend(loc='upper center', frameon=False)
        plt.show()

def get_sampled_element(myCDF):
    a = np.random.uniform(0, 1)
    return np.argmax(myCDF>=a)-1

def run_sampling(xc, dxc, myPDF, myCDF, nRuns):
    sample_list = []
    X = np.zeros_like(myPDF, dtype=int)
    for k in np.arange(nRuns):
        idx = get_sampled_element(myCDF)
        sample_list.append(xc[idx] + dxc * np.random.normal() / 2)
        X[idx] += 1
    return np.array(sample_list).reshape(nRuns, 1), X/np.sum(X)

def tokenlen_gen_from_data_distribution(data, nBins, size):
    count_c, bins_c, = np.histogram(data, bins=nBins)
    myPDF = count_c / np.sum(count_c)
    dxc = np.diff(bins_c)[0]
    xc = bins_c[0:-1] + 0.5 * dxc

    myCDF = np.zeros_like(bins_c)
    myCDF[1:] = np.cumsum(myPDF)

    tokenlen_list, X = run_sampling(xc, dxc, myPDF, myCDF, size)

    # plot_distrib1(xc, myPDF)
    # plot_line(bins_c, myCDF, xc, myPDF)
    # plot_distrib3(xc, myPDF, X)

    return tokenlen_list

def rand_gen_from_data_distribution(data, size, nBins):
    H, edges = np.histogramdd(data.values, bins=(nBins[0], nBins[1], nBins[2]))
    P = H/len(data)
    P_flatten = P.reshape(-1)

    dxc_logP, dxc_tPSA, dxc_QED = np.diff(edges[0])[0], np.diff(edges[1])[0], np.diff(edges[2])[0]
    xc_logP, xc_tPSA, xc_QED = edges[0][0:-1] + 0.5 * dxc_logP, edges[1][0:-1] + 0.5 * dxc_tPSA, edges[2][0:-1] + 0.5 * dxc_QED

    samples_idx = np.random.choice(len(P_flatten), size=size, p=P_flatten)
    samples_idx = np.array(np.unravel_index(samples_idx, P.shape)).T

    samples = np.zeros_like(samples_idx, dtype=np.float64)

    for i in range(len(samples_idx)):
        samples[i] = [xc_logP[samples_idx[i][0]], xc_tPSA[samples_idx[i][1]], xc_QED[samples_idx[i][2]]]

    random_noise = np.random.uniform(low=-0.5, high=0.5, size=np.shape(samples))
    random_noise[:, 0] = random_noise[:, 0] * dxc_logP
    random_noise[:, 1] = random_noise[:, 1] * dxc_tPSA
    random_noise[:, 2] = random_noise[:, 2] * dxc_QED

    samples = samples + random_noise

    return samples

import sys
from io import StringIO
import argparse
import time
import torch
import torch.nn.functional as F
import pdb
import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from nltk.corpus import wordnet
from torch.autograd import Variable
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import re
import numpy as np
import math


def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def gen_mol(cond, model, opt, SRC, TRG, toklen, z):
    model.eval()

    robustScaler = joblib.load(opt.load_weights + '/scaler.pkl')
    if opt.conds == 'm':
        cond = cond.reshape(1, -1)
    elif opt.conds == 's':
        cond = cond.reshape(1, -1)
    elif opt.conds == 'l':
        cond = cond.reshape(1, -1)
    else:
        cond = np.array(cond.split(',')[:-1]).reshape(1, -1)

    cond = robustScaler.transform(cond)
    cond = Variable(torch.Tensor(cond))
    
    sentence = beam_search(cond, model, SRC, TRG, toklen, opt, z)
    return sentence

""" def inference(opt, model, SRC, TRG):
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
  
    conds = opt.conds.split(';')
    toklen_data = pd.read_csv(opt.load_toklendata)
    toklen= int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=1)) + 3  # +3 due to cond2enc

    z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))

    for cond in conds:
        molecules.append(gen_mol(cond + ',', model, opt, SRC, TRG, toklen, z))
    toklen_gen = molecules[0].count(" ") + 1
    molecules = ''.join(molecules).replace(" ", "")
    m = Chem.MolFromSmiles(molecules)
    target_cond = conds[0].split(',')
    if m is None:
        #toklen-3: due to cond dim
        result = "   --[Invalid]: {}\n".format(molecules) + \
                 "   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}\n".format(target_cond[0], target_cond[1], target_cond[2], toklen-3)
    else:
        logP_v, tPSA_v, QED_v = Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)
        result = "   --[Valid]: {}\n".format(molecules) + \
                 "   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}\n".format(target_cond[0], target_cond[1], target_cond[2], toklen-3) + \
                 "   --From RDKit: logP={:,.4f}, tPSA={:,.4f}, QED={:,.4f}, GenToklen={}\n".format(logP_v, tPSA_v, QED_v, toklen_gen)

    return result  """

def inference(opt, model, SRC, TRG):
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []

    conds = opt.conds.split(';')
    toklen_data = pd.read_csv(opt.load_toklendata)
    toklen = int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=1)) + 3  # +3 due to cond2enc

    z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))

    for cond in conds:
        molecules.append(gen_mol(cond + ',', model, opt, SRC, TRG, toklen, z))
    toklen_gen = molecules[0].count(" ") + 1
    molecules = ''.join(molecules).replace(" ", "")
    m = Chem.MolFromSmiles(molecules)
    target_cond = conds[0].split(',')

    if m is None:
        # toklen-3: due to cond dim
        result = "   --[Invalid]: {}\n".format(molecules) + \
                 "   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}\n".format(target_cond[0], target_cond[1],
                                                                                   target_cond[2], toklen - 3)
    else:
        logP_v, tPSA_v, QED_v = Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)
        result = "   --[Valid]: {}\n".format(molecules) + \
                 "   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}\n".format(target_cond[0], target_cond[1],
                                                                                   target_cond[2], toklen - 3) + \
                 "   --From RDKit: logP={:,.4f}, tPSA={:,.4f}, QED={:,.4f}, GenToklen={}\n".format(logP_v, tPSA_v, QED_v,
                                                                                                 toklen_gen)

    return molecules, result


from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64

def generate_2d_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol)

        # Encode image as base64 string
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str
    else:
        return None

import nglview as nv
from rdkit import Chem
from io import BytesIO
import base64
from IPython.display import display
import py3Dmol


def chemcepterize_mol(mol, embed=20.0, res=0.5):
    dims = int(embed*2/res)
    #print(dims)
   
    #print(mol)
    #print(",,,,,,,,,,,,,,,,,,,,,,")
    cmol = Chem.Mol(mol.ToBinary())
    #print(cmol)
    #print(",,,,,,,,,,,,,,,,,,,,,,")
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    #print(coords)
    #print(",,,,,,,,,,,,,,,,,,,,,,")
    vect = np.zeros((dims,dims,4))
    #Bonds first
    for i,bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0,1,int(1/res*2)) #
        for f in frac:
            c = (f*bcoords + (1-f)*ecoords)
            idx = int(round((c[0] + embed)/res))
            idy = int(round((c[1]+ embed)/res))
            #Save in the vector first channel
            vect[ idx , idy ,0] = bondorder
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
            vect[ idx , idy, 1] = atom.GetAtomicNum()
            #Gasteiger Charges
            charge = atom.GetProp("_GasteigerCharge")
            vect[ idx , idy, 3] = charge
            #Hybridization
            hyptype = atom.GetHybridization().real
            vect[ idx , idy, 2] = hyptype
            
    return vect

from flask import Flask, jsonify, request,render_template
import argparse
import torch
import numpy as np
from rdkit import Chem
from keras.models import load_model
from rdkit import DataStructs
import base64
from rdkit.Chem import Draw
from io import BytesIO, StringIO
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras import backend as K
import io
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )  

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

optimizer = Adam(learning_rate=0.00025)
lr_metric = get_lr_metric(optimizer)

# Load the model with custom metric function
model = load_model('weights.best2.hdf5', custom_objects={'coeff_determination': coeff_determination, 'MeanSquaredError': MeanSquaredError, 'MeanAbsoluteError': MeanAbsoluteError, 'lr': lr_metric})

# load the dataset of SMILES strings
df = pd.read_csv('10ksmiles.csv')
df = df.drop(df.index[0])

from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mysql.connector
from flask import redirect, url_for,flash
from mysql.connector import pooling
from flask import g

app = Flask(__name__)


connection_pool = pooling.MySQLConnectionPool(
    pool_name="my_pool",
    pool_size=20,  # Adjust the pool size as needed
    host="localhost",
    user="root",
    password="wassa4ever",
    database="mol"
)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/generate_molecule')
def generate_molecule_page():
   return render_template('generate_molecule.html')

@app.route("/sign_up")
def signup():
    return render_template("signup_page.html")

@app.route('/mainpage')
def mainpage():
    return render_template('mainpage.html')

@app.route('/home_page')
def home_page():
   return render_template('home_page.html')

@app.route('/predict_properties')
def predict_properties_page():
   return render_template('predict_properties.html')

@app.route('/toxicity_prediction')
def toxicity_prediction_page():
   return render_template('toxicity_prediction.html')

@app.route('/find_similar_molecules')
def find_similar_molecules_page():
   return render_template('find_similar_molecules.html')

@app.route('/visualization')
def visualization_page():
   return render_template('2d_visualization.html')

@app.route('/logout')
def logout():
   return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    # Create a new connection for sign-up
    connection = connection_pool.get_connection()

    try:
        # Create a cursor object using the connection
        cursor = connection.cursor()

        # Execute the insert query
        sql = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
        val = (username, email, password)
        cursor.execute(sql, val)

        # Commit the changes to the database
        connection.commit()

        # Set success message
        success_message = "Welcome to our world! Your signup is successful."

        # Redirect to home page after a delay
        return f'''
            <script>
                setTimeout(function() {{
                    alert("{success_message}");
                    window.location.href = "{url_for('home')}";
                }}, 1500); // 3 seconds delay
            </script>
        '''
    finally:
        # Close the cursor and connection
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.route('/signin', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']

        # Create a new connection for sign-in
        connection = connection_pool.get_connection()

        try:
            # Create a cursor object using the connection
            cursor = connection.cursor()

            # Execute the select query
            sql = "SELECT * FROM users WHERE username = %s AND password = %s"
            val = (username, password)
            cursor.execute(sql, val)

            # Fetch all rows from the result set
            result = cursor.fetchall()

            # Check if a record is found
            if result:
                # Redirect to home page
                return render_template('mainpage.html')
            else:
                # Render sign-in page with error message
                error = "Invalid username or password"
                return render_template('index.html', error=error, username=username, password=password)
        finally:
            # Close the cursor and connection
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
    else:
        # Render sign-in page
        return render_template('index.html')


@app.route('/process_visualization', methods=['POST'])
def process_visualization():
    smiles = request.form['smiles']
    image_2d_str = generate_2d_image(smiles)

    return render_template('2d_visualization.html', image_2d_str=image_2d_str)


@app.route('/predict_properties', methods=['POST'])
def predict_properties():
    
    # Load the dataset
    dataprop=pd.read_csv("2000_new.csv")
    dataprop.rename(columns={'logp': 'LogP', 'tpsa': 'TPSA'}, inplace=True)

    # Convert the SMILES strings to RDKit molecules
    dataprop['Molecule'] = dataprop['canonical_smiles'].apply(lambda x: Chem.MolFromSmiles(x))

    # Calculate the molecular descriptors
    dataprop['MW'] = dataprop['Molecule'].apply(lambda x: Descriptors.MolWt(x))
    dataprop['HBA'] = dataprop['Molecule'].apply(lambda x: Descriptors.NumHAcceptors(x))
    dataprop['HBD'] = dataprop['Molecule'].apply(lambda x: Descriptors.NumHDonors(x))

    # Select the input features and target variables
    X = dataprop[['LogP', 'TPSA']]
    y = dataprop[['MW', 'HBA', 'HBD']]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest multi-output regressor model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    mor = MultiOutputRegressor(rf)
    mor.fit(X_train, y_train)


    # Get the SMILES string from the user input
    smiles = request.form['mol_smiles']
    mol = Chem.MolFromSmiles(smiles)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    y_pred = mor.predict([[logp, tpsa]])

    result = f'Molecular Weight: {y_pred[0][0]:.2f}, HBA: {y_pred[0][1]:.2f}, HBD: {y_pred[0][2]:.2f}'
    
    # Render the result in the HTML template
    return render_template('predict_properties.html', pred_text=result)


@app.route('/find_similar_molecules', methods=['POST'])
def find_similar_molecules():
    # get the SMILES string from the form
    initial_smiles = request.form['initial_smiles']

    # convert the SMILES to a molecule object
    initial_mol = Chem.MolFromSmiles(initial_smiles)

    # generate the Morgan fingerprint for the initial molecule
    fp = AllChem.GetMorganFingerprintAsBitVect(initial_mol, 2, nBits=1024)

    # calculate the Morgan fingerprint for each compound in the dataset
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024) for smiles in df['30']]

    # calculate the similarity between the initial molecule and each compound in the dataset
    similarity_scores = [DataStructs.TanimotoSimilarity(fp, f) for f in fps]

    # select the top 10 most similar compounds
    top_indices = np.argsort(similarity_scores)[-10:]
    top_compounds = [Chem.MolFromSmiles(df.iloc[i]['30']) for i in top_indices]
    similar_smiles = [Chem.MolToSmiles(mol) for mol in top_compounds]

    # Generate 2D images for each similar SMILES
    image_strs = [generate_2d_image(smile.strip()) for smile in similar_smiles]

    # Zip similar SMILES and image strings together
    zipped_data = zip(similar_smiles, image_strs)

    # render the template with the zipped data
    return render_template('find_similar_molecules.html', zipped_data=zipped_data)

    

@app.route('/predict_toxicity', methods=['POST'])
def predict_toxicity():
    # get the SMILES string from the form data
    mol_smiles = request.form['mol_smiles']
    # convert the SMILES string to an RDKit molecule object
    mol = Chem.MolFromSmiles(mol_smiles)
    # define the vectorize function to generate the tensor representation for the molecule
    def vectorize(mol):
        return chemcepterize_mol(mol, embed=12)
    # apply the vectorize function to generate the tensor representation for the molecule
    mol_tensor = vectorize(mol)
    # make a prediction on the tensor representation of the molecule using the loaded model
    tox_pred = model.predict(np.expand_dims(mol_tensor, axis=0))
    # return the toxicity prediction as a string
    tox_prediction = f"Toxicity Prediction: {tox_pred[0,0]}"
    return render_template('toxicity_prediction.html', tox_prediction=tox_prediction)

@app.route('/generate_molecule', methods=['POST'])
def generate_molecule():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('-load_weights', type=str, default="content/weight")
    parser.add_argument('-load_traindata', type=str, default="content/data/moses/prop_temp.csv")
    parser.add_argument('-load_toklendata', type=str, default="content/toklen_list.csv")
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-max_strlen', type=int, default=80) #max 80
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)

    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-cond_dim', type=int, default=3)
    parser.add_argument('-latent_dim', type=int, default=128)

    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-lr_beta1', type=int, default=0.9)
    parser.add_argument('-lr_beta2', type=int, default=0.98)

    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    
    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    logP = float(request.form['logP'])
    tPSA = float(request.form['tPSA'])
    QED = float(request.form['QED'])

    opt.conds = ','.join([str(logP), str(tPSA), str(QED)])

 
    prediction_molecules, prediction_text = inference(opt, model, SRC, TRG)
    image_2d_str = generate_2d_image(prediction_molecules.strip())
    
   

    return render_template('generate_molecule.html', prediction_text=prediction_text, image_2d_str=image_2d_str)
if __name__ == '__main__':
    app.run(debug=True)