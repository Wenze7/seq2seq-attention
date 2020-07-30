""" utility functions"""
import re
import os
import math
import numpy as np
import gensim
import torch
from torch import nn
from torch.nn import functional as F

PAD = 0
UNK = 1
START = 2
END = 3


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
    return word2id


def make_embedding(id2word, w2v_file, emb_dim, initializer=None):
    print("loading word vectors...")
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    print("vocab_size:", vocab_size)

    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                print("oov:", id2word[i])
                oovs.append(i)
    print("total oov: ", len(oovs))
    return embedding, oovs


def sequence_loss(logits, targets, xent_fn=None, pad_idx=0):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()

    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))
    if xent_fn:
        loss = xent_fn(logit, target)
    else:
        loss = F.cross_entropy(logit, target, reduce=False)
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))
    return loss


def sequence_mean(sequence, seq_lens, dim=1):
    if seq_lens:
        assert sequence.size(0) == len(seq_lens)   # batch_size
        sum_ = torch.sum(sequence, dim=dim, keepdim=False)
        mean = torch.stack([s/l for s, l in zip(sum_, seq_lens)], dim=0)
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=False)
    return mean


def len_mask(lens, device):
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).to(device)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)
    return mask