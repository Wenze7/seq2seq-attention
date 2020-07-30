import argparse
import json
import os
import re
import pickle as pkl
from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import build_batchers
from summ import Seq2SeqSumm
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer
from utils import PAD, UNK, START, END
from utils import make_vocab, make_embedding, sequence_loss


def configure_net(vocab_size, emb_dim, n_hidden, bidirectional, n_layer):
    net_args = {}
    net_args['vocab_size'] = vocab_size
    net_args['emb_dim'] = emb_dim
    net_args['n_hidden'] = n_hidden
    net_args['bidirectional'] = bidirectional
    net_args['n_layer'] = n_layer

    net = Seq2SeqSumm(**net_args)
    return net, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer'] = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size'] = batch_size
    train_params['lr_decay'] = lr_decay

    def criterion(logits, targets):
        return sequence_loss(logits, targets, xent_fn=None, pad_idx=PAD)

    return criterion, train_params


def main(args):
    # create data batcher, vocabulary batcher
    with open(os.path.join(args.data, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize)
    train_batcher, val_batcher = build_batchers(args.max_art, args.max_abs, word2id, args.data, args.cuda, args.debug)

    # make net
    net, net_args = configure_net(len(word2id), args.emb_dim, args.n_hidden, args.bi, args.n_layer)
    if args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding({i: w for w, i in word2id.items()}, args.w2v, args.emb_dim)
        net.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training('adam', args.lr, args.clip, args.decay, args.batch)

    # save experiment setting
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    with open(os.path.join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)

    # save meta information
    meta = {}
    meta['net'] = 'base_abstractor'
    meta['net_args'] = net_args
    meta['traing_params'] = train_params
    with open(os.path.join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)
    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training of the abstractor')
    parser.add_argument('--data', required=True, help='data directories')
    parser.add_argument('--path', required=True, help='root of the model')

    parser.add_argument('--vsize', type=int, action='store', default=25000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=300,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--n_hidden', type=int, action='store', default=512,
                        help='the number of hidden units of LSTM')
    parser.add_argument('--n_layer', type=int, action='store', default=2,
                        help='the number of layers of LSTM')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_art', type=int, action='store', default=300,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_abs', type=int, action='store', default=25,
                        help='maximun words in a single abstract sentence')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument('--ckpt_freq', type=int, action='store', default=100,
                        help='number of update steps for checkpoint and validation')
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    # args.bi = not args.no_bi
    args.bi = True
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)
