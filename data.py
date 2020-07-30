import json
import re
import os
import codecs
from os.path import join
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from cytoolz import compose
from batcher import coll_fn, prepro_fn
from batcher import convert_batch, batchify_fn
from batcher import BucketedGenerater
from utils import PAD, UNK, START, END
# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 6400


class CnnDmDataset(Dataset):
    def __init__(self, split: str, path: str) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with codecs.open(join(self._data_path, '{}.json'.format(i)), 'r', encoding='utf-8') as f:
            js = json.loads(f.read())
        return js


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


class MatchDataset(CnnDmDataset):
    def __init__(self, split, data_dir):
        super().__init__(split, data_dir)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted'])
        matched_arts = [art_sents[i] for i in extracts]
        return matched_arts, abs_sents[:len(extracts)]


def build_batchers(max_art, max_abs, word2id, data_dir, cuda, debug):
    prepro = prepro_fn(max_art, max_abs)

    def sort_key(sample):
        src, target = sample
        return (len(target), len(src))

    batchify = compose(batchify_fn(PAD, START, END, cuda=cuda), 
                       convert_batch(UNK, word2id))

    train_loader = DataLoader(
        MatchDataset('train', data_dir), batch_size=BUCKET_SIZE,
        shuffle=not debug, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn)
    val_loader = DataLoader(
        MatchDataset('val', data_dir), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=False)
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=False)
    return train_batcher, val_batcher
