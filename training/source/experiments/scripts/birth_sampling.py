"""Keep all original refs, then add weighted repeats in cache-local day blocks.

Future-derived categories affect training selection only. Validation is untouched.
Unclassified references retain weight 1, allowing incremental/full scan indexing.
The sampler covers each ref once; DataLoader drop_last can discard its final batch.
"""
import json
import math
from collections import OrderedDict

import numpy as np
from torch.utils.data import Sampler


class BirthSampler(Sampler):
    def __init__(self, dataset, index_path, seed=4201, block_days=8, extra_fraction=0.5):
        self.seed, self.block_days, self.epoch = int(seed), int(block_days), 0
        self.dataset = dataset
        self.extra_fraction = float(extra_fraction)
        if self.block_days < 1 or not 0 <= self.extra_fraction <= 1:
            raise ValueError('invalid block size or extra fraction')
        lookup = {}
        with open(index_path) as f:
            for line in f:
                row = json.loads(line)
                if row['split'] == 'train':
                    key = (row['date'], int(row['ref']))
                    if key in lookup:
                        raise ValueError(f'duplicate sampling record {key}')
                    lookup[key] = min(4.0, max(1.0, float(row['weight'])))
        self.weights = np.array([lookup.get((d, int(r)), 1.0) for _, d, r in dataset.refs])
        if not np.isfinite(self.weights).all():
            raise ValueError('invalid weights')
        self.indexed_windows = sum((d, int(r)) in lookup for _, d, r in dataset.refs)
        self.by_day = OrderedDict()
        for i, (_, day, _) in enumerate(dataset.refs):
            self.by_day.setdefault(day, []).append(i)
        self.extra_n = int(math.ceil(len(dataset.refs) * self.extra_fraction))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset.refs) + self.extra_n

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        repeats = rng.choice(len(self.weights), self.extra_n, replace=True,
                             p=self.weights / self.weights.sum())
        buckets = {d: list(v) for d, v in self.by_day.items()}
        for idx in repeats:
            buckets[self.dataset.refs[int(idx)][1]].append(int(idx))
        keys = list(self.by_day)
        order = rng.permutation(len(keys))
        result = []
        for start in range(0, len(order), self.block_days):
            block = [idx for k in order[start:start + self.block_days] for idx in buckets[keys[k]]]
            rng.shuffle(block)
            result.extend(block)
        return iter(result)
