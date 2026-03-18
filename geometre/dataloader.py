import re
import numpy as np
import torch
from torch.utils.data import Dataset


flatten = lambda s: list(map(int, re.findall(r'\d+', s)))
    

class TrainDataset(Dataset):
    def __init__(self, queries, nentity, negative_sample_size, answer, pattern_to_neg_pool=None):
        # queries is a list of (query, query_structure) pairs
        # pattern_to_neg_pool: optional dict mapping query_structure -> np.array of entity IDs
        #   to restrict negative sampling for those patterns (e.g. gene-only for 2p queries)
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer
        self.pattern_to_neg_pool = pattern_to_neg_pool or {}

        counts = {}
        for query, pattern in queries:
            counts[pattern] = counts.get(pattern, 0) + 1
        print("Queries per structure:")
        for pattern, count in sorted(counts.items()):
            print(f"  {pattern}: {count}")

        self.queries = queries
        self.len = len(queries)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query, query_structure = self.queries[idx]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        subsampling_weight = torch.ones_like(subsampling_weight) 
        negative_sample_list = []
        negative_sample_size = 0

        neg_pool = self.pattern_to_neg_pool.get(query_structure)
        while negative_sample_size < self.negative_sample_size:
            if neg_pool is not None:
                negative_sample = neg_pool[np.random.randint(len(neg_pool), size=self.negative_sample_size*2)]
            else:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            answer = self.answer[query]
            mask = np.isin(
                negative_sample,
                answer,
                assume_unique=True,
                invert=True
            ).ravel()
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure
    
    @staticmethod
    def collate_fn(data):
        
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure
        
        
    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count

    
class DisjointDataset(Dataset):
    def __init__(self, pairs):
        # pairs: list of (id_a, id_b) integer tuples
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return torch.LongTensor(self.pairs[idx])

    @staticmethod
    def collate_fn(data):
        return torch.stack(data, dim=0)  # (batch, 2)


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
