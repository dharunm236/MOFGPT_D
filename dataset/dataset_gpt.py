
import functools
import numpy as np
import torch
import os
from tqdm import tqdm
import multiprocessing
from torch.utils.data import Dataset


def count_duplicates(data):
    """
    Count the number of duplicates in a list
    """
    unq_set = set()
    num_duplicates = 0
    for i in data:
        if i in unq_set:
            num_duplicates += 1
        else:
            unq_set.add(i)
    return num_duplicates

class MOF_ID_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self,
                 data,
                 tokenizer,
                 ignore_index,
                 use_multiprocessing,
                 topology_labels_map):
        self.data = data
        self.topology_labels_map = topology_labels_map
        self.inv_topology_labels_map = {v: k for k, v in topology_labels_map.items()}
        # num_duplicates = count_duplicates(self.data[:, 0])
        # print(f"Number of duplicates: {num_duplicates}")
        self.tokenizer = tokenizer
        self.use_multiprocessing = use_multiprocessing
        #     self.data = data[:int(len(data)*use_ratio)]
        self.mofid = self.data[:, 0].astype(str)
        #     self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
        self.tokens = []
        self.topology_labels = []
        self.encode_all()
        for i in range(len(self.tokens)):
            if len(self.tokens[i]) > 512:
                print(f"Token length: {len(self.tokens[i])}")
        #     self.tokens = np.array(self.tokens)
        print("Tokenizing finished")
        print(f"Number of mofs: {len(self.tokens)}")
        self.label = self.data[:, 1].astype(float)
        self.ignore_index = ignore_index


    def __len__(self):
        return len(self.label)

    def encode_all(self):
        if self.use_multiprocessing:
            with multiprocessing.Pool() as pool:
                results = list(tqdm(pool.imap(self.tokenizer.encode,
                                              self.mofid),
                                    total=len(self.mofid),
                                    desc='Tokenizing',
                                    colour='green'))
            self.tokens = results
        else:
            self.tokens = [self.tokenizer.encode(i) for i in tqdm(self.mofid,
                                                                  desc='Tokenizing',
                                                                  colour='green',
                                                                  total=len(self.mofid))]
            if self.topology_labels_map is not None:
                self.topology_labels = [self.topology_labels_map[i.split('&&')[-1]] for i in self.mofid]
                                                            
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        # Load data and get label
        token_ids = torch.from_numpy(np.asarray(self.tokens[index]))
        target_token_ids = token_ids.clone()[1:]
        mask_ids = torch.ones_like(token_ids)
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)
        return {'token_ids': token_ids,
                'mask_ids': mask_ids,
                'target_token_ids': target_token_ids,
                'label': y.float(),
                'topology_label': torch.Tensor([self.topology_labels[index]]).long()}

    def collate_fn(self, data):
        """
        add padding to the batch of data
        """
        padded_tokens, \
            padded_masks, \
            target_tokens = self.tokenizer.pad_batched_tokens([i['token_ids'] for i in data],
                                                              [i['mask_ids']
                                                                  for i in data],
                                                              [i['target_token_ids'] for i in data])
        labels = torch.stack([i['label'] for i in data])
        topology_labels = torch.stack([i['topology_label'] for i in data])
        return {"token_ids": padded_tokens,
                "mask_ids": padded_masks,
                "target_token_ids": target_tokens,
                "label": labels,
                "topology_label": topology_labels}


# class MOF_pretrain_Dataset(Dataset):
#     'Characterizes a dataset for PyTorch'

#     def __init__(self, data, tokenizer, use_ratio=1):

#         self.data = data[:int(len(data)*use_ratio)]
#         self.mofid = self.data.astype(str)
#         self.tokens = np.array([tokenizer.encode(
#             i, max_length=512, truncation=True, padding='max_length') for i in self.mofid])
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.mofid)

#     @functools.lru_cache(maxsize=None)
#     def __getitem__(self, index):
#         # Load data and get label
#         X = torch.from_numpy(np.asarray(self.tokens[index]))

#         return X.type(torch.LongTensor)


class MOF_tsne_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, tokenizer):
        self.data = data
        self.mofid = self.data[:, 0].astype(str)
        self.tokens = np.array([tokenizer.encode(
            i, max_length=512, truncation=True, padding='max_length') for i in self.mofid])
        self.label = self.data[:, 1].astype(float)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        # Load data and get label
        X = torch.from_numpy(np.asarray(self.tokens[index]))
        y = self.label[index]
        topo = self.mofid[index].split('&&')[-1].split('.')[0]
        return X, y, topo


# if __name__ == "__main__":
#     config_filename = "../config/config.yaml"
#     with open(config_filename, 'r') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     config_tokenizer = config['tokenizer']
#     config_data = config['data']
#     config_model = config['model']

    