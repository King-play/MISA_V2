import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

from create_dataset import MOSI, MOSEI, UR_FUNNY, PAD, UNK


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MSADataset(Dataset):
    def __init__(self, config):

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    
    print(config.mode)
    config.data_len = len(dataset)


    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things

        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer

        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            # 修改这里：使用padding='max_length'而不是pad_to_max_length=True
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN+2, add_special_tokens=True, 
                padding='max_length', truncation=True, return_tensors=None)
            bert_details.append(encoded_bert_sent)

        try:
            # 找出当前批次中最大长度
            max_bert_len = max([len(sample["input_ids"]) for sample in bert_details])
            
            # 对每个样本进行padding到最大长度
            padded_input_ids = [
                sample["input_ids"] + [0] * (max_bert_len - len(sample["input_ids"]))
                for sample in bert_details
            ]
            padded_token_type_ids = [
                sample["token_type_ids"] + [0] * (max_bert_len - len(sample["token_type_ids"]))
                for sample in bert_details
            ]
            padded_attention_mask = [
                sample["attention_mask"] + [0] * (max_bert_len - len(sample["attention_mask"]))
                for sample in bert_details
            ]
            
            # Bert things are batch_first
            bert_sentences = torch.LongTensor(padded_input_ids)
            bert_sentence_types = torch.LongTensor(padded_token_type_ids)
            bert_sentence_att_mask = torch.LongTensor(padded_attention_mask)
            
        except Exception as e:
            print("Error in collate_fn:")
            for i, sample in enumerate(bert_details):
                print(f"Sample {i} input_ids length: {len(sample['input_ids'])}")
                print(f"Sample {i} token_type_ids length: {len(sample['token_type_ids'])}")
                print(f"Sample {i} attention_mask length: {len(sample['attention_mask'])}")
            raise e

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
