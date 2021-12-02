import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
from sklearn.model_selection import StratifiedKFold


def prepare_features(tokenizer, example, max_length):
    tokenized_example = tokenizer(example['Review'], max_length = max_length, padding = 'max_length', 
                                  add_special_tokens = True, truncation = True)
    if 'Sentiment' in example:
        tokenized_example["label"] = example['Sentiment']
    
    return tokenized_example

def create_k_folds(df, num_splits):
    stratifier = StratifiedKFold(shuffle = True, n_splits = num_splits, random_state = 42)
    for fold_idx, (train_idx, val_idx) in enumerate(stratifier.split(df, df['Sentiment'])):
        df.loc[val_idx, 'kfold'] = fold_idx
    return df

def create_dataloader(dataset, batch_size, mode):
    if mode == 'train':
        return DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size = batch_size,
            num_workers = 2,)
    
    return DataLoader(
        dataset,
        sampler = SequentialSampler(dataset),
        batch_size = batch_size,
        num_workers = 2,)

class CreateDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataset = {}
        dataset["input_ids"] = torch.tensor(self.data[idx]["input_ids"], dtype = torch.long)
        dataset["attention_mask"] = torch.tensor(self.data[idx]["attention_mask"], dtype = torch.long)
        
        if "label" in self.data[idx]:
            dataset["label"] = torch.tensor(self.data[idx]["label"], dtype = torch.long)
        
        if "token_type_ids" in self.data[idx]:
            dataset["token_type_ids"] = torch.tensor(self.data[idx]["token_type_ids"], dtype = torch.long)
        
        return dataset

class AverageMeter():
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def update(self, new_val, cnt = 1):
        self.val = new_val
        self.sum += new_val * cnt
        self.count += cnt
        self.avg = self.sum / self.count