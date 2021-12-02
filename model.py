import numpy as np
import tqdm
from tqdm import tqdm_notebook, tqdm

import torch
import torch.nn as nn

from transformers import get_linear_schedule_with_warmup, AdamW
from utils import AverageMeter

class Model(nn.Module):
    def __init__(self, config, model_transformer, num_labels):
        super(Model, self).__init__()
        self.config = config
        self.model_transformer = model_transformer
        self.num_labels = num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classification_head = nn.Linear(self.config.hidden_size, num_labels)
        self.optimizer = None
        self.scheduler = None
        self.model_state = None
        self._init_weights(self.classification_head)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def fetch_optimizer(self, learning_rate):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, learning_rate)
        return opt
    
    def fetch_scheduler(self, optimizer, num_train_steps):
        sch = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )
        return sch
    
    def loss(self, logits, labels):
        if self.num_labels == 1:
            self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            self.config.problem_type = "single_label_classification"
        else:
            self.config.problem_type = "multi_label_classification"
                
        if self.config.problem_type == "regression":
            criterion = nn.MSELoss()
            if self.num_labels == 1:
                loss = criterion(logits.squeeze(), labels.squeeze())
            else:
                loss = criterion(logits, labels)
                
        elif self.config.problem_type == "single_label_classification":
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
            
        elif self.config.problem_type == "multi_label_classification":
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, labels)
        return loss
    
    def predict_class(logits):
        logits = logits.detach().cpu().numpy()
        pred_classes = np.argmax(logits * (1 / np.sum(logits, axis = -1)).reshape(logits.shape[0], 1), axis = -1)
        return pred_classes
    
    def forward(self, input_ids, attention_mask, token_type_ids = None, labels = None):
        pooled_outputs = self.model_transformer(input_ids = input_ids, attention_mask = attention_mask, 
                                token_type_ids = token_type_ids)[1]
        logits = self.classification_head(pooled_outputs)
        
        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)
            
        return logits, loss
    
    def train_one_epoch(self, dataloader, accumulation_steps, early_stop = None):
        self.train()
        pbar = tqdm(dataloader, total = len(dataloader))
        losses = AverageMeter()
        if early_stop is not None:
            monitor_score = AverageMeter()
            
        for b_idx, b in enumerate(pbar):
            self.optimizer.zero_grad()
            b['input_ids'] = b['input_ids'].to('cuda')
            b['attention_mask'] = b['attention_mask'].to('cuda')
            b['token_type_ids'] = b['token_type_ids'].to('cuda')
            b['label'] = b['label'].to('cuda')
            logits, loss = self(b['input_ids'], b['attention_mask'], b['token_type_ids'], b['label'])
            loss.backward()
            if (b_idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
            losses.update(loss.item(), dataloader.batch_size)
            
            if early_stop is not None:
                if early_stop.monitor_score_function(logits, b['label']) is not None:
                    monitor_score.update(early_stop.monitor_score_function(logits, b['label']), dataloader.batch_size)
                else:
                    monitor_score.update(loss.item(), dataloader.batch_size)
            
            pbar.set_postfix({"Train_loss ": losses.avg})
            del b['input_ids'], b['attention_mask'], b['token_type_ids'], b['label']
        
        if early_stop is not None:
            self.model_state = early_stop.update(self, monitor_score.avg)
            
    def val_one_epoch(self, dataloader, early_stop = None):
        self.eval()
        pbar = tqdm(dataloader, total = len(dataloader))
        losses = AverageMeter()
        if early_stop is not None:
            monitor_score = AverageMeter()
            
        for b_idx, b in enumerate(pbar):
            with torch.no_grad():
                b['input_ids'] = b['input_ids'].to('cuda')
                b['attention_mask'] = b['attention_mask'].to('cuda')
                b['token_type_ids'] = b['token_type_ids'].to('cuda')
                b['label'] = b['label'].to('cuda')
                
                logits, loss = self(b['input_ids'], b['attention_mask'], b['token_type_ids'], b['label'])
                losses.update(loss.item(), dataloader.batch_size)
                
                if early_stop is not None:
                    if early_stop.monitor_score_function(logits, b['label']) is not None:
                        monitor_score.update(early_stop.monitor_score_function(logits, b['label']), dataloader.batch_size)
                    else:
                        monitor_score.update(loss.item(), dataloader.batch_size)
                    
                pbar.set_postfix({"Val_loss": losses.avg})
                del b['input_ids'], b['attention_mask'], b['token_type_ids'], b['label']
                
        if early_stop is not None:
            self.model_state = early_stop.update(self, monitor_score.avg)
        
    def fit(self, epochs, learning_rate, num_train_steps, accumulation_steps, train_dataloader, 
            val_dataloader = None, early_stop = None):
        self.optimizer = self.fetch_optimizer(learning_rate)
        self.scheduler = self.fetch_scheduler(self.optimizer, num_train_steps)
        for _ in range(epochs):
            if early_stop is not None and early_stop.monitor == "in_train":
                self.train_one_epoch(train_dataloader, accumulation_steps, early_stop)
            else:
                self.train_one_epoch(train_dataloader, accumulation_steps)
            if val_dataloader is not None:
                if early_stop is not None and early_stop.monitor == "in_val":
                    self.val_one_epoch(val_dataloader, early_stop)
                else:
                    self.val_one_epoch(val_dataloader)
            if self.model_state == 'stop_training':
                break
            self.scheduler.step()