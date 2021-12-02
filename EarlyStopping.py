import torch
import torch.nn as nn
import numpy as np

class EarlyStopping(nn.Module):
    def __init__(self, mode, patience, monitor, out_path):
        self.mode = mode
        if self.mode == 'max':
            self.best_value = -np.Inf
        if self.mode == 'min':
            self.best_value = np.Inf
        self.patience = patience
        self.counter = 0    
        self.out_path = out_path
        self.monitor = 'in_train' if monitor.startswith('train') else 'in_val'
    
    def update(self, model, new_value):
        if self.mode == 'max':
            if new_value > self.best_value:
                print("Improve from {} to {} -> Save model!".format(self.best_value, new_value))
                self.best_value = new_value
                self.save_checkpoint(model)
                self.counter = 0
            else:
                self.counter += 1
                print("Counter: {} out of {}".format(self.counter, self.patience))
        
        if self.mode == 'min':
            if new_value < self.best_value:
                print("Improve from {} to {} -> Save model!".format(self.best_value, new_value))
                self.best_value = new_value
                self.save_checkpoint(model)
                self.counter = 0
            else:
                self.counter += 1
                print("Counter: {} out of {}".format(self.counter, self.patience))
        
        if self.counter == self.patience:
            return 'stop_training'
                      
        return 'continue_training'
    
    def monitor_score_function(logits = None, labels = None):
        return
                
    def save_checkpoint(self, model):
        model_state_dict = model.state_dict()
        opt_state_dict = model.optimizer.state_dict() if model.optimizer is not None else None
        sch_state_dict = model.scheduler.state_dict() if model.scheduler is not None else None
        
        state_dict = {}
        state_dict['state_dict'] = model_state_dict
        state_dict['optimizer'] = opt_state_dict
        state_dict['scheduler'] = sch_state_dict
        
        torch.save(state_dict, self.out_path)