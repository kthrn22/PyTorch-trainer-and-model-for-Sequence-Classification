# PyTorch-trainer-and-model-for-Sequence-Classification

After cloning the repository, modify your training data so that the training data is a ```.csv``` file and it has 2 columns: ```Text``` and ```Label```

In the below example, we will assume that our training data has ```3``` labels, the name of our training data file is ```train_data.csv```

## Example Usage

### Import dependencies
```
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

from EarlyStopping import *
from modelling import *
from utils import *
```
### Specify arguments
```args.pretrained_path``` will be the path of our pretrained language model
```
class args:
    fold = 0
    pretrained_path = 'bert-base-uncased'
    max_length = 400
    train_batch_size = 16
    val_batch_size = 64
    epochs = 5
    learning_rate = 1e-5
    accumulation_steps = 2
    num_splits = 5
``` 
### Split data
In this example we will train the model using cross-validation. We will split our training data into ```args.num_splits``` folds.
```
df = pd.read_csv('./train_data.csv')
df = create_k_folds(df, args.num_splits)

df_train = df[df['kfold'] == args.fold].reset_index(drop = True)
df_valid = df[df['kfold'] == args.fold].reset_index(drop = True)
```
### Load the language model and its tokenizer 
```
config = AutoConfig.from_pretrained(args.path)
tokenizer = AutoTokenizer.from_pretrained(args.path)
model_transformer = AutoModel.from_pretrained(args.path)
```
### Prepare train and validation dataloaders
```
features = []
for i in range(len(df_train)):
    features.append(prepare_features(tokenizer, df_train.iloc[i, :].to_dict(), args.max_length))
    
train_dataset = CreateDataset(features)
train_dataloader = create_dataloader(train_dataset, args.train_batch_size, 'train')

features = []
for i in range(len(df_valid)):
    features.append(prepare_features(tokenizer, df_valid.iloc[i, :].to_dict(), args.max_length))
    
val_dataset = CreateDataset(features)
val_dataloader = create_dataloader(val_dataset, args.val_batch_size, 'val')
```
### Use EarlyStopping and customize the score function
NOTE: The customized score function should have 2 parameters: the logits, and the actual label
```
def accuracy(logits, labels):
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    pred_classes = np.argmax(logits * (1 / np.sum(logits, axis = -1)).reshape(logits.shape[0], 1), axis = -1)
    pred_classes = pred_classes.reshape(labels.shape)
    
    return np.sum(pred_classes == labels) / labels.shape[0]

es = EarlyStopping(mode = 'max', patience = 3, monitor = 'val_acc', out_path = 'model.bin')
es.monitor_score_function = accuracy
```
### Create and train the model
Calling the ```fit``` method, the training process will begin
```
model = Model(config, model_transformer, num_labels = 3)
model.to('cuda')
num_train_steps = int(len(train_dataset) / args.train_batch_size * args.epochs)
model.fit(args.epochs, args.learning_rate, num_train_steps, args.accumulation_steps, 
          train_dataloader, val_dataloader, es)
```



