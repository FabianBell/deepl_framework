# NLP Deep Learning Framework
This is a deepl learning framework for classification and seq2seq tasks.

## Installation
    pip install deep-nlp

## Example Project

### Structure 
    ├── data              --> containing the trainings and validation data
    |   ├── train.csv     --> training dataset
    |   └── val.csv       --> validation dataset
    ├── Experiment.py     --> containing the model and training logic
    └── dataset.py        --> containing the Dataset object 
    
### Dataset.py
```python
from torch.utils.data import Dataset
import pandas as pd

class ExampleDataset(Dataset):

  def __init__(self, split : str):
    self.data = pd.read_csv(f'{split}.csv')
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data.iloc[idx]
```
### Experiment.py
```python
from deep_nlp import Experiment, unpack
from dataset import ExampleDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

class ClassificationExperiment(Experiment):
  
  def get_tokenizer(self):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    return tokenizer
  
  def get_model(self):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    return model
    
  def batch_fn(self, batch):
    source, target = zip(*batch)
    source_inp = self.tokenizer(source, padding=True, return_tensors=True)
    target = torch.tensor(target)
    return unpack(source_inp, target)
  
def run_experiment():
    experiment = ClassificationExperiment(
        80,  # batch size
        20,  # number of epochs
        ExampleDataset,
        gpus=-1,  # use all available gpus
        lr=2.65e-5,
        weight_decay=4e-3,
        name='example_run'  # name for mlflow
    )
    experiment.run()

if __name__ == '__main__':
    run_experiment()
```
