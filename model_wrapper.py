import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.metrics.functional import accuracy, f1
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import mlflow
from .utils import pack

class TrainingModel(pl.LightningModule):
  """
  Pytorch Lightning wrapper model
  """

  def __init__(self, experiment, **conf):
    super().__init__()
    self.model = experiment.get_model()
    self.experiment = experiment

  def forward(self, **inp):
    out = self.model(**inp)
    return out

  def get_dataloader(self, split, shuffle):
    """
    Creates the dataloader for the given split (['train', 'val'])
    """
    # dataset was created in Experiment.run
    dataset = self.experiment.dataset_const(split)
    dataloader = DataLoader(dataset, batch_size=self.experiment.batch_size,
                        shuffle=shuffle, collate_fn=self.experiment.batch_fn,
                        num_workers=self.experiment.num_workers)
    return dataloader

  def train_dataloader(self):
    return self.get_dataloader('train', True)

  def val_dataloader(self):
    return self.get_dataloader('val', False)

  def configure_optimizers(self):
    if self.experiment.lr is None:
        optimizer = self.experiment.optimizer_const(self.parameters())
    elif self.experiment.weight_decay is None:
        optimizer = self.experiment.optimizer_const(
                self.parameters(),
                lr=self.experiment.lr)
    else:
        optimizer = self.experiment.optimizer_const(
                self.parameters(),
                lr=self.experiment.lr,
                weight_decay=self.experiment.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=self.experiment.patience, verbose=True)
    return {
        'optimizer' : optimizer,
        'scheduler' : scheduler,
        'monitor'   : 'val_loss'
    }

  def base_step(self, batch):
    inp, target = pack(batch)
    out = self(**inp, labels=target)
    return out

  def training_step(self, batch, batch_nb):
    out = self.base_step(batch)
    loss = out.loss
    return loss
  
  def validation_step(self, batch, batch_nb):
    out = self.base_step(batch)
    loss = out.loss
    logits = out.logits.view(-1, out.logits.size(-1))
    # allow to produce self generated labels
    if 'labels' in out:
        labels = out.labels
    else:    
        labels = batch[-2]
    labels = labels.view(-1)
    mask = labels != -100
    pred = logits.softmax(-1).argmax(-1)
    pred = pred[mask]
    labels = labels[mask]
    true_pred = (pred == labels).sum()
    num_elems = labels.shape[0]
    return {
            'val_loss' : loss, 
            'true_pred' : true_pred, 
            'num_elems' : num_elems
           }
  
  def validation_epoch_end(self, outputs):
    # flatten for seq2seq models -> every sentence is weighted equally
    true_pred = sum([elem['true_pred'] for elem in outputs])
    num_elems = sum([elem['num_elems'] for elem in outputs])
    val_acc = true_pred / num_elems
    val_acc = val_acc.item()
    self.log('val_acc', val_acc)
    report = {'accuracy' : val_acc}
    mlflow.log_metrics(report)
