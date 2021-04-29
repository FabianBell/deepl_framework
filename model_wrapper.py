import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.metrics.functional.classification import accuracy, f1_score
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import mlflow

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
    inp = batch[:-2]
    target = batch[-2]
    keys = batch[-1]
    inp = {k : v for k, v in zip(keys, inp)}
    out = self(**inp, labels=target)
    return out

  def training_step(self, batch, batch_nb):
    out = self.base_step(batch)
    loss = out.loss
    return loss
  
  def validation_step(self, batch, batch_nb):
    out = self.base_step(batch)
    loss = out.loss
    return {
            'val_loss' : loss, 
            'logits' : out.logits, 
            'labels' : batch[-2]
           }
  
  def validation_epoch_end(self, outputs):
    # flatten for seq2seq models -> every sentence is weighted equally
    logits = torch.cat([elem['logits'].view(-1, elem['logits'].size(-1)) for elem in outputs], 0)
    labels = torch.cat([elem['labels'].view(-1) for elem in outputs], 0)
    mask = labels != -100 # ignore padding labels
    logits = logits[mask]
    labels = labels[mask]
    pred = logits.softmax(-1).argmax(-1)
    num_classes = logits.shape[-1]
    val_acc = accuracy(pred, labels, num_classes=num_classes).item()
    self.log('val_acc', val_acc)
    report = {
        'accuracy' : val_acc,
        'micro_f1' : f1_score(pred, labels, num_classes=num_classes, class_reduction='micro').item(),
        'macro_f1' : f1_score(pred, labels, num_classes=num_classes, class_reduction='macro').item()
    }

    mlflow.log_metrics(report)
