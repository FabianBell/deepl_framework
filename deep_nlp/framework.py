"""
@package Framework for experiments and training of neural networks that follow the hugging face model style
"""
from .utils import unpack, pack
from .model_wrapper import TrainingModel
from copy import deepcopy
import mlflow
import os
from abc import ABC, abstractmethod
from torch.optim import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from transformers import logging
logging.set_verbosity_error()


DEFAULT_ARGS = {
    'return_dict': True,
}

OUTPUT_DIR = 'save'

mlflow.set_tracking_uri("mlruns")


class Experiment(ABC):
    """
    Configuration object of a hugging face experiment.
    """

    def __init__(self, batch_size, num_epochs, dataset_const, lr=None, weight_decay=None,
                 num_workers=4, optimizer_const=AdamW, patience=2,
                 accumulate_grad_batches=1, check_val_every_n_epoch=1,
                 gpus=None, name=None):
        self.name = name
        self.lr = lr
        self.num_workers = num_workers
        self.optimizer_const = optimizer_const
        self.batch_size = batch_size
        self.patience = patience
        self.num_epochs = num_epochs
        self.accumulate_grad_batches = accumulate_grad_batches
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.gpus = gpus
        self.dataset_const = dataset_const
        self.weight_decay = weight_decay

    @abstractmethod
    def get_model(self):
        """
        Get the model that has to be trained.
        """
        pass

    @abstractmethod
    def get_tokenizer(self):
        """
        Get the tokenizer for dynamic or static preprocessing.
        """
        pass
    
    @abstractmethod
    def batch_fn(self, batch):
        """
        Combines entries to a batch
        """
        pass

    def _get_params(self):
        """
        Prepares the parameter for logging
        """
        params = deepcopy(self.__dict__)
        params['batch_fn'] = self.batch_fn.__name__
        params['lr'] = 'optimizer default' if self.lr is None else self.lr
        params['optimizer_const'] = self.optimizer_const.__name__
        params['tokenizer'] = self.tokenizer.__class__.__name__
        return params
    
    def run(self):
        """
        Runs the experiment
        """
        # init tokenizer
        self.tokenizer = self.get_tokenizer()
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath='pl_checkpoints',
            filename='model-{epoch:02d}-{val_acc:.4f}',
            save_top_k=3,
            mode='max'
        )
        trainer = pl.Trainer(
            max_epochs=self.num_epochs,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            gpus=self.gpus,
            accumulate_grad_batches=self.accumulate_grad_batches,
            logger=None,
            accelerator='dp' if self.gpus is not None else None,
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=0,  # avoid logging of accuracy of initialized model
        )
        model = TrainingModel(self)
        mlflow.set_experiment(model.model.__class__.__name__)
        with mlflow.start_run(run_name=self.name):
            mlflow.log_params(self._get_params())
            trainer.fit(model)
            if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)
            model_path = os.path.join(
                OUTPUT_DIR, f'model_{model.model.__class__.__name__}.pt')
