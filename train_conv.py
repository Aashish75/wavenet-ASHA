import logging
logger = logging.getLogger(__name__)

import pathlib
from typing import Any, Dict
from ruamel import yaml


import torch.nn as nn
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
import determined as det

from determined import pytorch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

from model_conv import NgramWaveNet 

class NGramWavenetTrial(PyTorchTrial):
    
    ##
    ## Dataset interface
    ##
    def build_vocab(self, corpus):
        chars = sorted(list(set(''.join(corpus))))
        self.stoi = {s:i+1 for i,s in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()}
        self.vocab_size = len(self.itos)

    def word2index(self, word_count):
        word_index = {w: i for i, w in enumerate(word_count)}
        idx_word = {i: w for i, w in enumerate(word_count)}
        return word_index, idx_word
    
    def build_dataset(self, batchname, words, block_size):  
        X, Y = [], []
        
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix] # crop and append

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        logger.info(f"Data for {batchname} is {X.shape, Y.shape}")
        return X, Y
            
    def __init__(self, context: pytorch.PyTorchTrialContext, hparams: Dict) -> None:
        self.context = context
        self.trainer = None
        
        # Get our sample names
        words = open('names.txt', 'r').read().splitlines()
        self.build_vocab(words)
        
        #
        # Block size is the number of character to consider at once
        # In our example, we'll usually use 8
        #
        self.block_size = hparams['block_size']

        n1 = int(0.8*len(words))
        n2 = int(0.9*len(words))
        Xtr,  Ytr  = self.build_dataset("train", words[:n1],  self.block_size)     # 80%
        self.dataset_train = torch.utils.data.TensorDataset(Xtr, Ytr)
        
        Xdev, Ydev = self.build_dataset("dev", words[n1:n2],self.block_size)   # 10%
        self.dataset_validate = torch.utils.data.TensorDataset(Xdev, Ydev)    

        Xte,  Yte  = self.build_dataset("valid", words[n2:],  self.block_size)     # 10%
        self.dataset_validate = torch.utils.data.TensorDataset(Xte, Yte)
        
        self.batch_size = 32
        self.per_slot_batch_size = self.batch_size // self.context.distributed.get_size()
        
        # Define loss function.
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Define model.
        """
        self.pytorch_model = model.build_model(vocab_size = self.vocab_size,
                                                               hparams=hparams)
        """
        self.model = NgramWaveNet(
            vocab_size= self.vocab_size,
            n_embed=hparams["n_embed"],
            block_size= hparams["block_size"],
            learning_rate=hparams["learning_rate"],
            debug=False,
        )
        logging.info(f"Model: {self.pytorch_model}")
        self.model = self.context.wrap_model(self.pytorch_model)

        # Configure optimizer.
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.SGD(self.model.parameters(), 
                                 lr=hparams["learning_rate"],
                                 momentum = hparams["lr_momentum"],
                                 weight_decay = hparams['lr_weight_decay']
            )
        )
        pass

    def train_batch(self, batch: det.pytorch.TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch_data, labels = batch

        logging.debug(f"batch_data: {batch_data.shape}, labels: {labels.shape}")
        logging.debug(f"batch_data: {batch_data}, labels: {labels}")

        output = self.model(batch_data)
        logging.debug(f"output: {output.shape}")
        logging.debug(f"output: {output}")

        loss = self.loss_fn(output, labels)
        if batch_idx % 1000 == 0:
            logging.info(f"epoch {epoch_idx}, batch {batch_idx}, loss {loss.item()}")

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {"loss": loss}

    def evaluate_batch(self, batch: det.pytorch.TorchData) -> Dict[str, torch.Tensor]:
        batch_data, labels = batch

        output = self.model(batch_data)
        validation_loss = self.loss_fn(output, labels).item()

        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(batch_data)

        return {"validation_loss": validation_loss, "accuracy": accuracy}

    def build_training_data_loader(self) -> pytorch.DataLoader:
        # Convert the dataset into a PyTorch DataLoader.
        return pytorch.DataLoader(self.dataset_train, 
                                 batch_size=self.per_slot_batch_size)

    def build_validation_data_loader(self) -> pytorch.DataLoader:
        # Convert the dataset into a PyTorch DataLoader.
        return pytorch.DataLoader(self.dataset_validate, 
                                    batch_size=self.per_slot_batch_size)
    

def run(local: bool = False):
    """Initializes the trial and runs the training loop.

    This method configures the appropriate training parameters for both local and on-cluster
    training modes. It is an example of a standalone training script that can run both locally and
    on-cluster without any code changes.

    To run the training code solely locally or on-cluster, remove the conditional parameter logic
    for the unneeded training mode.

    Arguments:
        local: Whether to run this script locally. Defaults to false (on-cluster training).
    """

    info = det.get_cluster_info()

    if local:
        print(f"Running locally")
        # For convenience, use hparams from const.yaml for local mode.
        safeyaml = yaml.YAML(typ="safe", pure=True)
        conf = safeyaml.load(pathlib.Path("./inclass.yaml").read_text())
        hparams = conf["hyperparameters"]
        max_length = pytorch.Batch(15000)  # Train for 1000 batches.
        latest_checkpoint = None
    else:
        hparams = info.trial.hparams  # Get instance of hparam values from Determined cluster info.
        max_length = None  # On-cluster training trains for the searcher's configured length.
        latest_checkpoint = (
            info.latest_checkpoint
        )  # (Optional) Configure checkpoint for pause/resume functionality.

    with pytorch.init() as train_context:
        trial = NGramWavenetTrial(train_context, hparams=hparams)
        trainer = pytorch.Trainer(trial, train_context)
        trial.trainer = trainer ## for reports
        trainer.fit(max_length=max_length, latest_checkpoint=latest_checkpoint)


if __name__ == "__main__":
    print("Starting training process.")
    # Configure logging
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)

    local_training = det.get_cluster_info() is None
    logger.info(f"Local training: {local_training}")
    run(local=local_training)