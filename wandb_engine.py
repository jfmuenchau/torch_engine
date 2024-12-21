import wandb
import torch
import torch.nn as nn
import torchmetrics
from torch_engine.train_engine import train_step, test_step
from typing import Dict
from tqdm import tqdm

device="cuda" if torch.cuda.is_available() else "cpu"

def train_wandb(model:nn.Module, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader,
                loss_fn:nn.Module, acc_fn:torchmetrics.Accuracy, optimizer:torch.optim, epochs:int, configs:Dict, device:torch.device = device,
                hide_batch=False, hide_epochs=False):
    """
    Function that will perform training and track loss and accuracy and upload it to the Weights & Biases website. Make sure to call !wandb login [api_key] before running the function.
      Returns: None
    model: model for training
    train_dataloader: the dataloader that is going to be used for training
    test_dataloader: the dataloader that is goingt to be used for testing
    loss_fn: loss function
    acc_fn: accuracy function
    optimizer: optimizer that optimizes the parameters of the model
    epochs: number of epochs
    configs: dictionary containing parameters like epochs, name of the dataset, model architecture or batch size
      example: configs=dict(epochs=5, "architecture"="CNN", "dataset"="MNIST", batch_size=128)

    device: device used for training
    hide_batch: hide the batch progressbar. Default=False
    hide_epochs: hide the epochs progressbar. Default=False
    """
                  
    hyperparameters=configs
    with wandb.init(project="pytorch-test", config=hyperparameters):
        wandb.watch(models=model, criterion=loss_fn, log="all", log_freq=1, log_graph=True)
        for epoch in tqdm(range(1,epochs+1), desc="Epoch",disable=hide_epochs):
            train_loss, train_acc = train_step(model=model, train_dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn, acc_fn=acc_fn,
                                               epoch=epoch, hide_batch=hide_batch)
            wandb_log(loss=train_loss, epoch=epoch, acc=train_acc)

            test_loss, test_acc = test_step(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, acc_fn=acc_fn, 
                                            epoch=epoch, hide_batch=hide_batch)
            

def wandb_log(loss, epoch, acc=None):
    """
    Function that logs epochs, loss and accuracy (optional) to upload to the Weights & Biases website.
    """
  
    if acc:
        wandb.log({"epoch":epoch, "loss":loss, "accuracy":acc})

    else:
        wandb.log({"epoch":epoch, "loss":loss})
