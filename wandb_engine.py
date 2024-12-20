import wandb
import torch
import torch.nn as nn
import torchmetrics
from train_engine import train_step, test_step
from typing import Dict
from tqdm import tqdm

device="cuda" if torch.cuda.is_available() else "cpu"

def train_wandb(model:nn.Module, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader,
                loss_fn:nn.Module, acc_fn:torchmetrics.Accuracy, optimizer:torch.optim, epochs:int, configs:Dict, device:torch.device = device,
                hide_batch=False, hide_epochs=False):
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
    if acc:
        wandb.log({"epoch":epoch, "loss":loss, "accuracy":acc})

    else:
        wandb.log({"epoch":epoch, "loss":loss})
