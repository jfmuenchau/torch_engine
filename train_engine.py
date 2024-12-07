from timeit import default_timer as timer
import time
import torch
import torchmetrics
from torch import nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model:nn.Module, train_dataloader:torch.utils.data.DataLoader, loss_fn:nn.Module, acc_fn:torchmetrics.Accuracy,
               optimizer:torch.optim, device:torch.device = device, show_batch=False):

    model.to(device)
    model.train()
    acc_fn.to(device)
    
    train_loss, train_acc = 0,0

    if show_batch:
        pbar=tqdm(total=len(train_dataloader))
    else:
        pass

    for batch,(X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        batch+=1

        y_logits = model(X)
        y_pred = torch.softmax(y_logits, dim = 1)
        y_pred_labels = torch.argmax(y_pred,dim=1)

        loss = loss_fn(y_logits, y)
        acc = acc_fn(y_pred_labels, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        train_acc += acc

        if show_batch:
            pbar.update(1)
        else:
            pass

    if show_batch:
        pbar.close()
    else:
        pass

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    return train_loss.item(), train_acc.item()

def test_step(model:nn.Module, test_dataloader:torch.utils.data.DataLoader, loss_fn:nn.Module, acc_fn:torchmetrics.Accuracy,
              device:torch.device = device):

    model.to(device)
    model.eval()
    acc_fn.to(device)
    
    test_loss, test_acc = 0,0

    for X,y in test_dataloader:
        X, y = X.to(device), y.to(device)
        
        with torch.inference_mode():
            logits = model(X)
            y_preds = torch.softmax(logits, dim=1)
            y_pred_labels = torch.argmax(y_preds, dim=1)

        loss = loss_fn(logits, y)
        acc = acc_fn(y_pred_labels, y)

        test_loss += loss
        test_acc += acc

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    return test_loss.item(), test_acc.item()

def train(model:nn.Module, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader,
          loss_fn:nn.Module, acc_fn:torchmetrics.Accuracy, optimizer:torch.optim, epochs:int = 5,  device:torch.device = device,
          writer=None, show_batch=False, lr_scheduler:torch.optim.lr_scheduler=None, track_epoch_time=False):

    results = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[], "train_epoch_time":[], "test_epoch_time":[]}
    start_time = timer()
    for epoch in range(1, epochs+1):
        
        if track_epoch_time:
            train_start=time.time()
        train_loss, train_acc = train_step(model=model, 
                                           train_dataloader=train_dataloader, 
                                           loss_fn = loss_fn, 
                                           acc_fn = acc_fn, 
                                           optimizer = optimizer, 
                                           device = device,
                                           show_batch=show_batch)
        
        if track_epoch_time:
            train_end=time.time()
            train_epoch_time=train_end-train_start
            test_start=time.time()

        test_loss, test_acc = test_step(model=model, 
                                        test_dataloader=test_dataloader, 
                                        loss_fn=loss_fn, 
                                        acc_fn=acc_fn, 
                                        device=device)
        if track_epoch_time:
            test_end=time.time()
            test_epoch_time=test_end-test_start
            print(f"Train epoch time: {train_epoch_time:.3f}s | Test epoch time: {test_epoch_time:.3f}s")
            results["train_epoch_time"].append(train_epoch_time)
            results["test_epoch_time"].append(test_epoch_time)
        
        if lr_scheduler:
            lr_scheduler.step()
        
        else:
            pass

        print(f"Epoch: {epoch} | train_loss: {train_loss:.3f} | train_acc: {train_acc:.3f} | test_loss: {test_loss:.3f} | test_acc: {test_acc:.3f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
            writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss":train_loss, "test_loss":test_loss}, global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train_acc":train_acc, "test_acc":test_acc}, global_step=epoch)
            writer.flush()
            writer.close()

        else:
            pass
    
    stop_time = timer()
    print(f"[INFO]\nDevice: {device}\nTraining time: {stop_time-start_time:.3f}s")
    
    return results