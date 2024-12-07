""" Plot and Save results using matplotlib.pyplot. Saves to target_dir. Save:bool=False. Saving optional"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def plot_results(results_dict:Dict):
    num_epochs=len(results_dict["train_loss"])
    epochs=np.arange(1,num_epochs+1)
    figure=plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    # loss plot
    plt.plot(epochs,results_dict["train_loss"],label="Train Loss")
    plt.plot(epochs,results_dict["test_loss"], label="Test Loss")
    plt.legend(loc="best")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # acc plot
    plt.subplot(1,2,2)
    plt.plot(epochs,results_dict["train_acc"],label="Train Accuracy")
    plt.plot(epochs,results_dict["test_acc"],label="Test Accuracy")
    plt.legend(loc="best")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    return figure
