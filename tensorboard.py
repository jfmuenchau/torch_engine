import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def clear_tensorboard(path=r"C:\Users\JFM\AppData\Local\Temp\.tensorboard-info"):
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            print("Deleting file:", filename)
            os.remove(os.path.join(path, filename))


def create_summarywriter(model_name:str, extra:str=None):
    date=datetime.now().strftime("%Y-%m-%d")

    if extra:
        file_path=os.path.join("runs", date, model_name, extra)

    else:
        file_path=os.path.join("runs", date, model_name)

    if os.path.isdir(file_path):
        print("Directory already exists...")

    else:
        os.makedirs(file_path)

    print(f"[INFO] Created Summarywriter writing to {file_path}")

    return SummaryWriter(log_dir=file_path)
    
    
