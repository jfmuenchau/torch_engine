import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def clear_tensorboard(path=r"C:\Users\JFM\AppData\Local\Temp\.tensorboard-info"):
    """Deletes all files in the path folder so that tensorboard can be run in Jupyter Notebooks. Once you start tensorboard a new file will be created in the given path.
    Trying to load tensorboard again after that file was created will result in an error."""
    
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            print("Deleting file:", filename)
            os.remove(os.path.join(path, filename))


def create_summarywriter(model_name:str, extra:str=None):
    """Creates a SummaryWrite from torch.utils.tensorboard.SummaryWriter that can be used to track parameters during training.
    * model_name: Name of the model thats going to be trained.
    * extra: Extra string to append to the model name. Will create a new folder in "runs/date/model_name/extra". The SummaryWriter will save to this folder.
    """
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
    
    
