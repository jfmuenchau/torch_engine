from pathlib import Path
import torch
import os

default_root = r"D:\Python\PyTorch\models"

def save_model(model:torch.nn.Module,model_name:str, path:str=default_root, extra:str=None, state_dict=True, jit_script=False):
    """Saves a model to the given path. The function makes sure that giving the same model name twice wont leed to losing older files by adding extra tag (n) n=1,2,3,4
    * model: Torch nn.Module that needs to be saved.
    * model_name: Name of the model sets name of the folder its gonna be saved to.
    * path: Path to save the model to. Default is set to "D:\Python\PyTorch\models"
    * extra: Extra string that is going to be attached to the file name.
        + file_name = model_name + extra
    * state_dict: Save the model.state_dict().
    * jit_script: Save a jit script of the model."""
    

    create_path = Path.joinpath(Path(path), Path(model_name))
    dir = make_dir(create_path)

    if state_dict:
        save_state_dict(model=model, path=dir.joinpath(dir.name), extra=extra)

    if jit_script:
        save_jit_script(model=model, path=dir.joinpath(dir.name), extra=extra)

def make_dir(dir:str):
    dir = Path(str(dir))
    if dir.is_dir():
        pass
    else:
        Path.mkdir(dir, parents=True, exist_ok=True)
        print(f"Created directory: '{dir}'")
    return dir

def save_jit_script(model:torch.nn.Module, path:str, extra:str=None):
    if extra:
        test_name = Path(str(path)+str(extra)+"_script.pt")
    else:
        test_name = Path(str(path)+"_script.pt")
    unique_name = uniquify(test_name)
    
    model_scripted = torch.jit.script(model)
    model_scripted.save(unique_name)
    print(f"[INFO] Saved model script to: {unique_name}")


def save_state_dict(model:torch.nn.Module, path:str, extra:str=None):
    if extra:
        test_name = Path(str(path)+str(extra)+".pt")
    else:
        test_name = Path(str(path)+".pt")

    unique_name = uniquify(test_name)
    
    torch.save(model.state_dict(), unique_name)
    print(f"[INFO] Saved model state_dict to: {unique_name}")


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename+"("+str(counter)+")"+extension
        counter += 1

    return path

 
###Loading models/state dicts using tkinter interface
