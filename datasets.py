from pathlib import Path
import shutil
import random
from tqdm import tqdm

# Dependencies for create_dataloaders
from torchvision.datasets import ImageFolder
import torchvision.transforms
import torch


def make_dir(path:str):
    path=Path(path)
    if path.is_dir():
        pass
    else:
        Path.mkdir(path, parents=True, exist_ok=True)
    return path

def testfor_dir(path:str):
    path=Path(path)
    
    if path.is_dir():
        print(f"[INFO] {path} already exists.")
        return False

    else:
        return make_dir(path)

def create_smaller_dataset(source:str, destination:str, num_classes:int, slice_at:float=0.8):
    """Creates smaller dataset based on source. Picking num_classes random classes and storing the content into train and test directories.
    * source: Path to the source directory that stores the data.
    * destination: Desired directory to save the smaller dataset.
    * num_classes: Number of random selected classes for the smaller dataset.
    * slice_at: Float that sets the split size. For example 0.8 will put 80% of the data inside the directory into the training set."""
    
    class_paths=list(Path(source).glob("*"))
    samples = random.sample(class_paths, k=num_classes)

    target_dir=testfor_dir(destination)
    
    if target_dir==False:
        train_dir = Path(destination).joinpath("train")
        test_dir= Path(destination).joinpath("test")
        return train_dir, test_dir
        
    train_dir=target_dir.joinpath("train")
    test_dir=target_dir.joinpath("test")
    
    print(f"[INFO] Copying files to {target_dir}")
    for sample in tqdm(samples):
        img_list=list(sample.glob("*.jpg"))
        train_list=img_list[:int(slice_at*len(img_list))]
        test_list=img_list[int(slice_at*len(img_list)):]
            
        for train_sample in train_list:
            train_sample_path=train_dir.joinpath(sample.name, train_sample.name)
            make_dir(train_sample_path.parent)
            shutil.copy(train_sample, train_sample_path)
                    
        for test_sample in test_list:
            test_sample_path=test_dir.joinpath(sample.name, test_sample.name)
            make_dir(test_sample_path.parent)
            shutil.copy(test_sample, test_sample_path)
    print(f"[INFO] Created Dataset")
    return train_dir, test_dir

def create_dataloaders(train_dir:str, test_dir:str, transform:torchvision.transforms=None, batch_size:int=32):
    """Creates dataloaders from the given train_dir and test_dir with batch size = batch_size. Applies a given transform and returns the train dataloader, 
    the test dataloader and the classes in form of a list.
    * train_dir: Path to the train folder. Following structure is desired: train/classes/data
    * test_dir: Path to the test folder. Following structure is desired: test/classes/data
    * transform: Gives a transform to be performed. If no transform is selected only ToTensor() will be performed.
    * batch_size: Batch size of the train and test dataloader. Standard is 32"""
    
    if not transform:
        transform=torchvision.transforms.ToTensor()
    train_dataset=ImageFolder(root=train_dir, transform=transform)
    test_dataset=ImageFolder(root=test_dir, transform=transform)
    classes=train_dataset.classes

    train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader, classes
