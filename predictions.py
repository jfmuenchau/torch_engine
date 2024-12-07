import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple,List
import matplotlib.pyplot as plt

device="cuda" if torch.cuda.is_available() else "cpu"

def pred_plot_image(model:torch.nn.Module,img_path:str,class_names:List[str],img_size:Tuple=None, transform:transforms=None, device:torch.device=device):
    
    img=Image.open(img_path)
    
    if transform:
        transformed_img=transform(img)

    else:
        transform = torchvision.transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor])
        transformed_img= transform(img)

    transformed_img=transformed_img.to(device)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        y_logits=model(transformed_img.unsqueeze(dim=0))
        
    y_pred = y_logits.softmax(dim=1)
    y_pred_label=y_pred.argmax(dim=1)

    plt.imshow(img)
    plt.axis(False)
    plt.title(f"{class_names[y_pred_label]}: {y_pred.max():.3f}")
