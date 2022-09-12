import PIL
from PIL import Image
import torch
from torch import nn, optim
from torchvision import models
from torchvision import transforms as T
import argparse
import json
import numpy as np


def get_args():
   
    
    parser = argparse.ArgumentParser(description = 'Pimage')
    parser.add_argument('image', type = str, help = 'path of image ')
    parser.add_argument('checkpoint', type = str, default='./checkpoint.pth', 
                        help = 'Enter location to save checkpoint')
    parser.add_argument('--top_k', type = int, default=5, 
                        help = 'choose number of top most classes to view, default=3.')
    parser.add_argument('--category_names', type = str, default='cat_to_name.json',
                        help = 'Choose of categories of names')
    parser.add_argument('--gpu', type = str, default='gpu', help = 'Useing the GPU')

    args = parser.parse_args()
    return args


        
def predict(image, model, topk, cat_to_name, device): 
    
    model.to(device); 
    model.eval();
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    image.unsqueeze_(0) 
    
    with torch.no_grad():
        image=image.to(device) 
        output = model.forward(image)
        pr = torch.exp(output)
        toppestprob, tclasses = pr.topk(topk) 
        nclasses = np.array(tclasses)
        idx_to_class = {s: e for e, s in model.class_to_idx.items()} 
        tclasses = [idx_to_class[i] for i in np.array(tclasses)[0]] 
    return np.array(toppestprob)[0] , [cat_to_name[i] for i in tclasses] 

def imageproc(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pic = Image.open(image)
    t = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
    transpic = t(pic)
    return (np.array(transpic))


def main():
    args = get_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    ## Use the GPU if available, otherwise use CPU
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'

    if args.gpu and not (torch.cuda.is_available()):
        device = 'cpu'
        print("GPU isn't available. CPU is chosen")
    else:
        device = 'cpu'

        
    checkpoint = torch.load(args.checkpoint)

    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    imagnp = imageproc(args.image)

    toppestprob, topestflower = predict(imagnp, model, args.top_k, cat_to_name, device)

    flor0=topestflower[0]
    pro0=toppestprob[0]
    print("Toppest --{}-- flower classes to predict ==> {}".format(args.top_k, topestflower))
    print("Toppest class of flower==> {}".format(flor0))
    print("Toppest --{}-- predict of values==> {}".format(args.top_k, toppestprob))
    print("acc ==> {:.3}".format(100 * pro0))
    
    
    

if __name__ == "__main__":
    main()