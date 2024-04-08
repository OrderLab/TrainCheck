
import os
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import random
from tqdm import tqdm

import torch.optim as optim

import src.proxy_wrapper.proxy as Proxy
import logging

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2" 
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# Deterministic Behaviour
seed = 786
os.environ['PYTHONHASHSEED'] = str(seed)
## Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
## Python RNG
np.random.seed(seed)
random.seed(seed)

## CuDNN determinsim
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.cuda.empty_cache()


### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
data_transform = {'train':transforms.Compose([transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomRotation(30),
                                transforms.ColorJitter(brightness = 0, contrast = 0.5, saturation = 0.5, hue = 0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.2829, 0.2034, 0.1512],[0.2577, 0.1834, 0.1411])
                               ]),
                  'valid':transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.2829,0.2034,0.1512],[0.2577,0.1834,0.1411])
                                ]),
                 }


dir_file = 'dataset'
train_dir = os.path.join(dir_file,'train')
valid_dir = os.path.join(dir_file,'dev')
# image_dataset = {'train':datasets.ImageFolder(train_dir,transform = data_transform['train']),
#                 'valid':datasets.ImageFolder(valid_dir,transform = data_transform['valid']),
#                 }
train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transform['train'])
valid_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transform['valid'])

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, pin_memory=False, num_workers=1, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, pin_memory=False, num_workers=1, shuffle=False)


data_transfer = {'train':train_loader, 'valid':valid_loader}


# %%
import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 
# define VGG16 model
#model_transfer = models.resnet50(pretrained=True)

from efficientnet_pytorch import EfficientNet
model_transfer = EfficientNet.from_pretrained('efficientnet-b0')
n_inputs = model_transfer._fc.in_features

## Yuxuan Testing: Using CIFAR100, thus 100 classes
num_classes = 100
model_transfer._fc = nn.Sequential(
    nn.Linear(n_inputs, num_classes),
    nn.ReLU()
)
# %%
## Parallel GPU : https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
model_transfer = nn.DataParallel(model_transfer) # moved out from the above if statement

criterion_transfer = nn.CrossEntropyLoss() # moved out from the above if statement
model_transfer.to(device)   

model_transfer=Proxy.Proxy(model_transfer, "model_transfer-example.log", log_level = logging.INFO)
# %%
for name,param in model_transfer.module.named_parameters():
    total_nbn_layers = len([name for name, _ in model_transfer.named_parameters() if "bn" not in name])
    
    tunable_nbn_layers = total_nbn_layers // 4
    # tunable_nbn_layers = 5
    tunable_nbn_ind = total_nbn_layers - tunable_nbn_layers

    count = 0
    for name, param in model_transfer.named_parameters():
        if "bn" in name:
            param.requires_grad = False
        else:
            if count > tunable_nbn_ind:
                param.requires_grad = True
            else:
                param.requires_grad = False
            count += 1
            

for param in model_transfer.module._fc.parameters():
    param.requires_grad = False
    
print(model_transfer.module._fc[0].in_features)


# %%
for name, param in model_transfer.named_parameters():
    print(f'{name}: {param.requires_grad}')

# %%
input_size = (3, 224, 224)
from torchsummary import summary
summary(model_transfer, input_size)

# %%
use_cuda = torch.cuda.is_available()


# %%
## For adding multi-class accuracy
nb_classes = num_classes



# %%
# the following import is required for training to be robust to truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    os.makedirs(save_path, exist_ok=True)


    valid_loss_min = np.Inf 
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    res = []
    for epoch in tqdm(range(1, n_epochs+1), desc='Epochs'):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        correct = 0.0
        total = 0.0
        accuracy = 0.0
        

        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(loaders['train'], desc='Training')):
            # move to GPU
            if use_cuda:
                data, target = data.to('cuda',non_blocking = True), target.to('cuda',non_blocking = True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * (float(loss) - train_loss))

        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(tqdm(loaders['valid'], desc='Validation')):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output,target)
            valid_loss +=((1 / (batch_idx + 1)) * (float(loss) - valid_loss))
            del loss
            pred = output.data.max(1, keepdim=True)[1]
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

            
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        print()
        accuracy = 100. * (correct / total)
        print('\nValid Accuracy: %2d%% (%2d/%2d)' % (
        accuracy, correct, total))
        print()
        res.append({'Epoch': epoch, 'loss': train_loss,'valid_loss':valid_loss,'Valid_Accuracy': accuracy})
        print({'Epoch': epoch, 'loss': train_loss,'valid_loss':valid_loss,'Valid_Accuracy': accuracy})
        print()
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,valid_loss))
            torch.save(model.state_dict(), 'case_3_model.pt')
            valid_loss_min = valid_loss

        ## dump the confusion matrix and case_3_res.json for each epoch
        import json
        with open(os.path.join(save_path, f'case_{epoch}_res.json'), 'w') as fp:
            json.dump(res, fp)
        import pandas as pd
        df = pd.DataFrame(confusion_matrix.numpy())
        df.to_csv(os.path.join(save_path, f'case_{epoch}_confusion_matrix.csv'))
        del df

    return model, res



num_epochs = 40
lr = 0.01
optimizer_transfer = optim.Adam(filter(lambda p : p.requires_grad, model_transfer.parameters()), lr=lr)
# optimizer_transfer = optim.Adam(model_transfer.module._fc.parameters(),lr=lr)
model_transfer, res = train(num_epochs, data_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, f'results/{num_epochs}_{lr}')

# save model
torch.save(model_transfer.state_dict(), f'case_{num_epochs}_{lr}_model.pt')

# save res as json file
import json
with open('case_3_res.json', 'w') as fp:
    json.dump(res, fp)



