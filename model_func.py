import numpy as np

import torch
from torch import nn
from torch import optim

from torchvision import models

from collections import OrderedDict

import utility_func

def init_model(device, arch='vgg16', hidden_units=512, lr=0.001, print_model=True):
    # Function accepts two deep NN model architectures with pretrained weights
    if arch == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
    elif arch == 'densenet121':
        model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
    
    # Freeze model parameters to prevent backpropagation through them
    for para in model.parameters():
        para.requires_grad = False

    # VGG16 has 25088 in_features
    # DenseNet121 has 1024 in_features
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(model.classifier[0].in_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    # Log the model architecture when the flag is true
    if print_model:
        print(model)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    # Move the model to device (in this case GPU)
    model.to(device)

    return model, criterion, optimizer

def save_checkpoint(model, train_data, arch, hidden_units, lr, epochs, optimizer, save_dir):

    model.class_to_idx = train_data.class_to_idx

    torch.save({'arch': arch,
                'hidden_units': hidden_units,
                'learning_rate': lr,
                'epochs': epochs,
                'classifier': model.classifier,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                save_dir)
    
def load_checkpoint(filepath = 'checkpoint.pth'):
    checkpoint = torch.load(filepath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    lr = checkpoint['learning_rate']

    # Recreate the model
    model, _, _ = init_model(device, arch, hidden_units, lr, print_model=False)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()

    # run the PIL process image function and convert to numpy array of floats
    image = utility_func.process_image(image_path).numpy()
    image = torch.from_numpy(np.array([image])).float()

    with torch.no_grad():
        log_ps = model.forward(image.cuda())

    ps = torch.exp(log_ps).data

    return ps.topk(topk)