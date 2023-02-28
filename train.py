import argparse
import sys
import torch

import utility_func
import model_func

# Optional: Allows the argparse descriptions to be displayed in a single line
# https://stackoverflow.com/questions/52605094/python-argparse-increase-space-between-parameter-and-description#comment98774195_52606755
formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=52)

# Create the argparser
parser = argparse.ArgumentParser(
    formatter_class=formatter,
    prog = 'Train Network',
    description = 'Trains a neural network model',
)

# Declare parser arguments
parser.add_argument('data_dir', default='./flower_data/', help="Path of data directory", metavar="DIR_PATH")
parser.add_argument('-sd', '--save_dir', default='./checkpoint.pth', help="Path of checkpoint.pth", metavar="CP_PATH")
parser.add_argument('-a', '--arch', default='vgg16', help="Select pre-trained model architecture. Default = vgg16")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Set learning rate hyperparameter. Default = 0.001", metavar="RATE")
parser.add_argument('-hu', '--hidden_units', type=int, default=2048, help="Number of hidden units. Default = 2048", metavar="UNITS")
parser.add_argument('-e', '--epochs', type=int, default=3, help="Number of training epochs. Default = 3")
parser.add_argument('--gpu', action='store_true', help="Use GPU (CUDA) for training")

# If no arguments are passed then display command syntax
args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

# Process the arguments data
data_dir = args.data_dir
save_dir = args.save_dir
lr = args.learning_rate
arch = args.arch
hidden_units = args.hidden_units
epochs = args.epochs

# returns True if the flag is raised else False
use_gpu = args.gpu

# Print commands to test argparse data handling
# print(data_dir)
# print(save_dir)
# print(lr)
# print(arch)
# print(hu)
# print(epochs)
# print(use_gpu)

# End of argparse

# Check if --gpu flag is raised and CUDA is available
# and hence set torch device accordingly
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Start of training function
def training():

    # Load the training dataset
    train_data, trainloader, validloader, testloader = utility_func.load_data(data_dir)

    # Load model architecture
    model, criterion, optimizer = model_func.init_model(device, arch, hidden_units, lr)

    # Train the model
    # epochs = 3
    steps = 0
    running_loss = 0
    print_every = 5

    print("Training is now starting")
    print("Architecture:\t" + arch)
    print("Hidden Units:\t" + str(hidden_units))
    print("Epochs:\t\t" + str(epochs))
    print("Learning Rate:\t" + str(lr))
    print("GPU Enabled:\t" + str(use_gpu))

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Accuracy: {accuracy/len(validloader):.3f}")
                
                running_loss = 0
                model.train()
    
    print("Training completed")

    # TODO: Save the checkpoint
    model_func.save_chkpoint(model, train_data, arch, hidden_units, lr, epochs,
                             optimizer, save_dir)
    
    print("Saved checkpoint")

# Python idiom: Execute training() function only when script is directly executed from shell
if __name__ == "__main__":
    training()