#_# Python Libraries 
import os
import argparse
import numpy as np
import torch
import torchvision
from torchinfo import summary
import torch.optim as optim
import torch.nn as nn

#_# Our Source Code
from src import datasets, utils
from src.models import simple_cnn, wide_resnet
import exp_setup

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from PIL import Image
from torchvision.datasets import CIFAR10
from simple_cnn import simplecnnwithEncoderDecoder

# SimpleCNNWithSalt（另一种salt,其实就是encode,decode)
def preprocess_data(data, transform):
    data = [transform(Image.fromarray(image.astype(np.uint8))) for image in data]
    return data

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(test_loader), 100 * correct / total
"""
if __name__ == "__main__":

    args = exp_setup.get_parser().parse_args()

    args.device = "cpu"
    if torch.cuda.is_available():
        args.device = "cuda"        
        torch.cuda.set_device(args.gpu_id)
    print(torch.cuda.get_device_properties(args.device))
    
    if args.reproducibility:
        torch.manual_seed(args.rand_seed)
        torch.cuda.manual_seed(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.rand_seed)

    if args.split == 0:
        train_data, train_labels,  test_data,  test_labels = datasets.get_dataset(args, verbose=1)
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)

        train_dataset = TensorDataset(train_data, torch.tensor(train_labels, dtype=torch.long))
        test_dataset = TensorDataset(test_data, torch.tensor(test_labels, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        
        #dataset = ((train_data, train_labels),(None,None), (test_data,  test_labels))      
    else:
        raise Exception("Setting of split == 1 is not implemented in this version")    
        # train_data, train_labels, valid_data, valid_labels, test_data,  test_labels = datasets.get_dataset(args, verbose=1)
        # dataset = ((train_data, train_labels), (valid_data, valid_labels), (test_data,  test_labels))  
    
    model = SimpleCNNWithSalt(num_classes=args.num_classes, salt_dim=args.salt_dim).to(args.device)   
    
    # For LeNet
    #model = simple_cnn.SimpleCNN(num_classes=args.num_classes, salt_layer=args.salt_layer,
    #                            mean =  datasets.CIFAR10_MEAN, 
    #                            std = datasets.CIFAR10_STD, 
    #                            num_input_channels=args.num_input_channels)

    ## For WideResNet
    # model = wide_resnet.WideResNet(num_classes = args.num_classes,
    #                                     width = 3, 
    #                                     mean =  datasets.CIFAR10_MEAN, 
    #                                     std = datasets.CIFAR10_STD, 
    #                                     num_input_channels=args.num_input_channels)
    ## For SaltedWideResNet
    # model = wide_resnet.SaltyWideResNet(num_classes = args.num_classes,
    #                                     width = 3, 
    #                                     mean =  datasets.CIFAR10_MEAN, 
    #                                     std = datasets.CIFAR10_STD, 
    #                                     num_input_channels=args.num_input_channels)

    ## For ConvNet (PAMAP2)
    # model = simple_cnn.SenNet(salt_layer=args.salt_layer)

    # wideResnet、salt_cnn（encode_decode）
    """
"""
    model.to(args.device)
    if args.dataset == "cifar10":
        if args.salt_layer == -1:
            summary(model, [(1, args.num_input_channels, 32, 32)], device=args.device)       
        elif 0<= args.salt_layer <=5: 
            summary(model, [(1, args.num_input_channels, 32, 32),(1,1,1,1)], device=args.device)       
        else:
            summary(model, [(1, args.num_input_channels, 32, 32),(1,args.num_classes)], device=args.device)       
    elif args.dataset == "pamap":
        summary(model, [(1, 1, 27, 200),(1,1,1,1)], device=args.device)  
    
    utils.train_test(args, model, dataset, save_model=True)
    """
"""
    summary(model, input_size=(args.batch_size, 3, 32, 32), device=args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    utils.train_test(args, model, train_loader, test_loader, criterion, optimizer, save_model=True)
"""
def main():
    args = exp_setup.get_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    datasets_map = {
        "cifar10": datasets.eval_datasets,
        "pamap": datasets.eval_datasets,
    }

    if args.dataset not in datasets_map:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset_func = datasets_map[args.dataset]
    if args.split == 0:
        train_data, train_labels, test_data, test_labels = dataset_func(args)
    else:
        train_data, train_labels, valid_data, valid_labels, test_data, test_labels = dataset_func(args)

    train_data = preprocess_data(train_data, transform)
    test_data = preprocess_data(test_data, transform)

    train_dataset = TensorDataset(torch.stack(train_data), torch.tensor(train_labels))
    test_dataset = TensorDataset(torch.stack(test_data), torch.tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = simple_cnn(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_model(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
