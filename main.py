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

# SimpleCNNWithSalt（另一种salt,其实就是encode,decode)
def preprocess_data(data):
    data = data.transpose((0, 3, 1, 2))  # 将数据从 NHWC 转换为 NCHW
    return torch.tensor(data, dtype=torch.float32)


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
    summary(model, input_size=(args.batch_size, 3, 32, 32), device=args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    utils.train_test(args, model, train_loader, test_loader, criterion, optimizer, save_model=True)
