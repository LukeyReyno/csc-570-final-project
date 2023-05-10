import torch
import torchvision
import torch.nn as nn

from torchvision.models.googlenet import googlenet
from torchvision import datasets

def main():
    from torchvision.models import resnet18, ResNet18_Weights
    # here we load the pre-trained model
    #model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT)

    data_dir = 'data'

    train_dir = f"{data_dir}/hotdog-nothotdog-full/train/"
    test_dir = f"{data_dir}/hotdog-nothotdog-full/test/"
    train_classa_dir = f"{train_dir}/hotdog"
    train_classb_dir = f"{train_dir}/nothotdog"
    test_classa_dir = f"{test_dir}/hotdog"
    test_classb_dir = f"{test_dir}/nothotdog"
    
    import torchvision.transforms as transforms 

    # define augmentation pipelines
    train_tansform = transforms.Compose([
        # After running once, comment out this line and add in the following three lines.
        # Why does this make a difference?
        #transforms.Resize((64, 64)), 
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_ransform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    import random

    subset_size = 75
    training_batchsize = 50

    def random_indices(dataset, batch_size):
        return [random.choice(range(len(dataset))) for _ in range(batch_size)]

    train_dataset = datasets.ImageFolder(train_dir, train_tansform)

    train_dataset = torch.utils.data.Subset(train_dataset, random_indices(train_dataset, subset_size))

    test_dataset = datasets.ImageFolder(test_dir, test_ransform)
    test_dataset = torch.utils.data.Subset(test_dataset, random_indices(test_dataset, subset_size))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_batchsize, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False)

    """import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams.update({'font.size': 20})
    def imshow(input, title):
        # torch.Tensor => numpy
        input = input.numpy().transpose((1, 2, 0))
        # undo image normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input = std * input + mean
        input = np.clip(input, 0, 1)
        # display images
        plt.imshow(input)
        plt.title(title)
        plt.show()
    # load a batch of train image
    iterator = iter(train_dataloader)
    # visualize a batch of train image
    inputs, classes = next(iterator)
    out = torchvision.utils.make_grid(inputs[:training_batchsize])
    imshow(out, title="Image Grid")"""
    
    params_not_frozen = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_not_frozen.append(param)
    print("We have a lot of layers that are not frozen:",len(params_not_frozen))
    
    # Make sure our parameters are frozen
    for param in model.parameters():
        param.requires_grad = False
        

    num_features = model.fc.in_features     #extract fc layers features
    print('num_features',num_features)
    model.fc = nn.Linear(num_features, 2) #(num_of_class == 2) - here is the magic. 

    params_not_frozen = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_not_frozen.append(param)
    # print("Now only our last layer is not frozen:",len(params_not_frozen))
    
    def training_loop():
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()  #(set loss function)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        num_epochs = 50   #(set no of epochs)
        for epoch in range(num_epochs): #(loop for every epoch)
            print("Epoch {} running".format(epoch)) #(printing message)
            """ Training Phase """
            model.train()    #(training model)
            running_loss = 0.   #(set loss 0)
            running_corrects = 0 
            # load a batch data of images
            for i, (inputs, labels) in enumerate(train_dataloader):
                # forward inputs and get output
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # get loss value and update the network weights
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects / len(train_dataset) * 100.
            print('[Train #{}] Loss: {:.4f} Acc: {:.4f}%'.format(epoch, epoch_loss, epoch_acc))
            
            """ Testing Phase """
            model.eval()
            with torch.no_grad():
                running_loss = 0.
                running_corrects = 0
                for inputs, labels in test_dataloader:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / len(test_dataset)
                epoch_acc = running_corrects / len(test_dataset) * 100.
                print('[Test #{}] Loss: {:.4f} Acc: {:.4f}%'.format(epoch, epoch_loss, epoch_acc))

    training_loop()

if __name__ == '__main__':
    main()