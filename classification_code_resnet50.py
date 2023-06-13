import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def train_val_split(dataset, val_split=0.2, shuffle=True, random_seed=None):
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
    else:
        indices = list(range(len(dataset)))
    
    split = int(np.floor(val_split * len(dataset)))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    return train_sampler, val_sampler

def get_dataloaders(data_dir, image_size=(512, 512), batch_size=16, val_split=0.2, shuffle=True, random_seed=None):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
    }

    image_datasets = {
        x: torchvision.datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in ['train', 'val']
    }
    
    train_sampler, val_sampler = train_val_split(image_datasets['train'], val_split, shuffle, random_seed)
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=batch_size, sampler=train_sampler
        ),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'], batch_size=batch_size, sampler=val_sampler
        )
    }
    
    return dataloaders

def visualize_random_images(dataloader, classes):
    # Get a batch of training data
    inputs, labels = next(iter(dataloader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    plt.imshow(out.permute(1, 2, 0))
    plt.title([classes[label] for label in labels])
    plt.show()

dataloaders = get_dataloaders(data_dir)
visualize_random_images(dataloaders['train'], dataloaders['train'].dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
model = models.resnet50(pretrained=True)
print(model)

for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1024, 2),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
model.to(device)

epochs = 100
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        #print(logps, labels.view(-1, 1))
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'sicknessmodel.pth')

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average=None)
