import sys
import torch
import torch.nn as nn
from util.util import lr_scheduler  # Assuming lr_scheduler is still needed
from datasets.es_imagenet_new import ESImagenet_Dataset
import LIAF
from LIAFnet.LIAFResNet import *
import os
from tensorboardX import SummaryWriter

##################### Environment Preparation #####################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Choose a single GPU to use
save_folder = 'ES_exp_18_ES_new2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('runs/' + save_folder)

##################### Load Dataset #####################
config = LIAF.LIAFResNet.Config()
train_dataset = ESImagenet_Dataset(mode='train', data_set_path='/data/ES-imagenet-0.18/')
test_dataset = ESImagenet_Dataset(mode='test', data_set_path='/data/ES-imagenet-0.18/')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1, pin_memory=True)

##################### Model Setup #####################
snn = LIAFResNet(config)
snn.to(device)
print("Total number of parameters in networks is {}  ".format(sum(x.numel() for x in snn.parameters())))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=config.learning_rate)

##################### Training and Validation #####################
def val(snn, test_loader, batch_size):
    print('===> Evaluating models...')
    snn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = snn(inputs.to(device))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets.to(device)).sum().item()
    acc = 100. * correct / total
    writer.add_scalar('Accuracy', acc)
    print(f'Test Accuracy: {acc}%')
    return acc

for epoch in range(config.num_epochs):
    print(f'Starting epoch {epoch+1}/{config.num_epochs}')
    snn.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = snn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Loss: {running_loss/len(train_loader)}')

    val_acc = val(snn, test_loader, config.batch_size)
    print(f'Validation accuracy after epoch {epoch+1}: {val_acc}%')
    lr_scheduler(optimizer, epoch, config.learning_rate, 20)  # Adjust learning rate based on your lr_scheduler details
    writer.add_scalar('Loss/train', running_loss/len(train_loader), epoch)

