from model.Multitask import *
import torch
import torch.nn as nn
from tqdm import tqdm 
from torchvision import transforms
import warnings
import numpy as np
if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = 'cuda'
else:
    print("Using the CPU. Overall speed may be slowed down")
    device = 'cpu'

model = MultiTaskModel().to(device)
criterion = MultiTaskLossWrapper(2).to(device)
learning_rate = 1e-3
weight_decay = 1e-5
num_epoch = 10  # TODO: Choose an appropriate number of training epochs
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
weight_decay=weight_decay) # Try different optimizers

def train(model, trainloader, valloader, num_epoch=10):  # Train the model
    
    print("Start training...")
    trn_loss_hist = []
    trn_acc_hist = []
    val_acc_hist = []
    model.train()  # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        print('-----------------Epoch = %d-----------------' % (i+1))
        for batch, label in tqdm(trainloader):
            batch = batch.to(device)
            label0 = label[0].to(device)
            label1 = label[1].to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            # This will call Network.forward() that you implement
            pred = model(batch)
            # label0.float().requires_grad=True
            # label1.float().requires_grad=True

            loss = criterion(pred, label0,label1)
            print("fff")
            print(loss.gradients)
            loss.requres_grad = True  # Calculate the loss
            running_loss.append(loss.item())
            
            loss.backward()  # Backprop gradients to all tensors in the network
            optimizer.step()  # Update trainable weights
        print("\n Epoch {} loss:{}".format(i+1, np.mean(running_loss)))

        # Keep track of training loss, accuracy, and validation loss
        trn_loss_hist.append(np.mean(running_loss))
        trn_acc_hist.append(evaluate(model, trainloader))
        print("\n Evaluate on validation set...")
        val_acc_hist.append(evaluate(model, valloader))
    print("Done!")
    return trn_loss_hist, trn_acc_hist, val_acc_hist


def evaluate(model, loader):  # Evaluate accuracy on validation / test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    # with torch.no_grad():  # Do not calculate grident to speed up computation
    #     for batch, label in tqdm(loader):
    #         batch = batch.to(device)
    #         label = label.to(device)
    #         pred = model(batch)
    #         correct += (torch.argmax(pred, dim=1) == label).sum().item()
    #     acc = correct/len(loader.dataset)
    #     print("\n Evaluation accuracy: {}".format(acc))
    #     return acc
    #     #with torch.no_grad():  # Do not calculate grident to speed up computation
    for batch, label in tqdm(loader):
        batch = batch.to(device)
        label = label.to(device)
        pred = model(batch)
        correct += (torch.argmax(pred, dim=1) == label).sum().item()
    acc = correct/len(loader.dataset)
    print("\n Evaluation accuracy: {}".format(acc))
    return acc


# trn_loss_hist, trn_acc_hist, val_acc_hist = train(model, trainloader,
#                                                   valloader, num_epoch)

# ##############################################################################
# # TODO: Note down the evaluation accuracy on test set                        #
# ##############################################################################
# print("\n Evaluate on test set")
# evaluate(model, testloader)