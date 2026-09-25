import torch
import torch.nn.functional as F

def run_training(num_epochs, train_dataloader, net, optimizer):
    for epoch in range(num_epochs):
        for data in train_dataloader: 
            X, y = data 
            net.zero_grad() 
            output = net(X.view(-1, 3)) 
            loss = F.nll_loss(output, y) 
            loss.backward() 
            optimizer.step() 
        print(f"{epoch + 1} of {num_epochs} epochs - Loss: {loss}")

def run_test(test_dataloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            X, y = data
            output = net(X.view(-1,3))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    return round(correct/total, 3)