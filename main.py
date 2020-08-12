import os
import torch
import argparse

from utils import ToyDataset
from layers import Net
from torchvision import transforms
from torch.utils.data import DataLoader

def train(net, optim, criterion, x, device):
    x = x.to(device)
    z = net.forward(x)
    x_pred = net.reverse_forward(z)
    optim.zero_grad()
    loss = criterion(x_pred, x) # ADD determinant loss to this 
    loss.backward()
    optim.step()

    return loss

def test():
    # TODO
    pass

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : %s" %(device))

    preprocess = [
        transforms.ToTensor()
        ]
    train_transform = transforms.Compose(preprocess)

    train_data = ToyDataset(args.path, transform=train_transform)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    net = Net().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss()

    if args.load_model:
        ckpt = torch.load(args.save_path)
        net.load_state_dict(ckpt['net'])
        optim.load_state_dict(ckpt['optim'])
    
    for epoch in range(args.epoch):
        net.train()
        for x in enumerate(train_dataloader):
            loss = train(net, optim, criterion, x, device)

        if (epoch % args.log_epoch == 0):
            print("Epoch %d : Train Loss %f" %(epoch+1, loss))
        if (epoch % args.save_epoch == 0):
            torch.save({'net': net.state_dict(),'optim': optim.state_dict(), 'epoch': epoch+1}, args.save_path)
    
    # Test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NVP")
    parser.add_argument('--path', type=str, default='realnvp_toydata.csv')
    parser.add_argument('--epochs', type=int, default=100) # TODO : change value
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_epoch', type=int, default=5)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='model.ckpt')
    parser.add_argument('--batch_size', type=int, default=32) # TODO : change value
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()

    if not args.test:
        main(args)
    else:
        test()
        pass