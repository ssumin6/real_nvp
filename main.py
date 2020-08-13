import os
import torch
import argparse

from utils import ToyDataset
from layers import Net
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

def train(net, optim, prior, x, device):
    x = x.to(device)
    x = 0.05 + (1-0.05)*x
    z, log_det_loss = net.forward(x)
    optim.zero_grad()
    loss = prior.log_prob(z)
    # print("prior loss %f" %loss.mean())
    loss += log_det_loss
    # print("added loss %f" %loss.mean())
    loss = -loss.mean()
    loss.backward()
    optim.step()
    return loss

def draw_plt(x, y, name):
    plt.figure()
    plt.plot(x.cpu(), y.cpu(), 'o')
    plt.title("Distribution of %s" %name)
    plt.savefig("%s.jpg" %name)

def test(net=None, prior=None, device=None, ckpt=None):
    assert net is not None or ckpt is not None
    if net is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device : %s" %(device))
        net = Net(N=3, input_dim=2, hidden_dim=256).to(device)
        prior = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
        ckpt = torch.load(ckpt)
        net.load_state_dict(ckpt['net'])
        print("Load checkpoint at Epoch %d." %(ckpt["epoch"]))

    with torch.no_grad():
        net.eval()
        d = prior.sample_n(128)
        pred_x, _ = net.forward(d, reverse=True)
        
    draw_plt(d[:, 0], d[:, 1], name="z")
    draw_plt(pred_x[:, 0], pred_x[:, 1], name="pred_x")

def draw_loss_graph(losses):
    plt.figure()
    plt.plot(losses)
    plt.title("Train Loss Graph")
    plt.savefig("loss_graph.png")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : %s" %(device))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_data = ToyDataset(args.path, transform=None)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # 2D Normal Distribution
    prior = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))

    net = Net(N=3, input_dim=2, hidden_dim=256).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    losses = []

    if args.load_model:
        ckpt = torch.load(args.save_path)
        net.load_state_dict(ckpt['net'])
        optim.load_state_dict(ckpt['optim'])

    for epoch in range(args.epochs):
        net.train()
        for _, x in enumerate(train_dataloader):
            loss = train(net, optim, prior, x, device)
            losses.append(loss)
        if (epoch != 0 and epoch % 10 == 0):
            avg_loss = sum(losses[-10:]) / 10.
            print("Epoch %d : Train Loss %f" %(epoch+1, avg_loss))
        if (epoch % args.save_epoch == 0):
            torch.save({'net': net.state_dict(),'optim': optim.state_dict(), 'epoch': epoch+1}, args.save_path)
    draw_loss_graph(losses)
    with torch.no_grad():
        x = next(iter(train_dataloader))
        x = x.to(device)
        z, _ = net.forward(x)
        pred_x, _ = net.forward(x, reverse=True)
        draw_plt(x[:,0], x[:, 1], name="train_x")
        draw_plt(z[:,0], z[:, 1], name="train_z")
        draw_plt(pred_x[:,0], pred_x[:, 1], name="train_pred_x")
    # Test
    test(net, prior, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NVP")
    parser.add_argument('--path', type=str, default='realnvp_toydata.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--save_epoch', type=int, default=25)
    parser.add_argument('--save_path', type=str, default='model.ckpt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()

    if not args.test:
        main(args)
    else:
        test(ckpt=args.save_path)