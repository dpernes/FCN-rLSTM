import argparse
import random

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import imgaug.augmenters as iaa

import plotter
from datasets import Trancos
from model import FCN_rLSTM
import np_transforms as NP_T


def main():
    parser = argparse.ArgumentParser(description='Train FCN_rLSTM in Trancos dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', default='./fcn_rlstm.pth', metavar='', help='model file (output of train)')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/TRANCOS_v3', metavar='', help='data directory path')
    parser.add_argument('--valid', default=0.2, type=float, metavar='', help='fraction of the training data for validation')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int, metavar='', help='batch size')
    parser.add_argument('--lambda', default=1., type=float, metavar='', help='trade-off between density estimation and vehicle count losses (eq. 7 in the paper)')
    parser.add_argument('--weight_decay', default=0., type=float, metavar='', help='weight decay regularization')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='FCN_rLSTM', metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, metavar='', help='Visdom port')
    parser.add_argument('--seed', default=42, metavar='', help='random seed')
    args = vars(parser.parse_args())

    # use a fixed random seed for reproducibility purposes
    random.seed(args['seed'])
    np.random.seed(seed=args['seed'])
    torch.manual_seed(args['seed'])

    # if we have a GPU and args['use_cuda'] == True, use the GPU; otherwise, use the CPU
    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu:0'

    # define image transformations to be applied to each image in the dataset
    train_transf = T.Compose([
        NP_T.RandomHorizontalFlip(0.5),  # data augmentation: horizontal flipping (we could add more transformations)
        T.ToTensor()  # convert np.array to tensor
    ])
    valid_transf = T.ToTensor()  # no data augmentation in validation

    # instantiate the dataset
    train_data = Trancos(train=True, path=args['data_path'], transform=train_transf)
    valid_data = Trancos(train=True, path=args['data_path'], transform=valid_transf)

    # split the data into training and validation sets
    if args['valid'] > 0:
        valid_indices = set(random.sample(range(len(train_data)), int(len(train_data)*args['valid'])))  # randomly choose some images for validation
        valid_data = Subset(valid_data, list(valid_indices))
        train_indices = set(range(len(train_data))) - valid_indices  # remaining indices are for training
        train_data = Subset(train_data, list(train_indices))
    else:
        valid_data = None

    # create data loaders for training and validation
    train_loader = DataLoader(train_data,
                              batch_size=args['batch_size'],
                              shuffle=True)  # shuffle the data at the beginning of each epoch
    if valid_data:
        valid_loader = DataLoader(valid_data,
                                  batch_size=args['batch_size'],
                                  shuffle=False)  # no need to shuffle in validation
    else:
        valid_loader = None

    # instantiate the model and define an optimizer
    model = FCN_rLSTM(temporal=False).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # Visdom is a tool to visualize plots during training
    if args['use_visdom']:
        plt = plotter.VisdomLinePlotter(env_name=args['visdom_env'],
                                        port=args['visdom_port'])

    # training routine
    for epoch in range(args['epochs']):
        print('Epoch {}/{}'.format(epoch, args['epochs']-1))

        # training phase
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        for i, (X, density, count) in enumerate(train_loader):
            # copy the tensors to GPU (if available)
            X = X.contiguous().to(device)
            density = density.to(device)
            count = count.to(device)

            # forward pass through the model
            density_pred, count_pred = model(X)
            # compute the loss
            N = X.shape[0]
            density_loss = torch.sum((density_pred - density)**2)/(2*N)
            count_loss = torch.sum((count_pred - count)**2)/(2*N)
            loss = density_loss + args['lambda']*count_loss
            # backward pass and optimization step
            optimizer.zero_grad()  # very important! (otherwise, gradients accumulate)
            loss.backward()
            optimizer.step()

            print('{}/{} mini-batch loss: {:.3f}'
                  .format(i, len(train_loader)-1, loss.item()),
                  flush=True, end='\r')
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())

        train_loss = sum(loss_hist)/len(loss_hist)
        train_density_loss = sum(density_loss_hist)/len(density_loss_hist)
        train_count_loss = sum(count_loss_hist)/len(count_loss_hist)
        print()
        print('Train loss: {:.3f}'.format(train_loss))
        print('Train density loss: {:.3f}'.format(train_density_loss))
        print('Train count loss: {:.3f}'.format(train_count_loss))

        if args['use_visdom']:
            plt.plot('mse', 'train', 'global loss', epoch, train_loss)
            plt.plot('mse', 'train', 'density loss', epoch, train_density_loss)
            plt.plot('mse', 'train', 'density loss', epoch, train_count_loss)

        if valid_loader is None:
            continue

        # validation phase
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        for i, (X, density, count) in enumerate(valid_loader):
            # copy the tensors to GPU
            X = X.to(device)
            density = density.to(device)
            count = count.to(device)

            # forward pass through the model
            with torch.no_grad():  # no need to compute gradients in validation (faster and uses less memory)
                density_pred, count_pred = model(X)

            # compute the loss
            N = X.shape[0]
            density_loss = torch.sum((density_pred - density)**2)/(2*N)
            count_loss = torch.sum((count_pred - count)**2)/(2*N)
            loss = density_loss + args['lambda']*count_loss

            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())

        valid_loss = sum(loss_hist)/len(loss_hist)
        valid_density_loss = sum(density_loss_hist)/len(density_loss_hist)
        valid_count_loss = sum(count_loss_hist)/len(count_loss_hist)
        print()
        print('Valid loss: {:.3f}'.format(valid_loss))
        print('Valid density loss: {:.3f}'.format(valid_density_loss))
        print('Valid count loss: {:.3f}'.format(valid_count_loss))
        print()

        if args['use_visdom']:
            plt.plot('mse', 'valid', 'global loss', epoch, valid_loss)
            plt.plot('mse', 'valid', 'density loss', epoch, valid_density_loss)
            plt.plot('mse', 'valid', 'density loss', epoch, valid_count_loss)

    torch.save(model.state_dict(), args['model_path'])


if __name__ == '__main__':
    main()
