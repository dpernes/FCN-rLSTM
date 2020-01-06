import argparse
import random
import time

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import np_transforms as NP_T
import plotter
from datasets import TrancosSeq
from model import FCN_rLSTM
from utils import show_images, sort_seqs_by_len


def main():
    parser = argparse.ArgumentParser(description='Train FCN-rLSTM in Trancos dataset (sequential version).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_path', default='./fcn_rlstm.pth', type=str, metavar='', help='model file (output of train)')
    parser.add_argument('-d', '--data_path', default='/ctm-hdd-pool01/DB/TRANCOS_v3', type=str, metavar='', help='data directory path')
    parser.add_argument('--valid', default=0.2, type=float, metavar='', help='fraction of the training data for validation')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='', help='learning rate')
    parser.add_argument('--epochs', default=500, type=int, metavar='', help='number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, metavar='', help='batch size')
    parser.add_argument('--size_red', default=4, type=int, metavar='', help='size reduction factor to be applied to the images')
    parser.add_argument('--lambda', default=1e-4, type=float, metavar='', help='trade-off between density estimation and vehicle count losses (see eq. 7 in the paper)')
    parser.add_argument('--gamma', default=1e3, type=float, metavar='', help='precision parameter of the Gaussian kernel (inverse of variance)')
    parser.add_argument('--max_len', default=5, type=int, metavar='', help='maximum sequence length')
    parser.add_argument('--weight_decay', default=0., type=float, metavar='', help='weight decay regularization')
    parser.add_argument('--use_cuda', default=True, type=int, metavar='', help='use CUDA capable GPU')
    parser.add_argument('--use_visdom', default=False, type=int, metavar='', help='use Visdom to visualize plots')
    parser.add_argument('--visdom_env', default='FCN-rLSTM_train', type=str, metavar='', help='Visdom environment name')
    parser.add_argument('--visdom_port', default=8888, type=int, metavar='', help='Visdom port')
    parser.add_argument('--n2show', default=5, type=int, metavar='', help='number of examples to show in Visdom in each epoch')
    parser.add_argument('--vis_shape', nargs=2, default=[120, 160], type=int, metavar='', help='shape of the images shown in Visdom')
    parser.add_argument('--seed', default=42, type=int, metavar='', help='random seed')
    args = vars(parser.parse_args())

    # dump args to a txt file for your records
    with open(args['model_path'] + '.txt', 'w') as f:
        f.write(str(args)+'\n')

    # use a fixed random seed for reproducibility purposes
    if args['seed'] > 0:
        random.seed(args['seed'])
        np.random.seed(seed=args['seed'])
        torch.manual_seed(args['seed'])

    # if args['use_cuda'] == True and we have a GPU, use the GPU; otherwise, use the CPU
    device = 'cuda:0' if (args['use_cuda'] and torch.cuda.is_available()) else 'cpu:0'
    print('device:', device)

    # define image transformations to be applied to each image in the dataset
    train_transf = T.Compose([
        NP_T.RandomHorizontalFlip(0.5, keep_state=True),  # data augmentation: horizontal flipping (we could add more transformations)
        NP_T.ToTensor()  # convert np.array to tensor
    ])
    valid_transf = NP_T.ToTensor()  # no data augmentation in validation

    # instantiate the dataset
    train_data = TrancosSeq(train=True, path=args['data_path'], size_red=args['size_red'], transform=train_transf, gamma=args['gamma'], max_len=args['max_len'])
    valid_data = TrancosSeq(train=True, path=args['data_path'], size_red=args['size_red'], transform=valid_transf, gamma=args['gamma'], max_len=args['max_len'])

    # split the data into training and validation sets
    if args['valid'] > 0:
        valid_indices = set(random.sample(range(len(train_data)), int(len(train_data)*args['valid'])))  # randomly choose some images for validation
        valid_data = Subset(valid_data, list(valid_indices))
        train_indices = set(range(len(train_data))) - valid_indices  # remaining images are for training
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
    model = FCN_rLSTM(temporal=True, image_dim=(train_data[0][0].shape[2:])).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # Visdom is a tool to visualize plots during training
    if args['use_visdom']:
        loss_plt = plotter.VisdomLossPlotter(env_name=args['visdom_env'],
                                             port=args['visdom_port'])
        img_plt = plotter.VisdomImgsPlotter(env_name=args['visdom_env'],
                                            port=args['visdom_port'])

    # training routine
    for epoch in range(args['epochs']):
        print('Epoch {}/{}'.format(epoch, args['epochs']-1))

        # training phase
        model.train()  # set model to training mode (affects batchnorm and dropout, if present)
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        X, mask, density, count = None, None, None, None
        t0 = time.time()
        for i, (X, mask, density, count, _, seq_len) in enumerate(train_loader):
            # copy the tensors to GPU (if applicable)
            X, mask, density, count, seq_len = X.to(device), mask.to(device), density.to(device), count.to(device), seq_len.to(device)
            # sort the sequences by descending order of the respective lengths (as expected by the model)
            seqs, seq_len = sort_seqs_by_len([X, mask, density, count], seq_len)
            X, mask, density, count = seqs
            # transpose them so they have shape (seq_len, batch_size, *) (as expected by the model)
            X, mask, density, count = X.transpose(1, 0), mask.transpose(1, 0), density.transpose(1, 0), count.transpose(1, 0)

            # forward pass through the model
            density_pred, count_pred = model(X, mask=mask, lengths=seq_len)

            # compute the loss
            N = torch.sum(seq_len)
            density_loss = torch.sum((density_pred - density)**2)/(2*N)
            count_loss = torch.sum((count_pred - count)**2)/(2*N)
            loss = density_loss + args['lambda']*count_loss

            # backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('{}/{} mini-batch loss: {:.3f} | density loss: {:.3f} | count loss: {:.3f}'
                  .format(i, len(train_loader)-1, loss.item(), density_loss.item(), count_loss.item()),
                  flush=True, end='\r')

            # save the loss values
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())
            with torch.no_grad():  # evaluation metric, so no need to compute gradients
                count_err = torch.sum(torch.abs(count_pred - count))/N
            count_err_hist.append(count_err.item())
        t1 = time.time()
        print()

        # print the average training losses
        train_loss = sum(loss_hist)/len(loss_hist)
        train_density_loss = sum(density_loss_hist)/len(density_loss_hist)
        train_count_loss = sum(count_loss_hist)/len(count_loss_hist)
        train_count_err = sum(count_err_hist)/len(count_err_hist)
        print('Training statistics:')
        print('global loss: {:.3f} | density loss: {:.3f} | count loss: {:.3f} | count error: {:.3f}'
              .format(train_loss, train_density_loss, train_count_loss, train_count_err))
        print('time: {:.0f} seconds'.format(t1-t0))

        if args['use_visdom']:
            # plot the losses
            loss_plt.plot('global loss', 'train', 'MSE', epoch, train_loss)
            loss_plt.plot('density loss', 'train', 'MSE', epoch, train_density_loss)
            loss_plt.plot('count loss', 'train', 'MSE', epoch, train_count_loss)
            loss_plt.plot('count error', 'train', 'MAE', epoch, train_count_err)

            # show a few training examples (images + density maps)
            X *= mask  # show the active region only
            X, mask, density, count = X.transpose(1, 0), mask.transpose(1, 0), density.transpose(1, 0), count.transpose(1, 0)
            density_pred, count_pred = density_pred.transpose(1, 0), count_pred.transpose(1, 0)
            N, L, C, H, W = X.shape
            X, density, count = X.reshape(N*L, C, H, W).cpu().numpy(), density.reshape(N*L, 1, H, W).cpu().numpy(), count.reshape(N*L).cpu().numpy()
            density_pred, count_pred = density_pred.reshape(N*L, 1, H, W).detach().cpu().numpy(), count_pred.reshape(N*L).detach().cpu().numpy()
            n2show = min(args['n2show'], X.shape[0])  # show args['n2show'] images at most
            show_images(img_plt, 'train gt', X[0:n2show], density[0:n2show], count[0:n2show], shape=args['vis_shape'])
            show_images(img_plt, 'train pred', X[0:n2show], density_pred[0:n2show], count_pred[0:n2show], shape=args['vis_shape'])

        if valid_loader is None:
            print()
            continue

        # validation phase
        model.eval()  # set model to evaluation mode (affects batchnorm and dropout, if present)
        loss_hist = []
        density_loss_hist = []
        count_loss_hist = []
        count_err_hist = []
        X, mask, density, count = None, None, None, None
        t0 = time.time()
        for i, (X, mask, density, count, _, seq_len) in enumerate(valid_loader):
            # copy the tensors to GPU (if applicable)
            X, mask, density, count, seq_len = X.to(device), mask.to(device), density.to(device), count.to(device), seq_len.to(device)
            # sort the sequences by descending order of the respective lengths (as expected by the model)
            seqs, seq_len = sort_seqs_by_len([X, mask, density, count], seq_len)
            X, mask, density, count = seqs
            # transpose them so they have shape (seq_len, batch_size, *) (as expected by the model)
            X, mask, density, count = X.transpose(1, 0), mask.transpose(1, 0), density.transpose(1, 0), count.transpose(1, 0)

            # forward pass through the model
            with torch.no_grad():  # no need to compute gradients in validation (faster and uses less memory)
                density_pred, count_pred = model(X, mask=mask, lengths=seq_len)

            # compute the loss
            N = torch.sum(seq_len)
            density_loss = torch.sum((density_pred - density)**2)/(2*N)
            count_loss = torch.sum((count_pred - count)**2)/(2*N)
            loss = density_loss + args['lambda']*count_loss

            # save the loss values
            loss_hist.append(loss.item())
            density_loss_hist.append(density_loss.item())
            count_loss_hist.append(count_loss.item())
            count_err = torch.sum(torch.abs(count_pred - count))/N
            count_err_hist.append(count_err.item())
        t1 = time.time()

        # print the average validation losses
        valid_loss = sum(loss_hist)/len(loss_hist)
        valid_density_loss = sum(density_loss_hist)/len(density_loss_hist)
        valid_count_loss = sum(count_loss_hist)/len(count_loss_hist)
        valid_count_err = sum(count_err_hist)/len(count_err_hist)
        print('Validation statistics:')
        print('global loss: {:.3f} | density loss: {:.3f} | count loss: {:.3f} | count error: {:.3f}'
              .format(valid_loss, valid_density_loss, valid_count_loss, valid_count_err))
        print('time: {:.0f} seconds'.format(t1-t0))
        print()

        if args['use_visdom']:
            # plot the losses
            loss_plt.plot('global loss', 'valid', 'MSE', epoch, valid_loss)
            loss_plt.plot('density loss', 'valid', 'MSE', epoch, valid_density_loss)
            loss_plt.plot('count loss', 'valid', 'MSE', epoch, valid_count_loss)
            loss_plt.plot('count error', 'valid', 'MAE', epoch, valid_count_err)

            # show a few training examples (images + density maps)
            X *= mask  # show the active region only
            X, mask, density, count = X.transpose(1, 0), mask.transpose(1, 0), density.transpose(1, 0), count.transpose(1, 0)
            density_pred, count_pred = density_pred.transpose(1, 0), count_pred.transpose(1, 0)
            N, L, C, H, W = X.shape
            X, density, count = X.reshape(N*L, C, H, W).cpu().numpy(), density.reshape(N*L, 1, H, W).cpu().numpy(), count.reshape(N*L).cpu().numpy()
            density_pred, count_pred = density_pred.reshape(N*L, 1, H, W).cpu().numpy(), count_pred.reshape(N*L).cpu().numpy()
            n2show = min(args['n2show'], X.shape[0])  # show args['n2show'] images at most
            show_images(img_plt, 'valid gt', X[0:n2show], density[0:n2show], count[0:n2show], shape=args['vis_shape'])
            show_images(img_plt, 'valid pred', X[0:n2show], density_pred[0:n2show], count_pred[0:n2show], shape=args['vis_shape'])

    torch.save(model.state_dict(), args['model_path'])


if __name__ == '__main__':
    main()
