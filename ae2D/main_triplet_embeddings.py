import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd.variable import Variable

from models.triplet_net import get_model
from utils.triplet_loader import get_loader


def train_split(model, train_data_loader, test_data_loader, exp_dir):
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value]}]
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)  #todo: try margin=0
    optimizer = optim.Adam(params, lr=args.lr)

    writer = SummaryWriter(os.path.join(exp_dir, "log"))

    test_accuracy = None
    for epoch in range(1, args.epochs + 1):
        loss, test_accuracy = test(test_data_loader, model, criterion)
        writer.add_scalar('Loss/test', loss.item(), epoch*len(train_data_loader))
        writer.add_scalar('Accuracy/test', test_accuracy.item(), epoch*len(train_data_loader))
        loss, accuracy = train(train_data_loader, model, criterion, optimizer, epoch)
        writer.add_scalar('Loss/train', loss.item(), epoch*len(train_data_loader))
        writer.add_scalar('Accuracy/train', accuracy.item(), epoch*len(train_data_loader))
        if epoch % args.ckp_freq == 0:
            file_name = os.path.join(exp_dir, "checkpoint_" + str(epoch) + ".pth")
            torch.save({"epoch": epoch + 1, 'state_dict': model.state_dict()}, file_name)
    return test_accuracy.item()


def main():
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)
    cudnn.benchmark = True

    model = get_model(args, device)

    test_accuracies = {}
    for split in range(args.splits):
        triplet_train_path = os.path.join(args.splits_dir, f"train_triplets_{split}.txt")
        triplet_test_path = os.path.join(args.splits_dir, f"test_triplets_{split}.txt")
        train_data_loader, test_data_loader = get_loader(args, triplet_train_path, triplet_test_path)
        exp_dir = os.path.join(args.result_dir, args.exp_name, f"{split}")
        os.makedirs(exp_dir, exist_ok=True)
        accuracy = train_split(model, train_data_loader, test_data_loader, exp_dir)
        test_accuracies[split] = accuracy
    overall_result = np.mean(list(test_accuracies.values()))
    print(f"overall_result: {overall_result}")
    exp_fout = os.path.join(args.result_dir, args.exp_name, "overall.txt")
    with open(exp_fout, "w") as fout:
        fout.write(f"{overall_result}")


def load_model(args, model):
    # Load weights if provided
    if args.ckp:
        if os.path.isfile(args.ckp):
            print("=> Loading checkpoint '{}'".format(args.ckp))
            checkpoint = torch.load(args.ckp)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded checkpoint '{}'".format(args.ckp))
        else:
            print("=> No checkpoint found at '{}'".format(args.ckp))


def train(data, model, criterion, optimizer, epoch):
    print(f"******** Training epoch {epoch} ********")
    total_loss = 0
    model.train()
    for batch_idx, triplet in enumerate(data):
        anchor, pos, neg = triplet
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        E1, E2, E3 = model(anchor, pos, neg)
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)  # -1 means dist_E1_E3 should be bigger than dist_E1_E2
        target = Variable(target.to(device))
        loss = criterion(dist_E1_E2, dist_E1_E3, target)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_accuracy = 0
    model.eval()
    for batch_idx, triplet in enumerate(data):
        anchor, pos, neg = triplet
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        E1, E2, E3 = model(anchor, pos, neg)
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        prediction = dist_E1_E2 > dist_E1_E3
        total_accuracy += torch.sum(prediction).float()

    print('Train Loss: {:.4f}'.format(total_loss / len(data)))
    print('Train Accuracy: {:.4f}'.format(total_accuracy / len(data.dataset)))
    print("****************")
    return total_loss, total_accuracy / len(data.dataset)


def test(data, model, criterion):
    print("******** Testing ********")
    with torch.no_grad():
        model.eval()
        total_accuracy = 0
        total_loss = 0
        for batch_idx, triplet in enumerate(data):
            anchor, pos, neg = triplet
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            E1, E2, E3 = model(anchor, pos, neg)
            dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
            dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

            target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
            target = target.to(device)
            target = Variable(target)

            loss = criterion(dist_E1_E2, dist_E1_E3, target)
            total_loss += loss

            prediction = dist_E1_E2 > dist_E1_E3
            total_accuracy += torch.sum(prediction).float()

        print('Test Loss: {}'.format(total_loss / len(data)))
        print('Test Accuracy: {}'.format(total_accuracy / len(data.dataset)))
    print("****************")
    return total_loss, total_accuracy / len(data.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_dir', type=str, )
    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--encodings_path', type=str, )
    parser.add_argument('--result_dir', default='data', type=str,
                        help='Directory to store results')
    parser.add_argument('--exp_name', default='exp0', type=str,
                        help='name of experiment')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None,
                        help="List of GPU Devices to train on")
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--ckp_freq', type=int, default=10, metavar='N',
                        help='Checkpoint Frequency (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--margin', type=float, default=1.0, metavar='M',
                        help='margin for triplet loss (default: 1.0)')
    parser.add_argument('--ckp', default=None, type=str,
                        help='path to load checkpoint')
    parser.add_argument('--num_input', default=65, type=int,)
    parser.add_argument('--num_output', default=32, type=int,)

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.cuda:
        device = 'cuda'
        if args.gpu_devices is None:
            args.gpu_devices = [0]
    else:
        device = 'cpu'
    main()
