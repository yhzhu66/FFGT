import numpy as np
import random as random
import torch
import copy
import argparse
from models.Semi_FFGT import FFGT

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='amazon_photo', help='dataset used:choice: Cora, CiteSeer, acm, amazon_photo,.')
parser.add_argument('--lr', type=float, default=0.005, help='learning ratio.')
parser.add_argument('--wd', type=float, default=5e-5, help='weight delay.')
parser.add_argument('--order', type=int, default=2, help='number of multi-hop graph.')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout in hidden layers.')
parser.add_argument('--dropout_att', type=float, default=0.6, help='dropout in self-attention layers.')
parser.add_argument('--Trans_layer_num', type=int, default=2, help='the number of layers.')
parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension.')
parser.add_argument('--tau', type=float, default=0.1, help='hyparameter in Graph contrastive loss.')
parser.add_argument('--alpha', type=float, default=100, help='trade-off hybrid representation in loss function.')
parser.add_argument('--beta', type=float, default=0.1, help='trade-off high-frequency representation in loss function.')
parser.add_argument('--loadData', type=bool, default=True, help='True indicates loading existing perturbed data')
parser.add_argument('--attack_method', type=str, default='randomattack', help='From [metattack, nettack randomattack]')
parser.add_argument('--ptb_rate', type=float, default=0.3, help='perturbed ratio to graph data, selecting from metattack[0,0.05, 0.1, 0.15, 0.20, 0.25], nettack [0, 1,2,3,4,5] ,RandomAttack.[0,0.1.0.2.0.3.0.4.0.5].')
parser.add_argument('--nb_epochs', type=int, default=5000, help='maximal epochs.')
parser.add_argument('--patience', type=int, default=60, help='early stop.')


args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    if torch.cuda.is_available():
        args.device = torch.device('cuda:2')
    else:
        args.device = torch.device('cpu')

    ACC_seed = []
    Time_seed = []
    for seed in range(2020, 2024):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        embedder = FFGT(copy.deepcopy(args))
        test_acc, training_time, stop_epoch = embedder.training()
        ACC_seed.append(test_acc)
        Time_seed.append(training_time)
        torch.cuda.empty_cache()
    ACC_seed = np.array(ACC_seed)*100

    print("-->ACC %.4f  -->STD is: %.4f" %(np.mean(ACC_seed), np.std(ACC_seed)))
