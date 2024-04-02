import torch
import scipy.sparse as sp
import dgl
from greatx.attack.untargeted import Metattack,RandomAttack
from greatx.attack.targeted import Nettack
from greatx.nn.models import GCN

from tqdm import tqdm
import numpy as np

from greatx.training import Trainer
from  greatx.datasets import GraphDataset

from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes


def load_graph_atta_greatxVV(args):
    dataPathGreatX = './datasets/GreatX/'
    path1 = './datasets/NoiseGraph_GreatXgene/'
    name = path1 + args.attack_method + '_' + args.dataset + "_" + str(args.ptb_rate) + '.pt'
    data_name = args.dataset.lower()

    if args.loadData:
        if args.ptb_rate!=0:
            data, splits = torch.load(name)
        else:
            data = GraphDataset(root=dataPathGreatX, name=data_name)[0]
            splits = split_nodes(data.y, random_state=15)
    else:
        data = GraphDataset(root=dataPathGreatX, name=data_name)[0]
        splits = split_nodes(data.y, random_state=15)
        if args.ptb_rate!=0:
            trainer_before = Trainer(GCN(data.x.size(-1), int(data.y.max().item() + 1), bias=False, acts=None),device=args.device)
            ckp = ModelCheckpoint('model_before.pth', monitor='val_acc')
            trainer_before.fit(data, mask=(splits.train_nodes, splits.val_nodes), callbacks=[ckp])
            # logs = trainer_before.evaluate(data, splits.test_nodes)
            # print(f"Before attack\n {logs}")

            if args.attack_method.lower() == 'metattack':
                attacker = Metattack(data, device=args.device)
                attacker.setup_surrogate(trainer_before.model,labeled_nodes=splits.train_nodes,unlabeled_nodes=splits.test_nodes, lambda_=0.)
                attacker.reset()
                attacker.attack(args.ptb_rate)
                data = attacker.data().cpu()

            elif args.attack_method.lower() == 'nettack':
                indices = torch.randperm(splits.test_nodes.size(0))[:300]
                node_list = splits.test_nodes[indices]
                for target_node in tqdm(node_list):
                    attacker = Nettack(data, device=args.device)
                    attacker.set_max_perturbations(6)
                    attacker.setup_surrogate(trainer_before.model)
                    attacker.reset()
                    attacker.attack(target_node, num_budgets = args.ptb_rate)
                    data = attacker.data()
                splits.test_nodes = node_list
                data = attacker.data().cpu()

            elif args.attack_method.lower() == 'randomattack':
                attacker = RandomAttack(data)
                attacker.reset()
                attacker.attack(args.ptb_rate)
                data = attacker.data().cpu()

            torch.save([data, splits], name)

        else:
            pass

    features =  torch.FloatTensor(data.x.to_dense().numpy())

    adjacency_matrix_indx = data.edge_index.to_dense().numpy()
    adjacency_matrix = np.zeros((data.x.shape[0], data.x.shape[0]), dtype=int)
    adjacency_matrix[adjacency_matrix_indx[0], adjacency_matrix_indx[1]] = 1
    A =  torch.FloatTensor(adjacency_matrix)
    A_nomal = row_normalize(A)
    I = torch.eye(A.shape[1]).to(A.device)
    A_I = A + I
    A_I_nomal = row_normalize(A_I)

    idx_train = torch.LongTensor(splits.train_nodes)
    idx_val = torch.LongTensor( splits.val_nodes)
    idx_test = torch.LongTensor(splits.test_nodes)
    labels =  data.y
    return [A_I_nomal, A_nomal, A], features, labels, idx_train, idx_val, idx_test



def row_normalize(A):
    """Row-normalize dense matrix"""
    eps = 2.2204e-16
    rowsum = A.sum(dim=-1).clamp(min=0.) + eps
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A


def torch2dgl(graph):
    N = graph.shape[0]
    if graph.is_sparse:
        graph_sp = graph.coalesce()
    else:
        graph_sp = graph.to_sparse()
    edges_src = graph_sp.indices()[0]
    edges_dst = graph_sp.indices()[1]
    edges_features = graph_sp.values()
    graph_dgl = dgl.graph((edges_src, edges_dst), num_nodes=N)
    # graph_dgl.edate['w'] = edges_features
    return graph_dgl

def preprocess_features(features,eps =1e-6):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = rowsum + eps
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features