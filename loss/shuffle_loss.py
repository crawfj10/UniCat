import torch
import numpy as np


def shuffle_loss(feat, feat_shuffle, normalize=True):

    if normalize:
        feat = torch.nn.functional.normalize(feat, dim=1)
        feat_shuffle = torch.nn.functional.normalize(feat_shuffle, dim=1)

    dist_mat = torch.pow(feat, 2).sum(dim=1) + \
               torch.pow(feat_shuffle, 2).sum(dim=1) - \
               2*(feat*feat_shuffle).sum(dim=1)
    #print(dist_mat)
    return torch.mean(dist_mat)



if __name__ == '__main__':
    
    x = torch.tensor([[1, 1], [0, 1]])
    y = torch.tensor([[0, 1], [0, 1]])
    print(shuffle_loss(x.float(), y.float()))
