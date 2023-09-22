import torch
from torch import nn
from collections import defaultdict

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def get_dist_an(dist_mat, labels, centroid_labels, hard_mining=True):

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == labels.size(0)
    assert dist_mat.size(1) == centroid_labels.size(0)

    a, c = dist_mat.size(0), dist_mat.size(1)

    # shape [a, c]
    is_neg = centroid_labels.expand(a, -1).ne(labels.expand(c, -1).t())
    
    if hard_mining:
        dist_an, _ = torch.min(dist_mat[is_neg].contiguous().view(a, -1), 1, keepdim=True)
    else:
        raise NotImplementedError
        # is broken
        random_sort = torch.stack([torch.randperm(c) for _ in range(a)], dim=0)
        dist_an = dist_mat[is_neg].contiguous().view(a, -1)
        dist_an = torch.gather(dist_an, 1, random_sort)[:, 0]
    # shape [a]
    dist_an = dist_an.squeeze(1)

    return dist_an


def get_full_centroids(feats, labels, label_to_ind):

    centroids = []    
    label_set = list(label_to_ind)
    for label in label_set:
        centroid = torch.mean(feats[label_to_ind[label]], dim=0)
        centroids.append(centroid)
    return torch.stack(centroids, dim=0), labels[[label_to_ind[l][0] for l in label_set]]

def get_filtered_centroids(feats, labels, label_to_ind):
    filtered_centroids = []
    for i, label in enumerate(labels):
        other_ind = list(set(label_to_ind[label.item()]) - {i})
        filtered_centroid = feats[other_ind]
        filtered_centroids.append(torch.mean(filtered_centroid, dim=0))
    return torch.stack(filtered_centroids, dim=0)


class CentroidTripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0, hard_mining=True):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        self.hard_mining = hard_mining

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        
        label_to_ind = defaultdict(list)
        for i, label in enumerate(labels):
            label_to_ind[label.item()].append(i)

        full_centroids, full_centroid_labels = get_full_centroids(global_feat, labels, label_to_ind)
        dm_full = euclidean_dist(global_feat, full_centroids)
        dist_an = get_dist_an(dm_full, labels, full_centroid_labels, hard_mining=self.hard_mining)
        assert dist_an.size(0) == labels.size(0)

        filtered_centroids = get_filtered_centroids(global_feat, labels, label_to_ind)
        assert filtered_centroids.size(0) == labels.size(0)
        dist_ap = euclidean_dist(global_feat, filtered_centroids)
        dist_ap = torch.diag(dist_ap)
        assert dist_ap.size(0) == labels.size(0)
        
        #print(dist_ap.size(), dist_an.size())
        #raise Exception
        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


