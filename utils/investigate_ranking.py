import torch 
import shutil
import random
import os
import torch.nn as nn
from .metrics import euclidean_distance
from .reranking import re_ranking
import numpy as np
import re

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, use_cam=True, max_rank=10):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    print(num_q, num_g)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_matches = []
    all_order = []
    all_qidx = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        if use_cam:
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[q_idx][keep]
            order = order[keep]
        else:
            orig_cmc = matches[q_idx]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        num_valid_q += 1
        all_matches.append(orig_cmc[:max_rank])
        all_order.append(order[:max_rank])
        all_qidx.append(q_idx)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    return all_matches, all_order, all_qidx


class RankingHistory:

    def __init__(self, num_query, save_path, max_rank=10, feat_norm=True, reranking=False, use_cam=True):
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.use_cam = use_cam
        self.save_path = save_path
    
    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, img = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(img)

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_imgs = self.img_paths[:self.num_query]

        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_imgs = self.img_paths[self.num_query:]
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        all_matches, all_order, all_qidx  = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, use_cam=self.use_cam)


        q_inds = list(range(len(all_qidx)))
        random.shuffle(q_inds)
        
        n = 0
        for q_ind in q_inds:
            matches = all_matches[q_ind]
            order = all_order[q_ind]
            q_ind = all_qidx[q_ind]
            #print(q_ind, q_pids[q_ind], q_imgs[q_ind])
            #print([g_imgs[g] for g in all_order[q_ind]])
            #print([g_pids[g] for g in all_order[q_ind]])

            print(matches)
            #raise Exception
            txt = None
            while txt not in ('y', 'n'):
                txt = input('Save? y/n')
            if txt == 'y':
                save_fld = self.save_path + str(n) + '/'
                os.mkdir(save_fld)
                img_file = re.search('[^/]+$', q_imgs[q_ind]).group(0)
                shutil.copyfile(q_imgs[q_ind], save_fld + 'query_' + img_file)
                
                for i, g_ind in enumerate(order):
                    img_file = re.search('[^/]+$', g_imgs[g_ind]).group(0)
                    shutil.copyfile(g_imgs[g_ind], save_fld + 'r' + str(i) + '_' + img_file)
                n += 1
            if n == 10:
                break

