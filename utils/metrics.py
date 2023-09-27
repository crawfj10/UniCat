import torch
import numpy as np
import os
from utils.reranking import re_ranking
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from sklearn.decomposition import PCA
import umap
from scipy.spatial import Voronoi, voronoi_plot_2d


def euclidean_distance(qf, gf, average_gal=False, average_mode=0, g_pids=None, feat_norm=True):
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, use_cam=True,  max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    print(num_q, num_g)
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    full_filter = True

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    mode_ranks = defaultdict(list)
    mode_APs = defaultdict(list)
    mode_r1s = defaultdict(list)
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

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = (tmp_cmc / y) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
                 
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    assert num_valid_q == all_cmc.shape[0]
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, cfg, num_query, num_mode=1, max_rank=50, use_cam=True, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.cfg = cfg
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.use_cam = use_cam
        self.num_mode = num_mode
        if not self.use_cam:
            print('ATTENTION: Not filtering by camera')

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.viewids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
    
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
            
        # query
        qf = feats[:self.num_query]
        # gallery
        gf = feats[self.num_query:]

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP  = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, use_cam=self.use_cam)

        if self.cfg.TEST.VISUALIZE: 
            self.visualize_embeddings()
        
        return cmc, mAP

    # def visualize_embeddings(self):
    #     """Visualize the embeddings using t-SNE, UMAP, and PCA with gradient legend."""
    #     feats = torch.cat(self.feats, dim=0).numpy()
        
    #     # Define dimensionality reduction methods
    #     reducers = {
    #         "t-SNE": TSNE(n_components=2, random_state=0),
    #         "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation'),
    #         "PCA": PCA(n_components=2)
    #     }
        
    #     for name, reducer in reducers.items():
    #         print(f"Applying {name} dimensionality reduction")

    #         embeddings = reducer.fit_transform(feats)

    #         plt.figure(figsize=(10, 8))

    #         # Using different markers for each camid for better visualization
    #         markers = ['o', 's', 'D', '^', 'v', '*', '+', 'x', '|', '_']

    #         unique_pids = np.unique(self.pids)
    #         colors = plt.cm.jet(np.linspace(0, 1, len(unique_pids)))

    #         for idx, pid in enumerate(unique_pids):
    #             indices = np.where(np.array(self.pids) == pid)
    #             unique_camids_for_pid = np.unique(np.array(self.camids)[indices])
    #             for j, camid in enumerate(unique_camids_for_pid):
    #                 cam_indices = np.where(np.array(self.camids)[indices] == camid)
    #                 plt.scatter(embeddings[indices][cam_indices, 0], embeddings[indices][cam_indices, 1], 
    #                             color=colors[idx], marker=markers[j%len(markers)])

    #         # Add gradient color bar as legend
    #         norm = Normalize(vmin=min(unique_pids), vmax=max(unique_pids))
    #         cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), orientation='vertical')
    #         cbar.set_label('PID')

    #         plt.title(f"{name} visualization")
    #         plt.savefig(f"{name.lower()}_visualization.png", bbox_inches='tight', dpi=300)




    def visualize_embeddings(self):
        """Visualize the embeddings using t-SNE, UMAP, PCA, and t-SNE with Voronoi diagram."""
        feats = torch.cat(self.feats, dim=0).numpy()

        # Define dimensionality reduction methods
        reducers = {
            "t-SNE": TSNE(n_components=2, random_state=0),
            "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation'),
            "PCA": PCA(n_components=2)
        }

        for name, reducer in reducers.items():
            print(f"Applying {name} dimensionality reduction")

            embeddings = reducer.fit_transform(feats)

            plt.figure(figsize=(10, 8))

            # Define colors for unique PIDs
            unique_pids = np.unique(self.pids)
            colors = plt.cm.jet(np.linspace(0, 1, len(unique_pids)))

            # Scatter plot all the embeddings
            for idx, pid in enumerate(unique_pids):
                indices = np.where(np.array(self.pids) == pid)
                plt.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors[idx], s=10 if name == "t-SNE" else 50)  # smaller points for t-SNE
    
            # Add gradient color bar as legend
            norm = Normalize(vmin=min(unique_pids), vmax=max(unique_pids))
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), orientation='vertical')
            cbar.set_label('PID')

            plt.title(f"{name} visualization")
            plt.savefig(f"viz_output/{name.lower()}_visualization.png", bbox_inches='tight', dpi=300)

            if name == "t-SNE":
                # Compute the centroids of embeddings for each PID
                centroids = []
                for pid in unique_pids:
                    indices = np.where(np.array(self.pids) == pid)
                    centroid = embeddings[indices].mean(axis=0)
                    centroids.append(centroid)
                centroids = np.array(centroids)

                # Plot Voronoi diagram over t-SNE
                plt.figure(figsize=(10, 8))
                vor = Voronoi(centroids)
                fig, ax = plt.subplots()
                voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='black', line_alpha=1, point_size=2)

                # Scatter plot all the embeddings
                for idx, pid in enumerate(unique_pids):
                    indices = np.where(np.array(self.pids) == pid)
                    plt.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors[idx], s=10)

                # Add gradient color bar as legend
                norm = Normalize(vmin=min(unique_pids), vmax=max(unique_pids))
                cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), orientation='vertical')
                cbar.set_label('PID')

                plt.title("t-SNE visualization with Voronoi diagram")
                plt.savefig("viz_output/tsne_voronoi_visualization.png", bbox_inches='tight', dpi=300)
