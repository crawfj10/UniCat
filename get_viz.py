import os
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from sklearn.decomposition import PCA
import umap
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.manifold import TSNE

import torch
import torch.nn as nn

from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger

def get_features(model, data_loader, normalize=True, device='cuda', max_pids=10):
    assert max_pids <= 50, "max_pids must be less than 50 for RGBNT100"

    model.eval()
    img_paths = []
    pids = []
    camids = []
    feats = []
    
    print('Getting feats')
    for n_iter, (img, pid, camid_list, camid_tensor, img_path) in enumerate(data_loader):
        with torch.no_grad():
            img = img.to(device)
            camid_tensor = camid_tensor.to(device)
            _, feat = model(img, cam_label=camid_tensor) # (size B, num_mode*nbr_cls*h)
            img_paths.extend(img_path)
            feats.append(feat)
            camids.extend(camid_list)
            pids.extend(pid)
    print('Done')
    
    feats = torch.cat(feats, dim=0)
    if normalize:
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)

    # Select all camids related to the first 'max_pids' unique PIDs
    unique_pids = np.unique(pids)
    selected_pids = unique_pids[:max_pids]
    
    selected_indices = [idx for idx, pid in enumerate(pids) if pid in selected_pids]
    
    feats = feats[selected_indices]
    pids = torch.tensor([pids[idx] for idx in selected_indices], device=device)  # Convert to tensor
    camids = torch.tensor([camids[idx] for idx in selected_indices], device=device)  # Convert to tensor
    img_paths = [img_paths[idx] for idx in selected_indices]  # Keep as list
    
    return feats, pids, camids, img_paths



def visualize_embeddings(feats, pids, num_pid):
    '''
    Params: 
    - feats: tensor with shape (B, num_mode, h)
    - pids: list of PIDs corresponding to each embedding in feats
    - num_pid: number of unique PIDs to visualize
    '''
    B, num_mode, h = feats.shape
    feats = feats.view(B * num_mode, h).cpu().numpy()
    modality_shapes = ['o', 's', '^']  # circle, square, triangle for each modality

    # Update the pids list to repeat for each modality
    pids = np.tile(pids.cpu().numpy(), num_mode)

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

        unique_pids = np.unique(pids)
        
        # Create distinct colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pids)))
        
        legend_elements = []

        for idx, pid in enumerate(unique_pids):
            for modality in range(num_mode):
                indices = np.where((np.array(pids) == pid) & (np.arange(len(pids)) % num_mode == modality))
                plt.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors[idx], s=50, marker=modality_shapes[modality], label=f'PID {idx}')
            # For legend: Use only once for each PID
            legend_elements.append(plt.Line2D([0], [0], color=colors[idx], marker='o', label=f'PID {idx}', markersize=10))

        plt.legend(handles=legend_elements, loc='upper right')
        plt.title(f"{name} visualization")
        plt.savefig(f"viz_output/rgbn300_{name.lower()}_visualization_perpid_fusionav_pid-{num_pid}.png", bbox_inches='tight', dpi=300)

        if name == "t-SNE":
            centroids = []
            for pid in unique_pids:
                indices = np.where(np.array(pids) == pid)
                centroid = embeddings[indices].mean(axis=0)
                centroids.append(centroid)
            centroids = np.array(centroids)

            plt.figure(figsize=(10, 8))
            vor = Voronoi(centroids)
            fig, ax = plt.subplots()
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='black', line_alpha=1, point_size=2)

            for idx, pid in enumerate(unique_pids):
                for modality in range(num_mode):
                    indices = np.where((np.array(pids) == pid) & (np.arange(len(pids)) % num_mode == modality))
                    plt.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors[idx], s=10, marker=modality_shapes[modality])

            plt.legend(handles=legend_elements, loc='upper right')
            plt.title("t-SNE visualization with Voronoi diagram")
            plt.savefig(f"viz_output/rgbn300_tsne_voronoi_visualization_perpid_fusionav_pid-{num_pid}.png", bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_mode, num_classes, camera_num = make_dataloader(cfg)
    model = make_model(cfg, num_mode, num_class=num_classes, camera_num=camera_num)
    model.load_param(cfg.TEST.WEIGHT)
    device = 'cuda'
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    if cfg.MODEL.NAME == 'transformer':
        embed_size = cfg.MODEL.CLS_TOKEN_NUM * cfg.MODEL.EMBED_DIM
    else:
        embed_size = cfg.MODEL.EMBED_DIM 

    max_pid_to_viz = 5
    
    feats, pid, _, _ = get_features(model, val_loader, max_pids=max_pid_to_viz)

    # for viewing per pid, not modality specific
    feats = feats.view(-1, 1, num_mode*embed_size)  # size (B, num_mode, h), where B = nbr_val images, h = feature size

    # # for viewing per pid modality 
    # feats = feats.view(-1, num_mode, embed_size)  # size (B, num_mode, h), where B = nbr_val images, h = feature size

    print("Running visualization script")
    visualize_embeddings(feats, pid, max_pid_to_viz)
    print(feats.size())




    
# # test visualize embeddings function with dummy data
# feats = torch.randn(50, 1, 768).to('cuda')
# pids = torch.tensor([0, 1, 2, 3, 4] * 10).to('cuda')
# num_pid = 10
# visualize_embeddings(feats, pids, num_pid)




