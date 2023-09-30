import torch
import torch.nn as nn
import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger



def get_features(model, data_loader, normalize=True, device='cuda'):
    model.eval()
    img_paths = []
    pids = []
    camids = []
    feats = []
    #print(len(data_loader))
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
        #print(n_iter)
    print('Done')
    feats = torch.cat(feats, dim=0)
    if normalize:
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats, pids, camids, img_paths


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
    feats, _, _, _ = get_features(model, val_loader)
    feats = feats.view(-1, num_mode, embed_size)  # size (B, num_mode, h), where B = nbr_val images, h = feature size
    print(feats.size())    

    



