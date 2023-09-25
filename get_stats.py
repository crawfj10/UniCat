import numpy as np
import re
import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from collections import defaultdict

def do_model_stats(cfg, model, data_loader, ckpt_paths, num_query, camera_num, num_mode, eval_mode='multi'):
    r1_history = defaultdict(list)
    map_history = defaultdict(list)

    for path_ in ckpt_paths:
        print('Loading ckpt')
        model.load_param(path_)

        map_, cmc = do_inference(cfg,
                    model,
                     val_loader,
                     num_query, camera_num, num_mode, eval_mode=eval_mode, verbose=False)
        if isinstance(map_, dict):
            for m in map_:
                map_history[m].append(map_[m])
                r1_history[m].append(cmc[m][0])
        else:
            map_history['all'].append(map_)
            r1_history['all'].append(cmc[0])


    print('mAP history:')
    print(map_history, '\n')
    print('Rank-1 history:')
    print(r1_history, '\n')

    print('Mean mAP:')
    for m in map_history:
        print('Mode', m, ':')
        print('mAP:', np.mean(map_history[m]), np.std(map_history[m]))
        print('Rank-1:', np.mean(r1_history[m]), np.std(r1_history[m]))


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

    train_loader, train_loader_normal, val_loader, num_query, num_mode, num_classes, num_camera = make_dataloader(cfg)
    model = make_model(cfg, num_mode, num_class=num_classes, camera_num=num_camera)

    #root_path = '/home/ubuntu/jenni/ckpts/mm/rgbnt201/hpo/'
    root_path = '/home/ubuntu/jenni/ckpts/mm/rgbnt100/hpo/'
    bs = 128
    bs = 256
    #lr = '.008'
    lr = '.016'
    #lr = '.032'
    #file_ = 'vit_rnt_fusion_av_s256x128_bs_' + str(bs) + '_lr_' + lr + '_120.pth'
    file_ = 'vit_rnt_fusion_av_s128x256_bs_' + str(bs) + '_lr_' + lr + '_120.pth'
    seeds = [1235, 1236, 1237, 1238]

    paths = [root_path + 'seed_' + str(seed) + '/' + file_ for seed in seeds]

    do_model_stats(cfg, model, val_loader, paths, num_query, num_camera, num_mode, eval_mode='multi')
