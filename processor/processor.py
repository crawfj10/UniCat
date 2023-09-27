import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter, AverageMeterList
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from get_feats import get_features
import copy
from utils.investigate_ranking import RankingHistory
from loss.softmax_loss import CrossEntropyLabelSmooth
import torch.nn.functional as F
import numpy as np
import random

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, num_classes, num_cam):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
        logger = logging.getLogger("transreid.train")
        logger.info('start training')
    else:
        logger = None
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)

    loss_meter = AverageMeterList()
    acc_meter = AverageMeter()
    
    evaluator = R1_mAP_eval(cfg, num_query, max_rank=50, use_cam=cfg.TEST.USE_CAM and (num_cam > 1), feat_norm=cfg.TEST.FEAT_NORM)

    scaler = amp.GradScaler()
    # train
    if cfg.MODEL.ID_HARD_MINING:
        weight = torch.normal(0, 1, size=(num_classes, 768))
    for epoch in range(1, epochs + 1):
        if cfg.MODEL.ID_HARD_MINING:
            train_loader.batch_sampler.sampler.update_weight(weight)
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, pid, _, target_cam, paths) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = pid.to(device)
            target_cam = target_cam.to(device)
            
            loss_lst = []
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, print_ = (n_iter != 0) )
                losses = loss_fn(score, feat, target, target_cam, print_ = (n_iter == 0))
            id_loss, tri_loss, cen_loss, dis_loss = losses
            loss = None
            if id_loss is not None:
                id_loss = sum(id_loss) / len(id_loss)
                loss_lst.append(id_loss.item())
            else:
                loss_lst.append(0)
                id_loss = 0
            if tri_loss is not None:
                tri_loss = sum(tri_loss) / len(tri_loss)
                loss_lst.append(tri_loss.item())
            else:
                tri_loss = 0
                loss_lst.append(0)
            if cen_loss is not None:
                cen_loss = sum(cen_loss) / len(cen_loss)
                loss_lst.append(cen_loss.item())
            else:
                cen_loss = 0
                loss_lst.append(0)
            
            if dis_loss is not None and cfg.MODEL.CLS_TOKENS_LOSS:
                loss_lst.append(dis_loss.item())
            else:
                if dis_loss is not None:
                    loss_lst.append(dis_loss.item())
                else:
                    loss_lst.append(0)
                dis_loss = 0

            if epoch > cfg.MODEL.DIS_DELAY:
                loss = id_loss + tri_loss + cen_loss + dis_loss
            else:
                loss = dis_loss
            
            loss_lst.append(loss.item())
            scaler.scale(loss).backward()

            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            try:
                if isinstance(score, list):
                    if isinstance(score[0], tuple):
                        acc = (score[0][0].max(1)[1] == score[0][1]).float().mean()
                    else:
                        acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()
            except:
                acc = 0

            loss_meter.update(loss_lst)
            acc_meter.update(acc, 1)


            torch.cuda.synchronize()
            if logger and (n_iter + 1) % log_period == 0:
                loss_avg = loss_meter.avg
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ID_Loss: {:.3f}, TRIPLE_Loss: {:.3f}, CENTROID_loss {:.3f}, DISSIMILAR Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_avg[-1], loss_avg[0], loss_avg[1], loss_avg[2], loss_avg[3],  acc_meter.avg, scheduler._get_lr(epoch)[0]))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN or not logger:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), cfg.CKPT_DIR + '_{}.pth'.format(epoch))
            else:
                torch.save(model.state_dict(), cfg.CKPT_DIR + '_{}.pth'.format(epoch))
        
        if epoch % eval_period == 0 and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
            model.eval()
            for n_iter, (img, pid, camid, camid_tensor,  _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    camid_tensor = camid_tensor.to(device)
                    _, feat = model(img, cam_label=camid_tensor)
                    evaluator.update((feat, pid, camid))
            cmc, mAP = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
    return None, None

def do_inference(cfg, model, val_loader, num_query, num_cam, num_mode, eval_mode='multi', verbose=True):
    device = "cuda"
    assert eval_mode in ('multi', 'single')
    if eval_mode == 'multi':
        evaluator = R1_mAP_eval(cfg, num_query, num_mode=num_mode, max_rank=50, use_cam=cfg.TEST.USE_CAM and (num_cam > 1), feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
        evaluator.reset()
    else:
        assert not cfg.TEST.MEAN_FEAT 
        assert not cfg.MODEL.USE_FUSION  # model may have been trained with fusion but to obtain single-modal metrics turn this flag off
        evaluator = []
        for _ in range(num_mode):
            evaluator.append(R1_mAP_eval(cfg, num_query, num_mode=1, max_rank=50, use_cam=cfg.TEST.USE_CAM and (num_cam > 1), feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING))
            evaluator[-1].reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    
    if cfg.MODEL.NAME == 'transformer':
        embed_size = cfg.MODEL.CLS_TOKEN_NUM * cfg.MODEL.EMBED_DIM
    else:
        embed_size = cfg.MODEL.EMBED_DIM

    model.eval()
    print('Running all val')
    for n_iter, (img, pid, camid, camid_tensor, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camid_tensor = camid_tensor.to(device)
            _, feat = model(img, cam_label=camid_tensor) # feat has size (B, num_mode*nbr_cls*H)
            if eval_mode == 'multi':
                evaluator.update((feat, pid, camid))
            else:
                for m in range(num_mode):
                    evaluator[m].update((feat[:, m*embed_size:(m+1)*embed_size], pid, camid))
    torch.cuda.empty_cache()
    
    if eval_mode == 'multi':
        cmc, mAP = evaluator.compute()
        if verbose:
            print('mAP:', mAP, 'Rank-1:', cmc[0], 'Rank-5:', cmc[4], 'Rank-10:', cmc[9])
        return mAP, cmc

    # doing single-modality (intra-modal) inference 
    cmc_dict = {}
    map_dict = {}
    for m in range(num_mode):
        cmc, mAP = evaluator[m].compute()
        if verbose:
            print('Mode', str(m), ':')
            print('mAP:', mAP, 'Rank-1:', cmc[0], 'Rank-5:', cmc[4], 'Rank-10:', cmc[9])
        cmc_dict[m] = cmc
        map_dict[m] = mAP
    return map_dict, cmc_dict



def check_kl(cfg, model, val_loader, num_query, num_cam):

    device = "cuda"
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    mode_feats = defaultdict(list)
    model.eval() 
    print('Running all val')
    for n_iter, (img, pid, camid, camid_tensor, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camid_tensor = camid_tensor.to(device)
            score, _ = model(img, cam_label=camid_tensor)
            for m in range(len(score)):
                mode_feats[m].append(score[m])
        if n_iter == 5: break
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    for m in mode_feats:
        mode_feats[m] = torch.cat(mode_feats[m], dim=0)
        mode_feats[m] = F.log_softmax(mode_feats[m], dim=1)
    print(mode_feats[0].size(), mode_feats[1].size())
    print('doing kl')
    print(kl_loss(mode_feats[0], mode_feats[1]))
    print(kl_loss(mode_feats[1], mode_feats[0]))
    raise Exception


def check_ortho(cfg, model, val_loader, num_query, num_cam):
    
    assert not cfg.MODEL.USE_FUSION
    assert cfg.MODEL.CLS_TOKEN_NUM == 1
    assert not cfg.TEST.MEAN_FEAT
    
    device = "cuda"
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    
    mode_feats = defaultdict(list)
    model.eval()
    print('Running all val')
    for n_iter, (img, pid, camid, camid_tensor, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camid_tensor = camid_tensor.to(device)
            _, feat = model(img, cam_label=camid_tensor)  # (B, num_mode*H)
            for m in range(2):
                mode_feats[m].append(feat[:,m*768:(m+1)*768])
    
    num_trial = 100
    assert len(mode_feats) == 2

    for m in mode_feats:
        mode_feats[m] = torch.cat(mode_feats[m], dim=0)
        print(mode_feats[m].size())

    dist = []
    for _ in range(num_trial):
        x1_ind = random.randint(0, mode_feats[0].size(0)-1)
        x2_ind = random.randint(0, mode_feats[0].size(0)-1)
        dist.append(F.cosine_similarity(mode_feats[0][x1_ind], mode_feats[1][x2_ind], dim=0).item())
    print(np.mean(dist), np.std(dist))
    
    dist = []
    for _ in range(num_trial):
        x1_ind = random.randint(0, mode_feats[0].size(0)-1)
        dist.append(F.cosine_similarity(mode_feats[0][x1_ind], mode_feats[1][x1_ind], dim=0).item())
    print(np.mean(dist), np.std(dist))

    

def do_ranking_history(cfg, model, val_loader, num_query):
    raise NotImplementedError # TODO
    device = 'cuda'
    save_path = '/home/ubuntu/jenni/ranking_history/prai/'
    evaluator = RankingHistory(num_query, save_path, max_rank=10, use_cam=cfg.TEST.USE_CAM, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    print('Running all val')
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    evaluator.compute()





