import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter, AverageMeterList
from utils.metrics import R1_mAP_eval
from utils.metrics_new import R1_mAP_eval_new
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
import re

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
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, use_cam=cfg.TEST.USE_CAM and (num_cam > 1), feat_norm=cfg.TEST.FEAT_NORM)

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
            #print(img[0,0,:,0,0])
            #print(img[0,1,:,0,0])
            #print(img[0,2,:,0,0])
            #print(img.size())
            #raise Exception
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = pid.to(device)
            target_cam = target_cam.to(device)
            
            loss_lst = []
            with amp.autocast(enabled=True):
                score, feat, feat_bn = model(img, target, cam_label=target_cam, print_ = (n_iter != 0) )
                losses = loss_fn(score, feat, target, target_cam, print_ = (n_iter == 0))
            id_loss, tri_loss, cen_loss, dis_loss, cc_loss = losses
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
            if cc_loss is not None:
                cc_loss = sum(cc_loss) / len(cc_loss)
                loss_lst.append(cc_loss.item())
            else:
                cc_loss = 0
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
                loss = cc_loss
                #loss = id_loss + tri_loss + cen_loss + dis_loss
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
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ID_Loss: {:.3f}, TRIPLE_Loss: {:.3f}, CENTROID_loss {:.3f}, CC loss {:.3f},  DISSIMILAR Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_avg[-1], loss_avg[0], loss_avg[1], loss_avg[2], loss_avg[3], loss_avg[4],  acc_meter.avg, scheduler._get_lr(epoch)[0]))
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

def do_inference(cfg, model, val_loader, num_query, num_cam, num_mode, eval_mode='multi', verbose=True):
    device = "cuda"
    assert eval_mode in ('multi', 'single', 'cross')
    if eval_mode == 'multi':
        evaluator = R1_mAP_eval(num_query, num_mode=num_mode, max_rank=50, use_cam=cfg.TEST.USE_CAM and (num_cam > 1), feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
        evaluator.reset()
    elif eval_mode == 'single':
        assert not cfg.TEST.MEAN_FEAT 
        assert not cfg.MODEL.USE_FUSION  # model may have been trained with fusion but to obtain single-modal metrics turn this flag off
        evaluator = []
        for _ in range(num_mode):
            evaluator.append(R1_mAP_eval(num_query, num_mode=1, max_rank=50, use_cam=cfg.TEST.USE_CAM and (num_cam > 1), feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING))
            evaluator[-1].reset()
    else:
        # eval_mode == 'cross'
        assert not cfg.TEST.MEAN_FEAT
        assert not cfg.MODEL.USE_FUSION  # model may have been trained with fusion but to obtain cross-modal metrics turn this flag off
        evaluator = []
        for _ in range(num_mode*num_mode):
            evaluator.append(R1_mAP_eval(num_query, num_mode=1, max_rank=50, use_cam=cfg.TEST.USE_CAM and (num_cam > 1), feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING))
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

    num_shared = cfg.MODEL.SHARED_EMBED_DIM
    
    model.eval()
    print('Running all val')
    for n_iter, (img, pid, camid, camid_tensor, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camid_tensor = camid_tensor.to(device)
            _, feat = model(img, cam_label=camid_tensor) # feat has size (B, num_mode*nbr_cls*H)
            if eval_mode == 'multi':
                evaluator.update((feat, pid, camid))
            elif eval_mode == 'single':
                for m in range(num_mode):
                    evaluator[m].update((feat[:, m*embed_size:(m+1)*embed_size], pid, camid))
            else:
                # eval_mode == 'cross'
                for i, (m, n) in enumerate([(m,n) for m in range(num_mode) for n in range(num_mode)]):
                    if c + len(pid) <= num_query:
                        evaluator[i].update((feat[:, m*embed_size:(m)*embed_size+num_shared], pid, camid))
                    elif c >= num_query:
                        evaluator[i].update((feat[:, n*embed_size:(n)*embed_size+num_shared], pid, camid))
                    else:
                        evaluator[i].update((feat[:num_query-c, m*embed_size:(m)*embed_size+num_shared], pid[:num_query-c], camid[:num_query-c]))
                        evaluator[i].update((feat[num_query-c:, n*embed_size:(n)*embed_size+num_shared], pid[num_query-c:], camid[num_query-c:]))
    torch.cuda.empty_cache()
    
    if eval_mode == 'multi':
        cmc, mAP = evaluator.compute()
        if verbose:
            print('mAP:', mAP, 'Rank-1:', cmc[0], 'Rank-5:', cmc[4], 'Rank-10:', cmc[9])
        return mAP, cmc

    cmc_dict = {}
    map_dict = {}

    if eval_mode == 'single':
        for m in range(num_mode):
            cmc, mAP = evaluator[m].compute()
            if verbose:
                print('Mode', str(m), ':')
                print('mAP:', mAP, 'Rank-1:', cmc[0], 'Rank-5:', cmc[4], 'Rank-10:', cmc[9])
            cmc_dict[m] = cmc
            map_dict[m] = mAP
    elif eval_mode == 'cross':
        for i, (m,n) in enumerate([(m,n) for m in range(num_mode) for n in range(num_mode)]):
            cmc, mAP = evaluator[i].compute()
            if verbose:
                print('Mode', str(m) + '->' + str(n), ':')
                print('mAP:', mAP, 'Rank-1:', cmc[0], 'Rank-5:', cmc[4], 'Rank-10:', cmc[9])
            cmc_dict[(m, n)] = cmc
            map_dict[(m, n)] = mAP

    return map_dict, cmc_dict


def check_classifiers(cfg, model, train_loader, num_query, num_cam):

    device = "cuda"
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    mode_feats = defaultdict(list)
    model.eval()

    classifier = model.classifiers[0].weight
    print(classifier.size())
    cls_0, cls_1, cls_2 = classifier[:, :768], classifier[:,768:768*2], classifier[:, 768*2:768*3]
    print('Running all val')
    pid_to_s1, pid_to_s2, pid_to_s3 = {}, {}, {}
    pid_to_snet = {}
    all_s1, all_s2, all_s3 = [], [], []
    for n_iter, (img, pid, camid, camid_tensor, paths) in enumerate(train_loader):
        with torch.no_grad():
            img = img.to(device)
            camid_tensor = camid_tensor.to(device)
            scores, feat = model(img, cam_label=camid_tensor)
            f1, f2, f3 = feat[:, :768], feat[:, 768:768*2], feat[:, 768*2:768*3]
            s1 = cls_0 @ f1.T
            s2, s3 = cls_1 @ f2.T, cls_2 @ f3.T
            s1 = nn.functional.softmax(s1, dim=0)
            s2 = nn.functional.softmax(s2, dim=0)
            s3 = nn.functional.softmax(s3, dim=0)
            snet = nn.functional.softmax(scores[0], dim=1)

            s1 = s1[list(pid), list(range(s1.size(1)))]
            s2 = s2[list(pid), list(range(s2.size(1)))]
            s3 = s3[list(pid), list(range(s3.size(1)))]
            snet = snet[list(range(snet.size(0))), list(pid)]
            for i, id_ in enumerate(pid):
                path_ = paths[i]
                path_ = path_.split('/')[-1]
                id_ = int(re.match('[0-9]+', path_).group(0))
                if id_ not in pid_to_s1: 
                    pid_to_s1[id_] = []
                    pid_to_s2[id_] = []
                    pid_to_s3[id_] = []
                    pid_to_snet[id_] = []
                pid_to_s1[id_].append(s1[i].item())
                pid_to_s2[id_].append(s2[i].item())
                pid_to_s3[id_].append(s3[i].item())
                pid_to_snet[id_].append(snet[i].item())
                all_s1.append(s1[i].item())
                all_s2.append(s2[i].item())
                all_s3.append(s3[i].item())
            
    for pid in pid_to_s1:
        print(pid, np.mean(pid_to_snet[pid]), np.mean(pid_to_s1[pid]), np.mean(pid_to_s2[pid]), np.mean(pid_to_s3[pid]))
    print(np.mean(all_s1), np.mean(all_s2), np.mean(all_s3))
    raise Exception
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

    

def do_ranking_history(cfg, model, val_loader, num_query, num_cam, num_mode=3):

    device = 'cuda'
    save_path = '/home/ubuntu/jenni/ranking_history/flare/'
    
    evaluator = [RankingHistory(num_query, save_path + 'multi/', max_rank=10, use_cam=cfg.TEST.USE_CAM and (num_cam > 1), feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)]
    evaluator[-1].reset()
    for i in range(num_mode):
        evaluator.append(RankingHistory(num_query, save_path + str(i) + '/', max_rank=10, use_cam=cfg.TEST.USE_CAM and (num_cam > 1), feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING))
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
    img_path_list = []
    print('Running all val')
    for n_iter, (img, pid, camid, camids, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            _, feat = model(img, cam_label=camids)
            evaluator[0].update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)
            for m in range(num_mode):
                if m == 1:
                    new_imgpath = [re.sub('vis', 'ni', p) for p in imgpath]
                elif m == 2:
                    new_imgpath = [re.sub('vis', 'th', p) for p in imgpath]
                else:
                    new_imgpath = imgpath
                evaluator[m+1].update((feat[:, m*embed_size:(m+1)*embed_size], pid, camid, new_imgpath))
    for e in evaluator:
        e.compute()
        raise Exception





