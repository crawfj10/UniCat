# encoding: utf-8

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy, TemperatureCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .dissimilar_loss import Dissimilar
from .centroid_triplet_loss import CentroidTripletLoss
from .shuffle_loss import shuffle_loss

def make_loss(cfg, num_classes, num_views=0):
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    dissimilar = Dissimilar(dynamic_balancer=cfg.MODEL.DYNAMIC_BALANCER)
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    if 'centroid' in cfg.MODEL.METRIC_LOSS_TYPE:
        centroid_triplet = CentroidTripletLoss()
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    if cfg.MODEL.IF_TEMPERATURE_SOFTMAX == 'on':
        xent = TemperatureCrossEntropy()
        print("temperature softmax on")

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, print_=False):
            if 'ce' in cfg.MODEL.METRIC_LOSS_TYPE:
                if cfg.MODEL.IF_TEMPERATURE_SOFTMAX == 'on':
                    if isinstance(score, list):
                        if isinstance(score[0], tuple):
                            ID_LOSS = [xent(scor, lbl) for scor, lbl in score]
                        else:
                            ID_LOSS = [xent(scor, target, t) for scor, t in zip(score, cfg.MODEL.TEMPERATURE)]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)
                else:
                    if isinstance(score, list):
                        if isinstance(score[0], tuple):
                            ID_LOSS = [F.cross_entropy(scor, lbl) for scor, lbl in score]
                        else:
                            ID_LOSS = [F.cross_entropy(scor, target) for scor in score]
                        # ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSSes = ID_LOSS
                    else:
                        ID_LOSS = F.cross_entropy(score, target)
                        ID_LOSSes = [ID_LOSS]
                id_loss = [l * cfg.MODEL.ID_LOSS_WEIGHT for l in ID_LOSSes]
            else:
                id_loss = None
            if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
                if isinstance(feat, list):
                    TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                    # TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    TRI_LOSSes = TRI_LOSS
                else:
                    TRI_LOSS = triplet(feat, target)[0]
                    TRI_LOSSes = [TRI_LOSS]
                tri_loss = [tri_loss * cfg.MODEL.TRIPLET_LOSS_WEIGHT for tri_loss in TRI_LOSSes]
            else:
                tri_loss = None

            if 'centroid' in cfg.MODEL.METRIC_LOSS_TYPE:
                if isinstance(feat, list):
                    cen_loss = [centroid_triplet(feats, target)[0] for feats in feat]
                else:
                    cen_loss = [centroid_triplet(feats, target)[0]]
                cen_loss = [l * cfg.MODEL.TRIPLET_LOSS_WEIGHT for l  in cen_loss]
            else:
                cen_loss = None
                        
            if len(feat) > 1:
                dis_loss = cfg.MODEL.DIVERSE_CLS_WEIGHT*dissimilar(torch.stack(feat, dim=1))
            else:
                #dis_loss = None
            #if len(feat) == 1:
                dis_loss = cfg.MODEL.DIVERSE_CLS_WEIGHT*torch.abs(torch.mean(torch.sum(feat[0], dim=1) / torch.norm(feat[0], dim=1)))
            
            if tri_loss is None and cen_loss is None and id_loss is None:
                raise Exception('Unknown loss type')
            return id_loss, tri_loss, cen_loss, dis_loss
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


