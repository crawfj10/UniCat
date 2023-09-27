import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss, PartSoftmax
import random

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        self.in_planes = 2048
        self.base = ResNet(last_stride=last_stride,
                           block=Bottleneck,
                           layers=[3, 4, 6, 3])
        print('using resnet50 as a backbone')

       # if pretrain_choice == 'imagenet':
       #     self.base.load_param(model_path)
       #     print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, **kwargs):
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        #if self.neck == 'no':
        #    feat = global_feat
        #elif self.neck == 'bnneck':
        #    feat = self.bottleneck(global_feat)
        #cls_score = self.classifier(feat)
        return [global_feat]
        #if self.training:
        #    return cls_score, [global_feat]
        #else:
        #    if self.neck_feat == 'after':
        #        return cls_score, [feat]
        #    else:
        #        return cls_score, [global_feat]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            try:
                self.state_dict()[i].copy_(param_dict[i])
            except:
                print('WARNING:', i, 'was not copied into state dict')
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, num_channel=3):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = cfg.MODEL.EMBED_DIM
        self.ID_hardmining = cfg.MODEL.ID_HARD_MINING
        self.cls_token_num = cfg.MODEL.CLS_TOKEN_NUM
        self.feat_mean = cfg.TEST.MEAN_FEAT
        
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
        self.cfg = cfg
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        if cfg.MODEL.EXTRA_FEAT:
            assert self.cls_token_num == 1
            #self.cls_token_num = len(cfg.MODEL.EXTRA_FEAT) + 1
            self.in_planes *= (len(cfg.MODEL.EXTRA_FEAT) + 1)
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](in_chans=num_channel, stem_conv=cfg.MODEL.STEM_CONV, embed_dim=cfg.MODEL.EMBED_DIM, patch_size=cfg.MODEL.PATCH_SIZE, num_heads=cfg.MODEL.NUM_HEADS, mlp_ratio=cfg.MODEL.MLP_RATIO, qkv_bias=cfg.MODEL.QKV_BIAS, depth=cfg.MODEL.DEPTH, img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, cls_token_num=cfg.MODEL.CLS_TOKEN_NUM, use_pos=cfg.MODEL.USE_POS, extra_feat=cfg.MODEL.EXTRA_FEAT)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        if self.ID_LOSS_TYPE == 'partsoftmax':
            self.classifiers = nn.ModuleList([
            PartSoftmax(self.in_planes, self.num_classes, ratio=cfg.MODEL.PART_ID_RATIO)
            for i in range(self.cls_token_num)])
        else:
            self.classifiers = nn.ModuleList([
                nn.Linear(self.in_planes, self.num_classes, bias=False)
                for _ in range(self.cls_token_num)])
        self.bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(self.in_planes)
            for _ in range(self.cls_token_num)])
        for classifier, bottleneck in zip(self.classifiers, self.bottlenecks):
            if self.ID_LOSS_TYPE == 'softmax':
                classifier.apply(weights_init_classifier)
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
        
        #if pretrain_choice == 'self':
         #   self.load_param(model_path)
    
    def forward(self, x, label=None, cam_label= None, view_label=None, random_pos=False):
        global_feats = self.base(x, cam_label=cam_label, view_label=view_label, random_pos=random_pos)
        return [global_feats[:, i] for i in range(global_feats.shape[1])]


    def load_param(self, trained_path):
        print('loading')
        param_dict = torch.load(trained_path)
        for i in param_dict:
            #if 'classifier' in i or 'bottle' in i: continue
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except Exception as e: 
                print(e)
                print('WARNING:', i, 'was not copied into state dict:', e)
        #print(self.state_dict().keys())
        #raise Exception
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



class build_transformer_local_orig(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange, num_channel=3):
        super(build_transformer_local_orig, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = cfg.MODEL.EMBED_DIM
        class_head_type = cfg.MODEL.JPM_CLASS_HEAD_TYPE
        transformer_head_type = cfg.MODEL.JPM_TRANS_HEAD_TYPE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](in_chans=num_channel, stem_conv=cfg.MODEL.STEM_CONV, embed_dim=cfg.MODEL.EMBED_DIM, patch_size=cfg.MODEL.PATCH_SIZE, num_heads=cfg.MODEL.NUM_HEADS, mlp_ratio=cfg.MODEL.MLP_RATIO, qkv_bias=cfg.MODEL.QKV_BIAS, depth=cfg.MODEL.DEPTH, cls_token_num=cfg.MODEL.CLS_TOKEN_NUM, img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, use_pos=cfg.MODEL.USE_POS, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm

        if transformer_head_type == '0':
            # using different transformer blocks for global and local branches
            self.b1 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
            self.b2 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
        elif transformer_head_type == '1':
            # global and local branch share transformer block
            self.b1 = self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        else:
            raise NotImplementedError

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.divide_length = 4

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        #self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange
        self.random = cfg.MODEL.RANDOM
        self.global_only = cfg.MODEL.GLOBAL_ONLY

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange and not self.random:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        elif self.rearrange:
            rand_ind = list(range(1, features.size(1)))
            random.shuffle(rand_ind)
            x = features[:, rand_ind]
        else:
            x = features[:, 1:]
        
        local_feats = []
        for i in range(self.divide_length):
            local_feat = x[:, i*patch_length:(i+1)*patch_length]
            local_feat = self.b2(torch.cat((token, local_feat), dim=1))
            local_feats.append(local_feat[:, 0])
        
        return global_feat + local_feats
    
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except Exception as e:
                print(str(e))
                print('WARNING:', i, 'was not copied into state dict')
        #raise Exception
        print('Loading pretrained model from {}'.format(trained_path))

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


class CrossAttention(nn.Module):
    # two ViT backbones.. one for intra-modality attention and one for inter

    # then option to use separate CLS token or pooled versions during training
    def __init__(self, num_mode, num_class, camera_num, view_nun, cfg, factory):
        super(CrossAttention, self).__init__()
        raise NotImplementedError
        if cfg.MODEL.NAME == 'transformer':
            if cfg.MODEL.JPM:
                raise NotImplementedError
                print('===========building transformer with JPM module ===========')
                model = build_transformer_local_orig(num_class, camera_num, view_num, cfg, factory,
                                                 rearrange=cfg.MODEL.RE_ARRANGE)
            else:
                print('===========building transformer===========')
                model = build_transformer(num_class, camera_num, view_num, cfg, factory)
        else:
            raise NotImplementedError
            print('===========building ResNet===========')
            model = Backbone(num_class, cfg)
        if cfg.MODEL.SAME_MODEL:
            self.mode_backbones = nn.ModuleList([model for _ in range(num_mode)])
        else:
            self.mode_backbones = nn.ModuleList([copy.deepcopy(model) for _ in range(num_mode)])

        self.fusion_backbone = copy.deepcopy(self.mode_backbones[-1])


    def forward(x):
        # x shape: (B, M, C, H, W)
        raise NotImplementedError 
        for i in range(len(self.fusion_backbone.base.blocks)):
            for m in range(x.size(1)):
                mode_x = self.mode_backbones[m].base.blocks[i](x[:, m], label=label, cam_label=cam_label, view_label=view_label)
            fusion_seq = fuse(mode_xs, fusion_cls)
            fusion_x = self.fusion_backbone.base.blocks[i](fusion_seq)





class EarlyFusion(nn.Module):
    # Problem, how to instantiate? Patch encoders are pre-trained with 3 channel images
    # Also, is 3*n -> 768 big enough ? Need more complex patch encoder?

    def __init__(self, mode_to_channels, num_class, camera_num, view_num, cfg, factory):
        super(EarlyFusion, self).__init__()
        num_channel = sum([c for c in mode_to_channels.values()])
        if cfg.MODEL.NAME == 'transformer':
            if cfg.MODEL.JPM:
                raise NotImplementedError
                print('===========building transformer with JPM module ===========')
                model = build_transformer_local_orig(num_class, camera_num, view_num, cfg, factory,
                                                 rearrange=cfg.MODEL.RE_ARRANGE, num_channel=num_channel)
            else:
                print('===========building transformer===========')
                model = build_transformer(num_class, camera_num, view_num, cfg, factory, num_channel=num_channel)
        else:
            raise NotImplementedError
            print('===========building ResNet===========')
            model = Backbone(num_class, cfg)
        self.backbone = model

        #self.in_planes = 768
        self.in_planes = cfg.MODEL.EMBED_DIM
        num_classifier = cfg.MODEL.CLS_TOKEN_NUM

        self.num_classes = num_class
        self.classifiers = nn.ModuleList([nn.Linear(self.in_planes, self.num_classes, bias=False) for _ in range(num_classifier)])
        self.bottlenecks = nn.ModuleList([nn.BatchNorm1d(self.in_planes) for _ in range(num_classifier)])
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.feat_mean = cfg.TEST.MEAN_FEAT
        for classifier, bottleneck in zip(self.classifiers, self.bottlenecks):
            classifier.apply(weights_init_classifier)
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            raise NotImplementedError
            self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        # x shape: (B, M, C, H, W)
        (_, M, C, H, W) = x.size()
        x = x.reshape(-1, M*C, H, W)
        feats = self.backbone(x, label=label, cam_label=cam_label, view_label=view_label)
        bn_feats = [self.bottlenecks[i](feats[i]) for i in range(len(feats))]

        if self.training:
            scores = []
            for i, bn_feat in enumerate(bn_feats):
                scores.append(self.classifiers[i](bn_feat))
            return scores, feats
        else:

            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.mean(torch.stack(bn_feats, dim=1), dim=1) if self.feat_mean else torch.cat(bn_feats, dim=1)
            else:
                # print("Test with feature before BN")
                return torch.mean(torch.stack(feats, dim=1), dim=1) if self.feat_mean else torch.cat(feats, dim=1)


class LateFusion(nn.Module):
    # Options: pooling can happen via average, max or concatenation
    # CE and triplet losses could be trained separately for each modality (like DCFormer/TransReID's CLS tokens) or pooled versions of CLS tokens could be used for training

    def __init__(self, num_mode, num_class, camera_num, view_num, cfg, factory):
        super(LateFusion, self).__init__()
        if cfg.MODEL.NAME == 'transformer': 
            if cfg.MODEL.JPM:
                raise NotImplementedError
                print('===========building transformer with JPM module ===========')
                model = build_transformer_local_orig(num_class, camera_num, view_num, cfg, factory,
                                                 rearrange=cfg.MODEL.RE_ARRANGE)
            else:
                print('===========building transformer===========')
                model = build_transformer(num_class, camera_num, view_num, cfg, factory)
        else:
            print('===========building ResNet===========')
            model = Backbone(num_class, cfg)

        if cfg.MODEL.SAME_MODEL:
            self.mode_backbones = nn.ModuleList([model for _ in range(num_mode)])
        else:
            self.mode_backbones = nn.ModuleList([model if i == 0 else copy.deepcopy(model) for i in range(num_mode)])
        
        self.fusion_method = cfg.MODEL.FUSION_METHOD
        assert self.fusion_method in ('av', 'max', 'cat')
        
        self.use_fusion = cfg.MODEL.USE_FUSION
        if self.use_fusion:
            print('Training with fusion')
        
        self.in_planes = cfg.MODEL.EMBED_DIM
        if self.use_fusion:
            num_classifier = 1
            if self.fusion_method == 'cat':
                self.in_planes *= num_mode
        else:
            num_classifier = num_mode

        if cfg.MODEL.NAME == 'transformer':
            if cfg.MODEL.JPM:
                # TransReID
                num_classifier *= 5
            else:
                # DCFormer / ViT
                num_classifier *= cfg.MODEL.CLS_TOKEN_NUM
        
        self.num_classes = num_class 
        self.classifiers = nn.ModuleList([nn.Linear(self.in_planes, self.num_classes, bias=False) for _ in range(num_classifier)])
        self.bottlenecks = nn.ModuleList([nn.BatchNorm1d(self.in_planes) for _ in range(num_classifier)])
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.feat_mean = cfg.TEST.MEAN_FEAT
        for classifier, bottleneck in zip(self.classifiers, self.bottlenecks):
            classifier.apply(weights_init_classifier)
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            model_path = cfg.MODEL.PRETRAIN_PATH
            self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None, print_=False):
        # x shape: (B, M, C, H, W), B = batch, M = nbr modality
        mode_feats = []
        for m in range(x.size(1)):
            mode_x = self.mode_backbones[m](x[:, m], label=label, cam_label=cam_label, view_label=view_label)
            # mode_x is list of length nbr_cls of tensors size (B, h)  h = hidden dim
            mode_feats.append(mode_x)
             
        feats = []
        bn_feats = []
        if self.use_fusion:
            # corresponding CLS tokens of each modality will be fused 
            # iterating over CLS tokens
            for i in range(len(mode_feats[0])):
                token_feats = [mf[i] for mf in mode_feats]
                if self.fusion_method in ('av', 'max'):
                    token_feats = torch.stack(token_feats)
                    # token_feats shape: (M, B, h)
                    # fusing along modality dim
                    if self.fusion_method == 'av':
                        token_feats = torch.mean(token_feats, dim=0)
                    else:
                        token_feats = torch.max(token_feats, dim=0)
                else:
                    # self.fusion_method == 'cat'
                    # concatenating tokens along hidden dim
                    token_feats = torch.cat(token_feats, dim=-1)
                bn_token_feats = self.bottlenecks[i](token_feats)
                feats.append(token_feats)
                bn_feats.append(bn_token_feats)
        else:
            c = 0
            # iterating over modalities
            for i in range(len(mode_feats)):
                # iterating over modality CLS tokens
                for j in range(len(mode_feats[i])):
                    feat_ij = mode_feats[i][j]
                    bn_feat_ij = self.bottlenecks[c](feat_ij)
                    feats.append(feat_ij)
                    bn_feats.append(bn_feat_ij)
                    c += 1
        assert len(bn_feats) == len(self.classifiers)
        scores = [self.classifiers[i](bn_feat) for i, bn_feat in enumerate(bn_feats)]
        if self.training:
            return scores, feats
        else:
            
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return scores, torch.mean(torch.stack(bn_feats, dim=1), dim=1) if self.feat_mean else torch.cat(bn_feats, dim=1)
            else:
                # print("Test with feature before BN")
                return scores, torch.mean(torch.stack(feats, dim=1), dim=1) if self.feat_mean else torch.cat(feats, dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            #print(i)
            #if 'classifier' in i or 'bottleneck' in i: continue
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except Exception as e:
                print(str(e))
                print('WARNING:', i, 'was not copied into state dict')
        #raise Exception
        print('Loading pretrained model from {}'.format(trained_path))

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_mode, num_class, camera_num, view_num=0):
    if cfg.MODEL.LATE_FUSION or not cfg.MODEL.USE_FUSION:
        return LateFusion(num_mode, num_class, camera_num, view_num, cfg, __factory_T_type)
    mode_to_channels = {}
    for n in range(num_mode):
        mode_to_channels[n] = 3  # TODO: fix me. I don't want NIR to have 3 channels
    return EarlyFusion(mode_to_channels, num_class, camera_num, view_num, cfg, __factory_T_type)
