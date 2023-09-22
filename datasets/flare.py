# encoding: utf-8

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import re
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as T
import os

class Flare(BaseImageDataset):
    '''
        AKA WMVeID863 (Flare-Aware Cross-modal Enhancement Network for Multi-spectral Vehicle Re-identification)
    '''
    camera_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
    mode_flds = {'r': 'vis', 'n': 'ni', 't': 'th'}
    def __init__(self, root='', verbose=True, pid_begin = 0, mode='r', **kwargs):
        super(Flare, self).__init__()
        
        t_paths, q_paths, g_paths = [], [], []
        assert mode in ('r', 'n', 't', 'rn', 'rt', 'nt', 'rnt')

        t_paths = [osp.join(root, 'train', f) for f in os.listdir(osp.join(root, 'train'))]
        q_paths = [osp.join(root, 'query', f) for f in os.listdir(osp.join(root, 'query'))]
        g_paths = [osp.join(root, 'gallery', f) for f in os.listdir(osp.join(root, 'gallery'))]
        self.modalities = mode
        
        #self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_files(t_paths, relabel=True)
        query = self._process_files(q_paths, relabel=False, allowed_cam=None)
        gallery = self._process_files(g_paths, relabel=False, allowed_cam=None)

        if verbose:
            print("=> Flare (WMVeID863) loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_mode, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, _, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, _, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_files(self, id_paths, relabel=False, allowed_cam=None, allowed_mode=None):
        

        id_map = {}
        dataset = []

        for id_fld in id_paths:
            id_ = int(re.search('[0-9]+$', id_fld).group(0))
            if relabel:
                if id_ not in id_map: id_map[id_] = self.pid_begin + len(id_map)
                id_ = id_map[id_]
            samples = os.listdir(id_fld + '/vis')
            keep = False
            for sample in samples:
                camid = int(re.search('(?<=v)[0-9]', sample).group(0))

                sample_paths = [id_fld + '/' + self.mode_flds[mode] + '/' + sample for mode in self.modalities] 
                if any([not os.path.exists(p) for p in sample_paths]):
                    # dumb but happens
                    continue
                keep = True
                dataset.append((sample_paths, id_, camid))
            if not keep:
                print('deleting')
                del id_map[id_]
        return dataset


