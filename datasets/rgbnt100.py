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

class RGBNT100(BaseImageDataset):
    
    #camera_map = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 8: 3} # for harder eval
    # 1 - front, 2 - back, 3 - side, 4 - side, 5 - front/side, 6 - back/side, 7 - back/side, 8 - front/side
    #camera_map = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4}  # for harder eval 
    camera_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
    def __init__(self, root='', verbose=True, pid_begin = 0, mode='r', **kwargs):
        super(RGBNT100, self).__init__()
        
        t_paths, q_paths, g_paths = [], [], []
        assert mode in ('r', 'n', 't', 'rn', 'rt', 'nt', 'rnt')
        self.modalities = mode
        for m in mode:
            t_paths.append(osp.join(root, 'reid_set_' + m.upper() + '/bounding_box_train/'))
            g_paths.append(osp.join(root, 'reid_set_' + m.upper() + '/bounding_box_test/'))
            q_paths.append(osp.join(root, 'reid_set_' + m.upper() + '/query/'))
        
        #self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_files(t_paths, relabel=True)
        query = self._process_files(q_paths, relabel=False, allowed_cam=None)
        gallery = self._process_files(g_paths, relabel=False, allowed_cam=None)

        if verbose:
            print("=> RGBT100 loaded")
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

    def _process_files(self, mode_paths, relabel=False, allowed_cam=None, allowed_mode=None):
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid2label = {}
        dataset = []
        
        img_files = None
        for mode_path in mode_paths:
            mode_files = glob.glob(osp.join(mode_path, '*.jpg'))
            mode_files = [re.search('[^/]+$', f).group(0) for f in mode_files]
            assert img_files is None or set(img_files) == set(mode_files)
            img_files = mode_files
            
        for img_file in sorted(img_files):
            pid, camid = map(int, pattern.search(img_file).groups())            
            if camid not in self.camera_map: continue
            camid = self.camera_map[camid]
            if allowed_cam is not None and camid not in allowed_cam: continue
            
            if relabel: 
                if pid not in pid2label: pid2label[pid] = len(pid2label)
                pid = pid2label[pid]
            img_paths = [osp.join(mode_path, img_file) for mode_path in mode_paths]
            dataset.append((img_paths, self.pid_begin + pid, camid))
        return dataset


