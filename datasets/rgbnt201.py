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

class RGBNT201(BaseImageDataset):
    
    camera_map = {1:0, 2:1, 3:2, 4:3}
    def __init__(self, root='', verbose=True, pid_begin = 0, mode='r', **kwargs):
        super(RGBNT201, self).__init__()
        
        assert mode in ('r', 'n', 't', 'rn', 'rt', 'nt', 'rnt')
        self.modalities = mode

        t_flds, v_flds = [], []
        v_files = None
        for m in mode:
            t_flds.append(osp.join(root, 'train/' + m.upper()))
            v_flds.append(osp.join(root, 'validate/' + m.upper()))
            m_v_files = [f for fld in os.listdir(v_flds[-1]) for f in os.listdir(v_flds[-1] + '/' + fld)]
            assert v_files is None or set(m_v_files) == set(v_files)
            v_files = m_v_files
        
        q_files = os.listdir(osp.join(root, 'rgbir/query'))
        g_files = [f for f in v_files if f not in q_files]
        #g_paths = [f for fld in v_flds for f in os.listdir(fld) if f not in q_files]
        
        #self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_files(t_flds, relabel=True)
        query = self._process_files(v_flds, files=v_files, relabel=False, allowed_cam=None)
        gallery = self._process_files(v_flds, files=v_files, relabel=False, allowed_cam=None)

        if verbose:
            print("=> RGBT201 loaded")
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

    def _process_files(self, mode_paths, files=None, relabel=False, allowed_cam=None, allowed_mode=None):
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid2label = {}
        dataset = []
        
        if files is None:
            files = None
            for mode_path in mode_paths:
                mode_files = glob.glob(osp.join(mode_path, '*', '*.jpg'))
                mode_files = [re.search('[^/]+$', f).group(0) for f in mode_files]
                assert files is None or set(files) == set(mode_files)
                files = mode_files
        
        for img_file in sorted(files):
            pid, camid = pattern.search(img_file).groups(0) 
            camid = self.camera_map[int(camid)]
            if allowed_cam is not None and camid not in allowed_cam: continue
            old_pid = pid[-4:] 
            pid = int(pid)
            if relabel: 
                if pid not in pid2label: pid2label[pid] = len(pid2label)
                pid = pid2label[pid]
            img_paths = [osp.join(mode_path, old_pid, img_file) for mode_path in mode_paths]
            dataset.append((img_paths, self.pid_begin + pid, camid))
        return dataset


