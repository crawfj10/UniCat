from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            #print(img_path)
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        num_mode = None
        for imgs, pid, camid in data:
            pids += [pid]
            cams += [camid]
            
            assert num_mode is None or num_mode == len(imgs)
            num_mode = len(imgs)
        pids = set(pids)
        cams = set(cams)
        
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        
        return num_pids, num_imgs, num_mode, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_mode, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_mode, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gal_mode, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        imgs = []
        for img_path in img_paths:
            img = read_image(img_path)

            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        return imgs, pid, camid, img_paths[0]



class MergedImageDataset(Dataset):

    def __init__(self, dataset, transforms, ind_to_transform):
        raise NotImplementedError
        self.dataset = dataset
        self.transforms = transforms
        self.ind_to_transform = ind_to_transform

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        

        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        t_ind = self.ind_to_transform[index]
        if self.transforms[t_ind] is not None:
            img = self.transforms[t_ind](img)

        return img, pid, camid, trackid, img_path


