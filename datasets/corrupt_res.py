import torch
import torchvision.transforms as T
import numpy as np
from torchvision.utils import save_image
class ResCorrupter(object):

    def __init__(self, mode='mixed'):

        self.up_range = [0.3, 1.0]
        self.dn_range = [0.1, 0.3]
        self.counter = 0
        self.mode = mode

    def __call__(self, x):
        if self.mode == 'high' or (self.mode == 'mixed' and self.counter % 2 == 0):
            new_img = self.down_up(x, self.up_range)
        elif self.mode == 'low' or (self.mode == 'mixed' and self.counter % 2 == 1):
            new_img = self.down_up(x, self.dn_range)
        else:
            raise Exception('Unknown mode given to ResCorrupter')
        self.counter += 1
        return new_img
    
    def down_up(self, img, dn_range):
        dn_factor = np.random.uniform(low=dn_range[0], high=dn_range[1])
        dn_img = T.Resize([int(img.size(1)*dn_factor), int(img.size(2)*dn_factor)])(img)
        up_img = T.Resize([img.size(1), img.size(2)])(dn_img)
       # if self.counter < 100:
       #     print(self.counter, dn_factor)
       #     save_image(up_img.float(), '/home/ubuntu/jenni/res_test/market_res_' + str(self.counter) + '_' + str(dn_factor) + '.jpg')
       #     save_image(img.float(), '/home/ubuntu/jenni/res_test/market_res_' + str(self.counter) + '.jpg')
       # print(dn_factor, self.counter)
        return up_img
