import torch
import torch.utils.data as tordata
import os.path as osp
import os
import numpy as np
from functools import partial

from torch.utils.data import Dataset


'''
class CTPatchDataset(tordata.Dataset):
    def __init__(self, npy_root, hu_range, transforms=None):
        self.transforms = transforms
        hu_min, hu_max = hu_range
        data = torch.from_numpy(np.load(npy_root).astype(np.float32) - 1024)
        # normalize to [0, 1]
        data = (torch.clamp(data, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
        self.low_doses, self.full_doses = data[0], data[1]

    def __getitem__(self, index):
        low_dose, full_dose = self.low_doses[index], self.full_doses[index]
        if self.transforms is not None:
            low_dose = self.transforms(low_dose)
            full_dose = self.transforms(full_dose)
        return low_dose, full_dose

    def __len__(self):
        return len(self.low_doses)'''


class CTPatchDataset(Dataset):
    def __init__(self, npy_root, noise_level, target_transform=None):
        super(CTPatchDataset, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        #input_dir = 'input/5000'
        input_dir = os.path.join('input', str(noise_level))  # Use noise level as subdirectory
        

        clean_files = sorted(os.listdir(os.path.join(npy_root, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(npy_root, input_dir)))
        print (f"Input dir at: {os.path.join(npy_root, input_dir)}")
        
        print("Data loader val init")


        self.clean_filenames = [os.path.join(npy_root, gt_dir, x) for x in clean_files if is_numpy_file(x)]
        self.noisy_filenames = [os.path.join(npy_root, input_dir, x) for x in noisy_files if is_numpy_file(x)]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_npy(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_npy(self.noisy_filenames[tar_index])))

        # Add a channel dimension for grayscale images
        if clean.dim() == 2:  # If it's a 2D tensor, add a channel dimension
            clean = clean.unsqueeze(0)  # Shape becomes (1, 512, 512)
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(0)  # Shape becomes (1, 512, 512)

        #print("Data loader val get item")

                
        clean_filename = osp.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = osp.split(self.noisy_filenames[tar_index])[-1]
        #In RGB the image would be ((H, W, C) C = number of chanels (3), and the permute is used to get to shape (C, H, W).
        # In this case of Gray values, the shape is already (C, H, W).
        #clean = clean.permute(2,0,1)
        #noisy = noisy.permute(2,0,1)
        
        #combined = torch.stack([noisy, clean], dim=0)
        return noisy, clean, noisy_filename, clean_filename # WORKS?




#data_root = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset')
#dataset_dict = {
#    'cmayo_train_64': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/train_64.npy')),
#    'cmayo_test_512': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/test_512.npy')),
#}


def load_npy(filepath):
    img = np.load(filepath)
    return img

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])