import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import image


class RGBDataset(Dataset):
    def __init__(self, dataset_dir, has_gt):
        """
        In:
            dataset_dir: string, train_dir, val_dir, and test_dir in segmentation.py.
                         Be careful the images are stored in the subfolders under these directories.
            has_gt: bool, indicating if the dataset has ground truth masks.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        """
        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.dataset_dir = dataset_dir
        self.has_gt = has_gt
      
       
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_rgb, std_rgb)])
        self.dataset_length = os.listdir

    def __len__(self):
        if 'train' in self.dataset_dir:
            self.dataset_length = len(os.listdir('/home/jupyter/dataset/train/rgb'))
            
        if 'val' in self.dataset_dir:
            self.dataset_length = len(os.listdir('/home/jupyter/dataset/val/rgb'))
        if 'test' in self.dataset_dir:
            self.dataset_length = len(os.listdir('/home/jupyter/dataset/test/rgb'))
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Think about how to associate idx with the file name of images.
        """
        # TODO: read RGB image and ground truth mask, apply the transformation, and pair them as a sample.
        #rgb_img = None
        #gt_mask = None
        
        
      
        if 'train' in self.dataset_dir:
            rgbimg = "/home/jupyter/dataset/train/rgb/" + str(idx) + '_rgb.png'
            gtmask = "/home/jupyter/dataset/train/gt/" + str(idx) + '_gt.png'
        if 'val' in self.dataset_dir:
            rgbimg = "/home/jupyter/dataset/val/rgb/" + str(idx) + '_rgb.png'
            gtmask = "/home/jupyter/dataset/val/gt/" + str(idx) + '_gt.png'
        if 'test' in self.dataset_dir:
            rgbimg = "/home/jupyter/dataset/test/rgb/" + str(idx) + '_rgb.png'
            #gtmask = "/home/jupyter/dataset/test/gt/" + str(idx) + '_gt.png'
        
        rgb_img = image.read_rgb(rgbimg)

    
        if self.has_gt is False or 'test' in self.dataset_dir:
            
            rgb_img = self.transform(rgb_img)
            sample = {'input': rgb_img}
            
            
        if self.has_gt is True and 'test' not in self.dataset_dir: #Added the second part
            rgb_img = self.transform(rgb_img)
            gt_mask = image.read_mask(gtmask)
            gt_mask = torch.LongTensor(gt_mask)
            sample = {'input': rgb_img, 'target': gt_mask}
            
            
        return sample
