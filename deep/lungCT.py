import math
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import nibabel
import torch
from scipy import ndimage
import SimpleITK as sitk
from skimage.external import tifffile as tif
from sklearn import preprocessing
import matplotlib.pyplot as plt


def debug(img, img_name):
    os.makedirs('./debug', exist_ok=True)
    name=os.path.splitext(os.path.basename(img_name.replace('\\', '/')))[0]
    tif.imsave(f'./debug/{name}.tif', np.moveaxis(img.astype('int8'), -1, 0), bigtiff=False)

def isNaN(string):
    return string != string

def image_resample(input_image, ref_image=None, iso_voxel_size=1, is_label=False):
    resample_filter = sitk.ResampleImageFilter()

    input_spacing = input_image.GetSpacing()
    input_direction = input_image.GetDirection()
    input_origin = input_image.GetOrigin()
    input_size = input_image.GetSize()

    output_spacing = [iso_voxel_size, iso_voxel_size, iso_voxel_size]
    output_origin = input_origin
    output_direction = input_direction
    output_size = np.ceil(np.asarray(input_size) * np.asarray(input_spacing) / np.asarray(output_spacing)).astype(int)

    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetOutputOrigin(output_origin)
    resample_filter.SetSize(output_size.tolist())
    resample_filter.SetOutputDirection(output_direction)
    if is_label:
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_image = resample_filter.Execute(input_image)
    else:
        resample_filter.SetInterpolator(sitk.sitkLinear)
        resampled_image = resample_filter.Execute(input_image)

    return resampled_image

class LIDC_CT(Dataset):
    def __init__(self, phase_type, root_dir, img_list, sets, transforms, 
                        use_filtered=False, use_mask=False, isotropic_resample=False):
        
        if isinstance(img_list, str):
            df = pd.read_csv(img_list)
        else:
            df = img_list

        self.img_list = df['cropped_img_pth'].to_list()
        
        if 'cropped_mask_pth' in df.columns and use_mask:
            self.masks = df['cropped_mask_pth'].to_list()
        else:
            self.masks = [None] * len(self.img_list)  # No masks provided

        # Check if 'filtered_pth' exists, otherwise use cropped_pth
        if 'filtered_pth' in df.columns:
            self.filtered_img_list = df['filtered_pth'].to_list()
        else:
            self.filtered_img_list = [None] * len(self.img_list)  # No filtered images available

#         self.bbox_x = df['index_x'].to_list()
#         self.bbox_y = df['index_y'].to_list()
#         self.bbox_z = df['index_z'].to_list()
#         self.size_x = df['size_x'].to_list()
#         self.size_y = df['size_y'].to_list()
#         self.size_z = df['size_z'].to_list()

        self.use_filtered = use_filtered
        self.use_mask = use_mask and any(self.masks)  # Only use masks if they exist and use_mask is True
        self.isotropic_resample = isotropic_resample

        print("Processing {} data for {}".format(len(self.img_list), phase_type))
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase
        self.transformations = transforms
    
    def __len__(self):
        return len(self.img_list)

    def _check_for_mask(self, mask):
        if len(np.unique(mask)) > 2:
            raise ValueError('More than 2 labels found in Mask ...')
        return mask

    def rescale_img_to_rgb(self, img):
        img = (img - img.min()) / (img.max() - img.min())
        img *= 255.0
        img = img.astype(np.uint8)
        return img
    
    def load_image(self, img_name, label_name=None, idx=None):
        # Load image and (optionally) mask
        if 'nii' in img_name:
            if self.isotropic_resample:
                img = sitk.ReadImage(img_name)
                ct_spacing = np.asarray(img.GetSpacing())
                iso_voxel_size = ct_spacing.min()

                if label_name and self.use_mask:
                    mask = sitk.ReadImage(label_name)
                    mask = image_resample(mask, iso_voxel_size=iso_voxel_size, is_label=True)
                    mask = sitk.GetArrayFromImage(mask).astype("float32")
                    mask = mask.transpose(1,2,0)
                else:
                    mask = None

                # Resample image
                img = image_resample(img, iso_voxel_size=iso_voxel_size)
                img = sitk.GetArrayFromImage(img).astype("float32")
                img = img.transpose(1,2,0)

            else:
                img = nibabel.load(img_name).get_fdata()
                img = self.rescale_img_to_rgb(img)
                mask = None
                if label_name and self.use_mask:
                    mask = nibabel.load(label_name).get_fdata()
                
        elif 'nrrd' in img_name:
            img = sitk.ReadImage(img_name)
            img = sitk.GetArrayFromImage(img).astype("float32")
            img = img.transpose(1,2,0)

            if label_name and self.use_mask:
                mask = sitk.ReadImage(label_name)
                mask = sitk.GetArrayFromImage(mask).astype("float32")
                mask = mask.transpose(1,2,0)
            else:
                mask = None

        else:
            img = np.load(img_name)[0]
            img = self.rescale_img_to_rgb(img)
            mask = None
            if label_name and self.use_mask:
                mask = np.load(label_name)[0]

        # Resize and pad the images
        img = self.__pad__(img)
        img = self.__resize_data__(img)

        if mask is not None:
            mask = self.__pad__(mask)
            mask = self.__resize_data__(mask)
        
        return img, mask
    
        
    def __getitem__(self, idx):
        # Read image and optionally mask
        img_name = os.path.join(self.root_dir, self.img_list[idx].replace('\\', '/'))

        # Use filtered image if use_filtered is True and the image is available
        if self.use_filtered and self.filtered_img_list[idx] and not pd.isna(self.filtered_img_list[idx]):
            img_name = os.path.join(self.root_dir, self.filtered_img_list[idx].replace('\\', '/'))

        label_name = None
        if self.use_mask and self.masks[idx]:
            label_name = os.path.join(self.root_dir, self.masks[idx].replace('\\', '/'))

        # Load image and mask
        img, mask = self.load_image(img_name, label_name, idx)

        # If no mask, just use the image
        if self.use_mask and mask is not None:
            img = np.multiply(img, mask)

        # Convert to tensor and apply transformations
        img = torch.from_numpy(np.expand_dims(img.transpose(2,0,1), axis=0)).type(torch.FloatTensor)
        if mask is not None:
            mask = torch.from_numpy(np.expand_dims(mask.transpose(2,0,1), axis=0)).type(torch.FloatTensor)
        else:
            mask = np.zeros_like(img)  # Create a mask of zeros with the same shape as the image

        if self.transformations:
            # Random horizontal flipping
            if random.random() > 0.6:
                img = TF.hflip(img)
                if mask is not None:
                    mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.6:
                img = TF.vflip(img)
                if mask is not None:
                    mask = TF.vflip(mask)

        return img, mask, self.img_list[idx].replace('\\', '/')

    def __resize_data__(self, data, resize=[64,64,32]):
        """
        Resize the data to the input size , [64,64,32]
        """ 
        [height, width, depth] = data.shape
        scale = [resize[0]*1.0/height, resize[1]*1.0/width, resize[2]*1.0/depth]  
        data = ndimage.interpolation.zoom(data, scale, order=0)
        return data

    def __crop_data__(self, data, resize=[80,80,32]):
        """
        Crop the data to input size
        """

        [height, width, depth] = data.shape
        origin = [height//2-resize[0]//2, width//2-resize[1]//2, depth//2-resize[2]//2]
        data = data[origin[0]:origin[0]+resize[0], origin[1]:origin[1]+resize[1], origin[2]:origin[2]+resize[2]]

        return data

    def __pad__(self, data):
        # original image shape
        w, h, z = data.shape

        # maximum dimension of all the images
        scale = np.max([w, h])

        # equal padding on left/right, top/bottom
        hp = int((scale - w)+10 / 2)
        vp = int((scale - h)+10 / 2)

        # pad with zeros
        padding = ((hp, hp), (vp, vp), (0,0))
        return np.pad(data, padding, mode='constant', constant_values=0)


class LIDC_CT_Labels(LIDC_CT):
    def __init__(self, phase_type, root_dir, img_list, sets, transforms, label,
                        use_filtered=False, use_mask=False, isotropic_resample=False):
        
        if isinstance(img_list, str):
            df = pd.read_csv(img_list)
        else:
            df = img_list

        self.img_list = df['cropped_img_pth'].to_list()
        self.masks = df['cropped_mask_pth'].to_list()
        self.filtered_img_list = df['filtered_pth'].to_list()

        self.bbox_x = df['index_x'].to_list()
        self.bbox_y = df['index_y'].to_list()
        self.bbox_z = df['index_z'].to_list()
        self.size_x = df['size_x'].to_list()
        self.size_y = df['size_y'].to_list()
        self.size_z = df['size_z'].to_list()

        self.labels = df[label].to_list()
        self.categories = sorted(df[label].unique())

        self.use_filtered = use_filtered
        self.use_mask = use_mask
        self.isotropic_resample = isotropic_resample

        print("Processing {} data for {}".format(len(self.img_list), phase_type))
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase
        self.transformations = transforms

    def __getitem__(self, idx):
        # read image and labels

        img_name = os.path.join(self.root_dir, self.img_list[idx].replace('\\', '/'))
        if self.use_filtered:
            if not isNaN(self.filtered_img_list[idx]):
                img_name = os.path.join(self.root_dir, self.filtered_img_list[idx].replace('\\', '/'))

        label_name = os.path.join(self.root_dir, self.masks[idx].replace('\\', '/'))
        assert os.path.isfile(img_name)
        assert os.path.isfile(label_name)

        img, mask = self.load_image(img_name, label_name, idx)

        if self.use_mask:
            img = np.multiply(img, mask)
    
        # extend dimension and convert to tensor
        img = torch.from_numpy(np.expand_dims(img.transpose(2,0,1), axis=0)).type(torch.FloatTensor)
        mask = torch.from_numpy(np.expand_dims(mask.transpose(2,0,1), axis=0)).type(torch.FloatTensor)
        
        # apply transformations
        if self.transformations:

            # Random horizontal flipping
            if random.random() > 0.6:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.6:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
        
        label = self.labels[idx]
        semantic_feature = [[label]]
        encoder = preprocessing.OneHotEncoder(categories=[self.categories], sparse=False)
        onehot = encoder.fit_transform(semantic_feature)[0]

        return img, mask, onehot, self.img_list[idx].replace('\\', '/')
