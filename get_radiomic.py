#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import radiomics
import six
import SimpleITK as sitk
from radiomics import featureextractor
import sys
import os
import argparse
import nrrd
import matplotlib.pyplot as plt
import nibabel as nib


# %%

def read_file(file_path):
#     print(file_path)
    img = nib.load(file_path)
    data = img.get_fdata()
    header = img.header
    return data, header

def get_radiomics(extractor, image, label):
    feats = {}
    featureVector = extractor.execute(image, label)
    for (key, val) in six.iteritems(featureVector):
        if key.startswith("original") or key.startswith("log") or key.startswith("wavelet") :
            feats[key] = val
    return feats

def scale_HU(image_array, lower_bound=-1000, upper_bound=500):
    image_array[image_array > upper_bound] = upper_bound
    image_array[image_array < lower_bound] = lower_bound
    return image_array

def get_radiomics_feat(image_dir, masks, ids, scale_HU_value=True):
    all_nodules_feats_dict = {'pid':[]}
    for idx, mask_pth in enumerate(masks):
        pid = ids[idx]
#         print(pid)
#         print(image_dir[idx])
#         print(masks[idx])
        # read image
        data, header = read_file(image_dir[idx])
        if scale_HU_Value:
            data = scale_HU(data) # scale to 0-1500 range
        sitk_image = sitk.GetImageFromArray(data)
        # read mask
        mask = nib.load(mask_pth)
        mask_data = mask.get_fdata()
        mask_header = mask.header
        assert data.shape == mask_data.shape
        sitk_mask = sitk.GetImageFromArray(mask_data)
            
        # extract features
        feats = get_radiomics(extractor, sitk_image, sitk_mask)
        feats_values = [float(x) for x in feats.values()]
        
        # write header
        if idx == 0:
            for header in feats.keys():
                all_nodules_feats_dict[header] = []
        # write values
        for feat_key in feats:
            all_nodules_feats_dict[feat_key].append(float(feats[feat_key]))
        all_nodules_feats_dict['pid'].append(pid)
        print(f"PID:{pid} complete!")
    return all_nodules_feats_dict

if __name__ == '__main__':
    # setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, help = "Input csv file path")
    parser.add_argument('--output_csv', type=str, help = "Output csv file path")
    
    args = parser.parse_args()
    input_csv = args.input_csv
    output_dset = args.output_csv
    if not os.path.exists('/'.join(output_dset.split('/')[:-1])):
        os.makedirs('/'.join(output_dset.split('/')[:-1]))

    # additional options
    scale_HU_Value = True
    paramPath = './CT.yaml'

    # read radiomics configuration
#     print('Parameter file, absolute path:', os.path.abspath(paramPath))
    extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)

    input_df = pd.read_csv(input_csv)
    pids = input_df["PID"].values
    print('num of cases:', len(pids))
    
#     for idx, cnd in enumerate(cnds):
#     print(f"counter:{idx}")
    imgs = input_df["img_path"].values
    masks = input_df["seg_path"].values
#         print('Reading cnd:', path_to_read)
    df_dict = get_radiomics_feat(imgs, masks, pids, scale_HU_Value)
    df_dict = pd.DataFrame(df_dict)
    df_dict = df_dict.rename(columns = {'pid':'PID'})
    df_dict.to_csv(output_dset, index=False)
    
    print(f"Feature extraction complete. Saved feature dataset to {output_dset}")




# %%




