import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import os
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
from deep.build_unet_model import UNet
from deep.build_resnet_model import generate_model
from deep.setting import parse_opts 
from deep.lungCT import LIDC_CT
import argparse


def getIoU(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

# +
def test(test_loader, model, output, dataset_csv, mode=None):
    # run inference pipeline
    print('Test: {} batches per epoch'.format(len(test_loader)))

    all_features = []

    with torch.no_grad():
        model.eval()
        for volumes, label_masks, volume_names in tqdm(test_loader):
            volumes = volumes.cuda()
            out_masks, latent = model(volumes)
            
            if mode=='unet_lidc':
                output_prob = torch.sigmoid(out_masks).detach().cpu().numpy()
                output_prob_thresh = (output_prob > 0.5) * 1
                
            for i in range(len(volume_names)):

                features_temp = latent[i].view(-1, 1).cpu().numpy() 
                features_temp = np.squeeze(features_temp)
                all_features.append(features_temp)
                
    feature_df = pd.DataFrame(all_features)
    
    # Load the original dataset CSV to get the 'PID' column
    df = pd.read_csv(dataset_csv)
    # Ensure the number of rows match
    assert len(df) == len(feature_df), "Mismatch between dataset and extracted features"
    feature_df['PID'] = df['PID']
    # Reorder columns
    cols = feature_df.columns.tolist() 
    cols = ['PID'] + [col for col in cols if col != 'PID']  # Move 'PID' to the front
    feature_df = feature_df[cols]
    
#     Save the feature DataFrame as a CSV
    feature_df.to_csv(output, index=False)

    print(f"Feature extraction complete. Saved feature dataset to {output}")



# -

if __name__ == '__main__':
    # setting
    sets = parse_opts()
    mode = sets.mode
    dataset_csv = sets.input_csv
    output_name = sets.output_csv
    if not os.path.exists('/'.join(output_name.split('/')[:-1])):
        os.makedirs('/'.join(output_name.split('/')[:-1]))
    data_root = sets.dataroot
    
    df = pd.read_csv(dataset_csv)
    
    assert "cropped_img_pth" in df.columns, "Cropped_img_pth column missing in input csv"

    # additional options
    use_filtered=False
    use_mask=True
    isotropic_resample=False
    
       # get model
    os.environ["CUDA_VISIBLE_DEVICES"]=str(sets.gpu_id[0]) 
    if mode == 'unet_lidc':
        model = UNet(in_nc=1, out_nc=1, nf=64).cuda()
        model = nn.DataParallel(model)
        path_to_weight = './deep/weights/UNet_epoch_106_best_weight_0.646983.pth.tar'
        checkpoint = torch.load(path_to_weight)
        model.load_state_dict(checkpoint['state_dict'])
    elif mode == 'resnet_34_med3d': 
        sets.model_depth = 34
        model, _ = generate_model(sets)
        path_to_weight = './deep/weights/resnet_34_23dataset.pth'
        net_dict = model.state_dict()
        pretrain = torch.load(path_to_weight)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
        new_parameters = []
    elif mode == 'resnet_34_kinetics': 
        sets.model_depth = 34
        sets.n_seg_classes = 400
        sets.input_D = 3
        model, _ = generate_model(sets)
        path_to_weight = './deep/weights/resnet-34-kinetics.pth'
        net_dict = model.state_dict()
        pretrain = torch.load(path_to_weight)
        pretrain['state_dict']['module.conv1.weight'] = pretrain['state_dict']['module.conv1.weight'][:, :1, ...]
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
        new_parameters = []
    

    print('Model weights loaded!')

    # run inference
    test_dataset = LIDC_CT('Test', data_root, dataset_csv, sets, transforms=None, 
                                    use_filtered=use_filtered, use_mask=use_mask,
                                    isotropic_resample=isotropic_resample)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=sets.num_workers, pin_memory=False)
    test(test_loader, model, output_name, dataset_csv, mode=mode)


