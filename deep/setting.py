

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help = "Deep Learning Model type")
    parser.add_argument('--input_csv', type=str, help = "Input csv file path")
    parser.add_argument('--output_csv', type=str, help = "Output csv file path")
    parser.add_argument('--dataroot', type=str, help = "Path to cropped image data folder")
    
    parser.add_argument(
        '--data_root',
        default='./nlst/preprocessed',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--img_test_list',
        default='./nlst/NLST_preprocessed_dataset.csv',
        type=str,
        help='Path for image list file')

    parser.add_argument(
        '--n_seg_classes',
        default=1,
        type=int,
        help="Number of segmentation classes"
    )
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--num_workers',
        default=16,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--input_D',
    default=32,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=64,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=64,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.'
    )
    parser.add_argument(
        '--pretrain_path',
        default='',
        type=str,
        help=
        'Path for pretrained model.'
    )
    parser.add_argument(
        '--new_layer_names',
        default=[], # 'conv_segment' 
        type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=34,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.')
    args = parser.parse_args()
    
    args.save_folder = "./deep/custom/{}_{}".format(args.model, args.model_depth)
    return args
