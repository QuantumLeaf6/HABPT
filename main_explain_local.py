from prototree.prototree import ProtoTree
from util.data import get_dataloaders
from util.visualize_prediction import gen_pred_vis1
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from shutil import copy
from copy import deepcopy
import os

def get_local_expl_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Explain a prediction')
    parser.add_argument('--prototree',
                        type=str,
                        default=r'E:\zty\run_prototree500\tree_1\checkpoints\best_test_model',
                        help='Directory to trained ProtoTree')
    parser.add_argument('--log_dir',
                        type=str,
                        default=r'E:\zty\run_prototree500\tree_1',
                        help='The directory in which results should be logged. Should be same log_dir as loaded ProtoTree')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',
                        help='Data set on which the ProtoTree was trained')
    parser.add_argument('--sample_dir',
                        type=str,
                        default = r'E:/zty/data/CUB_200_2011/dataset1/test-800/008.TUM/TUM-YVLKCAPM.tif',
                        help='Directory to image to be explained, or to a folder containing multiple test images')
    parser.add_argument('--results_dir',
                        type=str,
                        default='local_explanations',
                        help='Directory where local explanations will be saved')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Resize images to this size')
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='upsampling_results',
                        help='Directoy for saving the prototypes, patches and heatmaps')
    parser.add_argument('--upsample_threshold',
                        type=float,
                        default=0.98,
                        help='Threshold (between 0 and 1) for visualizing the nearest patch of an image after upsampling. The higher this threshold, the larger the patches.')
    parser.add_argument('--log_probabilities',
                        action='store_true',
                        help='Flag that uses log probabilities when set. Useful when getting NaN values.')
    args = parser.parse_args()
    return args

def explain_local(args):
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
        
    # Log which device was actually used
    print('Device used: ',str(device))

    # Load trained ProtoTree
    tree = ProtoTree.load(args.prototree).to(device=device)
    # Obtain the dataset and dataloaders
    args.batch_size=64 #placeholder
    args.augment = True #placeholder
    _, _, _, classes, _ = get_dataloaders(args)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    test_transform = transform_no_augment =         transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
            # transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
            ]),
            transforms.ToTensor(),
            normalize,
        ])
    
    sample = test_transform(Image.open(args.sample_dir)).unsqueeze(0).to(device)

    gen_pred_vis1(tree, sample, args.sample_dir, args.results_dir, args, classes)


if __name__ == '__main__':
    args = get_local_expl_args()
    try:
        Image.open(args.sample_dir)
        print("Image to explain: ", args.sample_dir)
        explain_local(args)
    except: #folder is not image
        class_name = args.sample_dir.split('/')[-1]
        if not os.path.exists(os.path.join(os.path.join(args.log_dir, args.results_dir),class_name)):
            os.makedirs(os.path.join(os.path.join(args.log_dir, args.results_dir),class_name))
        for filename in os.listdir(args.sample_dir):
            print(filename)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                args_1 = deepcopy(args)
                args_1.sample_dir = args.sample_dir+"/"+filename
                args_1.results_dir = os.path.join(args.results_dir, class_name)
                explain_local(args_1)


if __name__ == '__main__':
    args = get_local_expl_args()
    try:
        Image.open(args.sample_dir)
        print("Image to explain: ", args.sample_dir)
        explain_local(args)
    except: #folder is not image
        class_name = args.sample_dir.split('/')[-1]
        if not os.path.exists(os.path.join(os.path.join(args.log_dir, args.results_dir),class_name)):
            os.makedirs(os.path.join(os.path.join(args.log_dir, args.results_dir),class_name))
        for filename in os.listdir(args.sample_dir):
            print(filename)
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                args_1 = deepcopy(args)
                args_1.sample_dir = args.sample_dir+"/"+filename
                args_1.results_dir = os.path.join(args.results_dir, class_name)
                explain_local(args_1)