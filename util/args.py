import os
import argparse
import pickle
import numpy as np
import random
import torch.optim
from torch.optim.optimizer import Optimizer
import argparse
import torch
from geoopt.optim import RiemannianAdam
import geoopt
"""
    Utility functions for handling parsed arguments

"""
def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a ProtoTree')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',#CUB-200-2011
                        help='Data set on which the ProtoTree should be trained')
    parser.add_argument('--seed',
                        type=int,
                        default=3407,
                        help='Random seed for reproducibility')
    parser.add_argument('--net',
                        type=str,
                        default='resnet50_inat',
                        help='Base network used in the tree. Pretrained network on iNaturalist is only available for resnet50_inat (default). Others are pretrained on ImageNet. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, densenet121, densenet169, densenet201, densenet161, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn or vgg19_bn')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,#64
                        help='Batch size when training the model using minibatch gradient descent')
    parser.add_argument('--depth',
                        type=int,
                        default=3,
                        help='The tree is initialized as a complete tree of this depth')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='The number of epochs the tree should be trained')
    # parser.add_argument('--optimizer',
    #                     type=str,
    #                     default='AdamW',
    #                     help='The optimizer that should be used when training the tree')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='The optimizer learning rate for training the prototypes')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.001,
                        help='The optimizer learning rate for training the 1x1 conv layer and last conv layer of the underlying neural network (applicable to resnet50 and densenet121)')
    parser.add_argument('--lr_net',
                        type=float,
                        default=1e-5, 
                        help='The optimizer learning rate for the underlying neural network')
    parser.add_argument('--lr_pi',
                        type=float,
                        default=0.001,
                        help='The optimizer learning rate for the leaf distributions (only used if disable_derivative_free_leaf_optim flag is set')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='The optimizer momentum parameter (only applicable to SGD)')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./results/HABPT',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--W1',
                        type=int,
                        default =1,
                        help='Width of the prototype. Correct behaviour of the model with W1 != 1 is not guaranteed')
    parser.add_argument('--H1',
                        type=int,
                        default = 1,
                        help='Height of the prototype. Correct behaviour of the model with H1 != 1 is not guaranteed')
    parser.add_argument('--num_features',
                        type=int,
                        default = 256,
                        help='Depth of the prototype and therefore also depth of convolutional output')
    parser.add_argument('--milestones',
                        type=str,
                        default='',
                        help='The milestones for the MultiStepLR learning rate scheduler')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        help='The factor by which the learning rate is reduced when validation stops improving')

    # Hyperbolic geometry (negative curvature is stored as a positive magnitude in the Poincare ball)
    parser.add_argument('--curvature',
                        type=float,
                        default=6.0,
                        help='Magnitude of the negative curvature of the Poincare ball. '
                             'Set to 0 for the Euclidean control setting.')
    # Reproducibility: default seeds used for repeated runs
    parser.add_argument('--seeds',
                        type=str,
                        default='42,512,1024,3407',
                        help='Comma separated random seeds used for repeated runs')
    # Warm-up and early stopping
    parser.add_argument('--patience',
                        type=int,
                        default=10,
                        help='Early stopping patience in epochs without validation improvement')
    parser.add_argument('--min_delta',
                        type=float,
                        default=1e-4,
                        help='Minimum improvement in validation accuracy counted as progress')
    parser.add_argument('--cluster_coef',
                        type=float,
                        default=0.8,
                        help='Weight of the prototype clustering loss in the total loss')
    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained backbone network')
    parser.add_argument('--state_dict_dir_tree',
                        type=str,
                        # default=r'.\pretrain_model\P\best_test_model_7class',
                        #1zhong
                        # default=r'.\pretrain_model\P\best_test_model1',
                        # 2zhong
                        default=r'.\pretrain_model\P\best_test_model2',
                        # default='',
                        # 3zhong
                        # default=r'.\pretrain_model\P\best_test_model3',
                        # 4zhong
                        # default=r'.\pretrain_model\P\best_test_model4',
                        # 5zhong
                        # default=r'.\pretrain_model\P\best_test_model5',
                        help='The directory containing a state dict (checkpoint) with a pretrained prototree. Note that training further from a checkpoint does not seem to work correctly. Evaluating a trained prototree does work.')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default=6,
                        help='Number of warm-up epochs during which the pretrained features_net stays frozen')
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='upsampling_results',
                        help='Directoy for saving the prototypes, patches and heatmaps')
    parser.add_argument('--upsample_threshold',
                        type=float,
                        default=0.98,
                        help='Threshold (between 0 and 1) for visualizing the nearest patch of an image after upsampling. The higher this threshold, the larger the patches.')
    parser.add_argument('--disable_pretrained',
                        action='store_true',#action='store_false',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset). When not set, resnet50_inat is initalized with weights from iNaturalist2017. Other networks are initialized with weights from ImageNet'
                        )
    parser.add_argument('--disable_derivative_free_leaf_optim',
                        action='store_true',
                        help='Flag that optimizes the leafs with gradient descent when set instead of using the derivative-free algorithm'
                        )
    parser.add_argument('--kontschieder_train',
                        action='store_true',
                        help='Flag that first trains the leaves for one epoch, and then trains the rest of ProtoTree (instead of interleaving leaf and other updates). Computationally more expensive.'
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        default='RiemannianAdam',
                        choices=['RiemannianAdam'],
                        help='The optimizer that should be used when training the tree')
    parser.add_argument('--kontschieder_normalization',
                        action='store_true',
                        help='Flag that disables softmax but uses a normalization factor to convert the leaf parameters to a probabilitiy distribution, as done by Kontschieder et al. (2015). Will iterate over the data 10 times to update the leaves. Computationally more expensive.'
                        )
    parser.add_argument('--log_probabilities',
                        action='store_false',
                        help='Flag that uses log probabilities when set. Useful when getting NaN values.'
                        )
    parser.add_argument('--pruning_threshold_leaves',
                        type=float,
                        default=0.95,
                        help='An internal node will be pruned when the maximum class probability in the distributions of all leaves below this node are lower than this threshold.')
    parser.add_argument('--nr_trees_ensemble',
                        type=int,
                        default=5,
                        help='Number of ProtoTrees to train and (optionally) use in an ensemble. Used in main_ensemble.py') 
    args = parser.parse_args()
    args.milestones = get_milestones(args)
    return args

def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_scheduler(optimizer, args: argparse.Namespace):
    """Reduce the learning rate by ``args.gamma`` when validation accuracy stalls."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.gamma,
        patience=args.patience,
        min_lr=1e-7,
    )


"""
    Parse the milestones argument to get a list
    :param args: The arguments given
    """
def get_milestones(args: argparse.Namespace):
    if args.milestones != '':
        milestones_list = args.milestones.split(',')
        for m in range(len(milestones_list)):
            milestones_list[m]=int(milestones_list[m])
    else:
        milestones_list = []
    return milestones_list

def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    

def load_args(directory_path: str) -> argparse.Namespace:
    """
    Load the pickled arguments from the specified directory
    :param directory_path: The path to the directory from which the arguments should be loaded
    :return: the unpickled arguments
    """
    with open(directory_path + '/args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args


def get_optimizer(tree, args: argparse.Namespace) -> torch.optim.Optimizer:
    """
    Construct the optimizer as dictated by the parsed arguments
    :param tree: The tree that should be optimized
    :param args: Parsed arguments containing hyperparameters. The '--optimizer' argument specifies which type of
                 optimizer will be used. Optimizer specific arguments (such as learning rate and momentum) can be passed
                 this way as well
    :return: the optimizer corresponding to the parsed arguments, parameter set that can be frozen, and parameter set of the net that will be trained
    """

    optim_type = args.optimizer
    # create parameter groups
    params_to_freeze = []
    params_to_train = []

    dist_params = []
    for name, param in tree.named_parameters():
        if 'dist_params' in name:
            dist_params.append(param)
    # set up optimizer
    if 'resnet50' in args.net:
        # freeze resnet50 except last convolutional layer
        for name, param in tree._net.named_parameters():
            if 'layer4.2' not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)

        if optim_type == 'SGD':
            paramlist = [
                {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay,
                 "momentum": args.momentum},
                {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay,
                 "momentum": args.momentum},
                {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay,
                 "momentum": args.momentum},
                {"params": tree.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0, "momentum": 0}]
            if args.disable_derivative_free_leaf_optim:
                paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})
        else:
            paramlist = [
                {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                {"params": tree.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0}]

            if args.disable_derivative_free_leaf_optim:
                paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})

    elif args.net == 'densenet121':
        # freeze densenet121 except last convolutional layer
        for name, param in tree._net.named_parameters():
            if 'denseblock4' not in name and 'norm5' not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)

        paramlist = [
            {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": tree.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0}]
        if args.disable_derivative_free_leaf_optim:
            paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})

    else:
        paramlist = [
            {"params": tree._net.parameters(), "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": tree.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0}]
        if args.disable_derivative_free_leaf_optim:
            paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})

    if optim_type == 'SGD':
        return torch.optim.SGD(paramlist,
                               lr=args.lr,
                               momentum=args.momentum), params_to_freeze, params_to_train
    if optim_type == 'Adam':
        return torch.optim.Adam(paramlist, lr=args.lr, eps=1e-07), params_to_freeze, params_to_train
    if optim_type == 'AdamW':
        return torch.optim.AdamW(paramlist, lr=args.lr, eps=1e-07,
                                 weight_decay=args.weight_decay), params_to_freeze, params_to_train

    if optim_type == 'RiemannianAdam':
        paramlist = [
            {"params": params_to_freeze, "lr": args.lr_net},
            {"params": params_to_train, "lr": args.lr_block},
            {"params": tree._add_on.parameters(), "lr": args.lr_block},
            {"params": tree.prototype_layer.parameters(), "lr": args.lr}]
        if args.disable_derivative_free_leaf_optim:
            paramlist.append({"params": dist_params, "lr": args.lr_pi})

        return geoopt.optim.RiemannianAdam(paramlist, lr=args.lr), params_to_freeze, params_to_train

    raise Exception('Unknown optimizer argument given!')


