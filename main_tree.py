import os
import argparse
from copy import deepcopy

import torch

from prototree.prototree import ProtoTree
from util.log import Log
from util.args import get_args, save_args, get_optimizer, set_seed, get_scheduler
from util.data import get_dataloaders
from util.init import init_tree
from util.net import get_network, freeze
from util.visualize import gen_vis
from util.analyse import *
from util.save import *
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.test import eval, eval_fidelity
from prototree.prune import prune
from prototree.project import project, project_with_class_constraints
from prototree.upsample import upsample
from prototree.test2 import eval_ensemble


def check_nan(tensor):
    if torch.isnan(tensor).any():
        print('NaN value detected!')


def run_tree(args=None):
    args = args or get_args()
    set_seed(args.seed)

    # Create a logger
    log = Log(args.log_dir)
    print('Log dir: ', args.log_dir, flush=True)
    log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'mean_train_acc',
                   'mean_train_crossentropy_loss_during_epoch')
    save_args(args, log.metadata_dir)

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
    log.log_message('Device used: ' + str(device))
    log.log_message('Poincare ball curvature magnitude c = ' + str(args.curvature))

    log_prefix = 'log_train_epochs'
    log_loss = log_prefix + '_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)
    features_net, add_on_layers = get_network(num_channels, args)
    tree = ProtoTree(num_classes=len(classes),
                     feature_net=features_net,
                     args=args,
                     add_on_layers=add_on_layers)

    for xs, ys in trainloader:
        features_out = features_net(xs)
        print(f'Feature output shape (after feature_net): {features_out.shape}')
        check_nan(xs)
        break

    tree = tree.to(device=device)

    optimizer, params_to_freeze, params_to_train = get_optimizer(tree, args)
    scheduler = get_scheduler(optimizer, args)
    tree, epoch = init_tree(tree, optimizer, scheduler, device, args)

    tree.save(f'{log.checkpoint_dir}/tree_init')
    log.log_message('Max depth %s, so %s internal nodes and %s leaves'
                    % (args.depth, tree.num_branches, tree.num_leaves))
    analyse_output_shape(tree, trainloader, log, device)

    leaf_labels = dict()
    best_train_acc = 0.
    best_test_acc = 0.
    test_losses = []

    if epoch < args.epochs + 1:
        '''
            TRAIN AND EVALUATE TREE
        '''
        epochs_without_improvement = 0
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epoch, args.epochs + 1):
            log.log_message('\nEpoch %s' % str(epoch))
            freeze(tree, epoch, params_to_freeze, params_to_train, args, log)
            log_learning_rates(optimizer, args, log)

            if tree._kontschieder_train:
                train_info = train_epoch_kontschieder(tree, trainloader, optimizer, epoch,
                                                      args.disable_derivative_free_leaf_optim,
                                                      device, log, log_prefix)
            else:
                train_info = train_epoch(tree, trainloader, optimizer, epoch,
                                         args.disable_derivative_free_leaf_optim, device, log,
                                         log_prefix, cluster_coef=args.cluster_coef)
            save_tree(tree, optimizer, scheduler, epoch, log, args)
            best_train_acc = save_best_train_tree(tree, optimizer, scheduler, best_train_acc,
                                                  train_info['train_accuracy'], log)
            leaf_labels = analyse_leafs(tree, epoch, len(classes), leaf_labels,
                                        args.pruning_threshold_leaves, log)

            # Evaluate tree on the validation/test set every 10 epochs or on the last epoch
            if args.epochs > 160:
                do_eval = epoch % 10 == 0 or epoch == args.epochs
            else:
                do_eval = True
            if do_eval:
                eval_info = eval(tree, testloader, epoch, device, log)
                if 'loss' in eval_info:
                    test_losses.append(eval_info['loss'])
                else:
                    print(f"Warning: 'loss' key not found in eval_info for epoch {epoch}")
                original_test_acc = eval_info['test_accuracy']

                # Reduce LR on plateau and stop early when validation stops improving
                if eval_info['test_accuracy'] > best_test_acc + args.min_delta:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc,
                                                    eval_info['test_accuracy'], log)
                log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'],
                               train_info['train_accuracy'], train_info['loss'])
                scheduler.step(eval_info['test_accuracy'])
                if epochs_without_improvement >= args.patience:
                    log.log_message(f'Early stopping at epoch {epoch} after {args.patience} '
                                    'epochs without validation improvement.')
                    break
            else:
                log.log_values('log_epoch_overview', epoch, 'n.a.',
                               train_info['train_accuracy'], train_info['loss'])

    else:  # tree was loaded and not trained, so evaluate only
        '''
            EVALUATE TREE
        '''
        eval_info = eval(tree, testloader, epoch, device, log)
        original_test_acc = eval_info['test_accuracy']
        best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc,
                                            eval_info['test_accuracy'], log)
        log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], 'n.a.', 'n.a.')

    log.log_message('Training Finished. Best training accuracy was %s, best test accuracy was %s\n'
                    % (str(best_train_acc), str(best_test_acc)))
    trained_tree = deepcopy(tree)
    leaf_labels = analyse_leafs(tree, epoch + 1, len(classes), leaf_labels,
                                args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)

    '''
        PRUNE
    '''
    prune(tree, args.pruning_threshold_leaves, log)
    name = 'pruned'
    save_tree_description(tree, optimizer, scheduler, name, log)
    pruned_tree = deepcopy(tree)
    leaf_labels = analyse_leafs(tree, epoch + 2, len(classes), leaf_labels,
                                args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    pruned_test_acc = eval_info['test_accuracy']
    pruned_tree = tree

    '''
        PROJECT
    '''
    project_info, tree = project_with_class_constraints(deepcopy(pruned_tree), projectloader, device, args, log)
    name = 'pruned_and_projected'
    save_tree_description(tree, optimizer, scheduler, name, log)
    pruned_projected_tree = deepcopy(tree)
    average_distance_nearest_image_smallest3(project_info, tree, log)
    leaf_labels = analyse_leafs(tree, epoch + 3, len(classes), leaf_labels,
                                args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    pruned_projected_test_acc = eval_info['test_accuracy']

    return (trained_tree.to('cpu'), pruned_tree.to('cpu'), pruned_projected_tree.to('cpu'),
            original_test_acc, pruned_test_acc, pruned_projected_test_acc, project_info)


if __name__ == '__main__':
    args = get_args()
    run_tree(args)
