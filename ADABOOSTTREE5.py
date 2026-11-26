# from shutil import copy
# from copy import deepcopy
# import torch
# import os
# import numpy as np
# from prototree.prototree import ProtoTree
# from util.log import Log
# from util.args import get_args, save_args, get_optimizer, set_seed, get_scheduler
import os
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prototree.prototree import ProtoTree
from util.data import get_dataloaders, get_data
# from util.analyse import analyse_ensemble
# from prototree.prototree import ProtoTree
from util.log import Log
from tqdm import tqdm
from util.args import get_args, save_args, get_optimizer, set_seed, get_scheduler
from util.data import get_dataloaders,get_dataloaders1
from util.init import init_tree
from util.net import get_network, freeze
from util.visualize import gen_vis
from util.analyse import *
from util.save import *
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.test2 import eval, eval_fidelity, eval_ensemble
from prototree.prune import prune
from prototree.project import project, project_with_class_constraints
from prototree.upsample import upsample
import torch
from shutil import copy
from copy import deepcopy
import gc

# def set_seed(seed_value):
#     """Set seed for reproducibility."""
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed(seed_value)
#     # torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
#     np.random.seed(seed_value)  # Numpy module.
#     random.seed(seed_value)  # Python random module.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
def check_nan(tensor):
    if torch.isnan(tensor).any():
        print("NaN value detected!")

# from main_tree import run_tree
def train_epoch(tree: ProtoTree,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                w:list,
                disable_derivative_free_leaf_optim: bool,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch',
                cluster_coef: float = 0.8
                ) -> dict:
    tree = tree.to(device)
    # Make sure the model is in eval mode
    tree.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    # Create a log if required
    log_loss = f'{log_prefix}_losses'

    nr_batches = float(len(train_loader))
    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)


    # all_args = get_args()
    # args = deepcopy(all_args)
    # trainset, projectset, testset, classes, shape = get_data(all_args)
    # cuda = not args.disable_cuda and torch.cuda.is_available()
    # trainloader = torch.utils.data.DataLoader(trainset,
    #                                           batch_size=1,
    #                                           shuffle=True,
    #                                           pin_memory=cuda,
    #                                           # sampler=sampler
    #                                           )
    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      desc=progress_prefix + ' %s' % epoch,
                      ncols=0)
    # Make sure the model is in train mode


    true = []
    pred = []

    # Normalized training-sample weights drive the AdaBoost sample reweighting
    w_in = np.asarray(w, dtype=float)
    w_in = w_in / w_in.sum()
    for i, (xs, ys) in train_iter:
        tree.train()


        # Reset the gradients
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        # Perform a forward pass through the network
        ys_pred,info = tree.forward(xs)

        # Compute the loss: cross-entropy plus the weighted prototype clustering loss
        if tree._log_probabilities:
            crs_loss = F.nll_loss(ys_pred, ys)
        else:
            crs_loss = F.nll_loss(torch.log(ys_pred), ys)
        cluster_loss = info['min_distances'].mean()
        loss = crs_loss + cluster_coef * cluster_loss
        # Compute the gradient
        loss.backward()
        # Update model parameters
        torch.nn.utils.clip_grad_norm_(tree.parameters(), max_norm=0.5)
        #check for NaN values in gradients
        for name, param in tree.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN values found in gradients of {name}")
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(tree.parameters(), max_norm=1.0)
        optimizer.step()

        if not disable_derivative_free_leaf_optim:
            # Update leaves with the derivative-free algorithm
            #Make sure the tree is in eval mode
            tree.eval()
            with torch.no_grad():
                target = eye[ys] #shape (batchsize, num_classes)
                for leaf in tree.leaves:
                    if tree._log_probabilities:
                        # log version
                        update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - ys_pred, dim=0))
                    else:
                        update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred, dim=0)
                    leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                    F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
                    leaf._dist_params += update

        # Count the number of correct classifications
        ys_pred = torch.argmax(ys_pred, dim=1)


    # print(pred)
    # x=0

        correct = torch.sum(torch.eq(ys_pred, ys))
        acc = correct.item() / float(len(xs))
        train_iter.set_postfix_str(f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}')
        # Compute metrics over this batch
        total_loss += loss.item()
        total_acc += acc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)
        miss = []
        true_label = []
        pred_label = []
        true.append(ys)
        pred.append(ys_pred)
    # Flatten the per-batch labels; batch sizes may be smaller than the configured size
    for batch_true, batch_pred in zip(true, pred):
        true_label.extend(batch_true.tolist())
        pred_label.extend(batch_pred.tolist())
    print('pred_label:', len(pred_label))

# if total_acc/float(i+1) > best_train_acc:
#     best_epoch = epoches
#     best_train_acc = total_acc/float(i+1)


    for q in range(len(pred_label)):

    # for j in range(len(true)):
    #         print(k,j,pred[k][j],true[k][j],pred[k][j] != true[k][j])
    #         for x in (pred[k][j] != true[k][j]):
    #             missr = int(x)
    #             print(missr)
    # # for i in range(len(train_loader)):
    # #     for j in range(64):
    # #         # missr = [int(x) for x in (pred[i][j] != true[i][j])]
    #
        if pred_label[q] != true_label[q]:
            x = 1
        else:
            x = 0
        missr = int(x)
            #             print(missr)
#         print(i,j,pred[i][j] != true[i][j],x)
        miss.append(missr)
    # print("len(miss):",len(miss))
    # print(miss,w)
    print('Shape of w_in:', w_in.shape)
    print('Length of miss:', len(miss))
    error = np.dot(w_in, miss)
    # error = float(format(error, '.4f'))
    print('error=', error)
    print("w_epochinit[0]",w[0])
    if error<0.875:
        theta = 0.5 * (np.log((1 - error) / error))
        # theta =float(format(theta, '.4f'))
        print('theta=', theta)
        # print(true_label[5000], pred_label[5000])
        print("len(w)",len(w))
        w_after = [0 for x in range(len(w))]
        print("len(w_after)", len(w_after))
        # for j in range(len(train_loader)):
        #     for i in range(32):

        for k in range(len(w)):
            # print(f"len(true_label): {len(true_label)}, len(w): {len(w)}")
            if true_label[k] == pred_label[k]:
                w_after[k] = w[k] * np.exp(-theta)

            else:
                w_after[k] = w[k] * np.exp(theta)
        # print(k,w[k])
                    # w_after.append(w[k])
        print('w-after[0]:', w_after[0])
        sum=0
        for i in range(len(w_after)):
            sum += w_after[i]
        print("sum",sum)
        for p in range(len(w_after)):
            w_after[p] /= sum
        print('w-after[0]-stan:', w_after[0])
    else:
        theta=0.0000001
        w_after = w
    train_info['theta'] = theta
    train_info['w'] = w_after
    train_info['loss'] = total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    return train_info

def run_tree(w_init,args=None):
    args = args or get_args()
    set_seed(args.seed)
    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
    log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'mean_train_acc',
                   'mean_train_crossentropy_loss_during_epoch')
    # Log the run arguments
    save_args(args, log.metadata_dir)
    if not args.disable_cuda and torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    log.log_message('Device used: ' + str(device))

    # Create a log for logging the loss values
    log_prefix = 'log_train_epochs'
    log_loss = log_prefix + '_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    # Obtain the dataset and dataloaders
    trainloader, trainloader1,projectloader, testloader, classes, num_channels = get_dataloaders1(args,w_init)
    # Create a convolutional network based on arguments and add 1x1 conv layer
    features_net, add_on_layers = get_network(num_channels, args)
    # Create a ProtoTree
    tree = ProtoTree(num_classes=len(classes),
                     feature_net=features_net,
                     args=args,
                     add_on_layers=add_on_layers)

    # Check for NaN values in the data after loading
    for xs, ys in trainloader:
        #
        features_out = features_net(xs)
        print(f"Feature output shape (after feature_net): {features_out.shape}")

        check_nan(xs)
        break
    # Prototypes and features are embedded onto the Poincare ball inside the
    # projection layer via the exponential map, so no parameter conversion is
    # needed here. The tree itself is trained with a Riemannian optimizer.
    tree = tree.to(device=device)
    # creat a ProtoForest
    # num_tree = 3
    # trees=[]................................................................................
    # for _ in range(num_tree):
    #     tree = ProtoTree(num_classes=len(classes),
    #                      feature_net=features_net,
    #                      args=args,
    #                      add_on_layers=add_on_layers)
    #     tree = tree.to(device =device)
    #     trees.append(tree)
    # Determine which optimizer should be used to update the tree parameters
    optimizer, params_to_freeze, params_to_train = get_optimizer(tree, args)
    scheduler = get_scheduler(optimizer, args)
    tree, epoch = init_tree(tree, optimizer, scheduler, device, args)

    tree.save(f'{log.checkpoint_dir}/tree_init')
    log.log_message(
        "Max depth %s, so %s internal nodes and %s leaves" % (args.depth, tree.num_branches, tree.num_leaves))
    analyse_output_shape(tree, trainloader, log, device)

    leaf_labels = dict()
    best_train_acc = 0.
    best_test_acc = 0.
    best_train_epoch = 0.
    w_best = []
    theta = 0
    # w = []
    if epoch < args.epochs + 1:
        '''
            TRAIN AND EVALUATE TREE
        '''
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epoch, args.epochs + 1):
            log.log_message("\nEpoch %s" % str(epoch))
            # Freeze (part of) network for some epochs if indicated in args
            freeze(tree, epoch, params_to_freeze, params_to_train, args, log)
            log_learning_rates(optimizer, args, log)

            # Train tree
            if tree._kontschieder_train:
                train_info = train_epoch_kontschieder(tree, trainloader, optimizer, epoch,
                                                      args.disable_derivative_free_leaf_optim, device, log, log_prefix)
            else:
                train_info = train_epoch(tree, trainloader, optimizer, epoch, w_init, args.disable_derivative_free_leaf_optim,
                                         device, log, log_prefix, cluster_coef=args.cluster_coef)
            save_tree(tree, optimizer, scheduler, epoch, log, args)
            best_train_acc, best_train_epoch, w_best,theta_best= save_best_train_tree1(tree, optimizer, scheduler, train_info['w'],train_info['theta'],best_train_acc,
                                                                             epoch,
                                                                             train_info['train_accuracy'], log)
            leaf_labels = analyse_leafs(tree, epoch, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
            print("w_best[0]:",w_best[0])
            # Evaluate tree 
            if epoch > 100:  # Change the condition to check if epoch is greater than 200
                if epoch % 10 == 0 or epoch == args.epochs:  # Every 10 epochs or the last epoch
                    eval_info = eval(tree, testloader, epoch, device, log)
                    original_test_acc = eval_info['test_accuracy']
                    best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc,
                                                        eval_info['test_accuracy'], log)
                    log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'],
                                   train_info['train_accuracy'], train_info['loss'])
                else:
                    # Log training progress without test evaluation
                    log.log_values('log_epoch_overview', epoch, "n.a.", train_info['train_accuracy'],
                                   train_info['loss'])
            else:
                # Log training progress without test evaluation for epochs <= 200
                log.log_values('log_epoch_overview', epoch, "n.a.", train_info['train_accuracy'],
                               train_info['loss'])

            scheduler.step(train_info['train_accuracy'])
            # # Evaluate tree 
            # if args.epochs >100:
            #     if epoch % 10 == 0 or epoch == args.epochs:
            #         eval_info = eval(tree, testloader, epoch, device, log)
            #         original_test_acc = eval_info['test_accuracy']
            #         best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc,
            #                                             eval_info['test_accuracy'], log)
            #         log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'],
            #                        train_info['train_accuracy'], train_info['loss'])
            #     else:
            #         log.log_values('log_epoch_overview', epoch, "n.a.", train_info['train_accuracy'],
            #                        train_info['loss'])
            # else:
            #     eval_info = eval(tree, testloader, epoch, device, log)
            #     original_test_acc = eval_info['test_accuracy']
            #     best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc,
            #                                         eval_info['test_accuracy'], log)
            #     log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'],
            #                    train_info['loss'])
            #
            # scheduler.step()

    else:  # tree was loaded and not trained, so evaluate only
        '''
            EVALUATE TREE
        '''
        eval_info = eval(tree, testloader, epoch, device, log)
        original_test_acc = eval_info['test_accuracy']
        best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
        log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], "n.a.", "n.a.")

    '''
        EVALUATE AND ANALYSE TRAINED TREE
    '''
    log.log_message(
        "Training Finished. Best training accuracy was %s, best training epoch was %s\n,best test accuracy was %s\n" % (
        str(best_train_acc), int(best_train_epoch), str(best_test_acc)))
    trained_tree = deepcopy(tree)
    leaf_labels = analyse_leafs(tree, epoch + 1, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)

    '''
        PRUNE
    '''
    pruned = prune(tree, args.pruning_threshold_leaves, log)
    name = "pruned"
    save_tree_description(tree, optimizer, scheduler, name, log)
    pruned_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree
    leaf_labels = analyse_leafs(tree, epoch + 2, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    pruned_test_acc = eval_info['test_accuracy']

    pruned_tree = tree

    '''
        PROJECT
    '''
    project_info, tree = project_with_class_constraints(deepcopy(pruned_tree), projectloader, device, args, log)
    name = "pruned_and_projected"
    save_tree_description(tree, optimizer, scheduler, name, log)
    pruned_projected_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree with projected prototypes
    average_distance_nearest_image(project_info, tree, log)
    leaf_labels = analyse_leafs(tree, epoch + 3, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    # y_true = eval_info['y_true']
    # y_pred = eval_info['y_pred']
    pruned_projected_test_acc = eval_info['test_accuracy']

    eval_info_samplemax = eval(tree, testloader, name, device, log, 'sample_max')
    get_avg_path_length(tree, eval_info_samplemax, log)
    eval_info_greedy = eval(tree, testloader, name, device, log, 'greedy')
    get_avg_path_length(tree, eval_info_greedy, log)
    fidelity_info = eval_fidelity(tree, testloader, device, log)
    # the_best_test_acc = eval_info['best_test_acc']
    # Upsample prototype for visualization
    project_info = upsample(tree, project_info, projectloader, name, args, log)
    # visualize tree
    gen_vis(tree, name, args, classes)

    return trained_tree.to('cpu'), w_best, theta_best,pruned_tree.to('cpu'),pruned_projected_tree.to(
        'cpu'), original_test_acc, pruned_test_acc, pruned_projected_test_acc, project_info, eval_info_samplemax, eval_info_greedy, fidelity_info

def run_tree1(w_init,args=None):
    args = args or get_args()
    set_seed(args.seed)
    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
    log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'mean_train_acc',
                   'mean_train_crossentropy_loss_during_epoch')
    # Log the run arguments
    save_args(args, log.metadata_dir)
    if not args.disable_cuda and torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    log.log_message('Device used: ' + str(device))

    # Create a log for logging the loss values
    log_prefix = 'log_train_epochs'
    log_loss = log_prefix + '_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    # Obtain the dataset and dataloaders
    trainloader, trainloader1,projectloader, testloader, classes, num_channels = get_dataloaders1(args,w_init)
    # Create a convolutional network based on arguments and add 1x1 conv layer
    features_net, add_on_layers = get_network(num_channels, args)
    # Create a ProtoTree
    tree = ProtoTree(num_classes=len(classes),
                     feature_net=features_net,
                     args=args,
                     add_on_layers=add_on_layers)

    # Check for NaN values in the data after loading
    for xs, ys in trainloader:
        #
        features_out = features_net(xs)
        print(f"Feature output shape (after feature_net): {features_out.shape}")

        check_nan(xs)
        break
    # Prototypes and features are embedded onto the Poincare ball inside the
    # projection layer via the exponential map, so no parameter conversion is
    # needed here. The tree itself is trained with a Riemannian optimizer.
    tree = tree.to(device=device)
    # creat a ProtoForest
    # num_tree = 3
    # trees=[]................................................................................
    # for _ in range(num_tree):
    #     tree = ProtoTree(num_classes=len(classes),
    #                      feature_net=features_net,
    #                      args=args,
    #                      add_on_layers=add_on_layers)
    #     tree = tree.to(device =device)
    #     trees.append(tree)
    # Determine which optimizer should be used to update the tree parameters
    optimizer, params_to_freeze, params_to_train = get_optimizer(tree, args)
    scheduler = get_scheduler(optimizer, args)
    tree, epoch = init_tree(tree, optimizer, scheduler, device, args)

    tree.save(f'{log.checkpoint_dir}/tree_init')
    log.log_message(
        "Max depth %s, so %s internal nodes and %s leaves" % (args.depth, tree.num_branches, tree.num_leaves))
    analyse_output_shape(tree, trainloader1, log, device)

    leaf_labels = dict()
    best_train_acc = 0.
    best_test_acc = 0.
    best_train_epoch = 0.
    w_best = []
    theta = 0
    # w = []
    if epoch < args.epochs + 1:
        '''
            TRAIN AND EVALUATE TREE
        '''
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epoch, args.epochs + 1):
            log.log_message("\nEpoch %s" % str(epoch))
            # Freeze (part of) network for some epochs if indicated in args
            freeze(tree, epoch, params_to_freeze, params_to_train, args, log)
            log_learning_rates(optimizer, args, log)

            # Train tree
            if tree._kontschieder_train:
                train_info = train_epoch_kontschieder(tree, trainloader1, optimizer, epoch,
                                                      args.disable_derivative_free_leaf_optim, device, log, log_prefix)
            else:
                train_info = train_epoch(tree, trainloader1, optimizer, epoch, w_init, args.disable_derivative_free_leaf_optim,
                                         device, log, log_prefix, cluster_coef=args.cluster_coef)
            save_tree(tree, optimizer, scheduler, epoch, log, args)
            best_train_acc, best_train_epoch, w_best,theta_best= save_best_train_tree1(tree, optimizer, scheduler, train_info['w'],train_info['theta'],best_train_acc,
                                                                             epoch,
                                                                             train_info['train_accuracy'], log)
            leaf_labels = analyse_leafs(tree, epoch, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
            print("w_best[0]:",w_best[0])

            # Evaluate tree
            if args.epochs > 200:
                if epoch % 10 == 0 or epoch == args.epochs:
                    eval_info = eval(tree, testloader, epoch, device, log)
                    original_test_acc = eval_info['test_accuracy']
                    best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc,
                                                        eval_info['test_accuracy'], log)
                    log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'],
                                   train_info['train_accuracy'], train_info['loss'])
                else:
                    log.log_values('log_epoch_overview', epoch, "n.a.", train_info['train_accuracy'],
                                   train_info['loss'])
            else:
                eval_info = eval(tree, testloader, epoch, device, log)
                original_test_acc = eval_info['test_accuracy']
                best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc,
                                                    eval_info['test_accuracy'], log)
                log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'],
                               train_info['loss'])

            scheduler.step(train_info['train_accuracy'])

    else:  # tree was loaded and not trained, so evaluate only
        '''
            EVALUATE TREE
        '''
        eval_info = eval(tree, testloader, epoch, device, log)
        original_test_acc = eval_info['test_accuracy']
        best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
        log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], "n.a.", "n.a.")

    '''
        EVALUATE AND ANALYSE TRAINED TREE
    '''
    log.log_message(
        "Training Finished. Best training accuracy was %s, best training epoch was %s\n,best test accuracy was %s\n" % (
        str(best_train_acc), int(best_train_epoch), str(best_test_acc)))
    trained_tree = deepcopy(tree)
    leaf_labels = analyse_leafs(tree, epoch + 1, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)

    '''
        PRUNE
    '''
    pruned = prune(tree, args.pruning_threshold_leaves, log)
    name = "pruned"
    save_tree_description(tree, optimizer, scheduler, name, log)
    pruned_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree
    leaf_labels = analyse_leafs(tree, epoch + 2, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    pruned_test_acc = eval_info['test_accuracy']

    pruned_tree = tree

    '''
        PROJECT
    '''
    project_info, tree = project_with_class_constraints(deepcopy(pruned_tree), projectloader, device, args, log)
    name = "pruned_and_projected"
    save_tree_description(tree, optimizer, scheduler, name, log)
    pruned_projected_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree with projected prototypes
    average_distance_nearest_image(project_info, tree, log)
    leaf_labels = analyse_leafs(tree, epoch + 3, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    # y_true = eval_info['y_true']
    # y_pred = eval_info['y_pred']
    pruned_projected_test_acc = eval_info['test_accuracy']
    # the_best_test_acc = eval_info['best_test_acc']
    eval_info_samplemax = eval(tree, testloader, name, device, log, 'sample_max')
    get_avg_path_length(tree, eval_info_samplemax, log)
    eval_info_greedy = eval(tree, testloader, name, device, log, 'greedy')
    get_avg_path_length(tree, eval_info_greedy, log)
    fidelity_info = eval_fidelity(tree, testloader, device, log)

    # Upsample prototype for visualization
    project_info = upsample(tree, project_info, projectloader, name, args, log)
    # visualize tree
    gen_vis(tree, name, args, classes)

    return trained_tree.to('cpu'), w_best, theta_best,pruned_tree.to('cpu'),pruned_projected_tree.to(
        'cpu'), original_test_acc, pruned_test_acc, pruned_projected_test_acc, project_info, eval_info_samplemax, eval_info_greedy, fidelity_info

def run_ensemble():
    all_args = get_args()
    set_seed(all_args.seed)
    # Create a logger
    log = Log(all_args.log_dir)
    print("Log dir: ", all_args.log_dir, flush=True)
    # Log the run arguments
    save_args(all_args, log.metadata_dir)

    if not all_args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not os.path.isdir(os.path.join(all_args.log_dir, "files")):
        os.mkdir(os.path.join(all_args.log_dir, "files"))

        # Obtain the data loaders
    # trainloader, projectloader, test_loader, classes, num_channels = get_dataloaders(all_args)
    trainset, projectset, testset, classes, shape = get_data(all_args)

    n_train = len(trainset)
    # Initialize data weights
    # w_init = np.ones(n_train) / n_train
    # w = format((np.ones(n_train) / n_train), '.4f')
    # print('w[2]-before:', w_init[2])
    # Initialize model weights
    # theta = np.zeros(n_train)
    log_dir_orig = all_args.log_dir

    trained_orig_trees = []
    trained_pruned_trees = []
    trained_pruned_projected_trees = []
    orig_test_accuracies = []
    pruned_test_accuracies = []
    pruned_projected_test_accuracies = []
    project_infos = []
    infos_sample_max = []
    infos_greedy = []
    infos_fidelity = []
    theta_final=[]
    the_best_test_acc=[]
    # Train trees in ensemble one by one and save corresponding trees and accuracies
    # Train trees in the ensemble one by one and store each tree and its accuracy
    for pt in range(1, all_args.nr_trees_ensemble + 1):
        torch.cuda.empty_cache()

        print("\nTraining tree ", pt, "/", all_args.nr_trees_ensemble, flush=True)
        log.log_message('Training tree %s...' % str(pt))

        args = deepcopy(all_args)
        args.log_dir = os.path.join(log_dir_orig, 'tree_' + str(pt))
        if pt == 1:
            w_init = np.ones(n_train) / n_train
            trained_tree, w_best, theta_best, pruned_tree, pruned_projected_tree, original_test_acc, pruned_test_acc, pruned_projected_test_acc, project_info, eval_info_samplemax, eval_info_greedy, info_fidelity = run_tree(
                w_init, args)
        else:
            w_init = w_best
            trained_tree, w_best, theta_best, pruned_tree, pruned_projected_tree, original_test_acc, pruned_test_acc, pruned_projected_test_acc, project_info, eval_info_samplemax, eval_info_greedy, info_fidelity = run_tree1(
                w_init, args)
        # trainloader, trainloader1, projectloader, test_loader, classes, num_channels = get_dataloaders(all_args)

        print(w_init)

        # w = w_after
        print("w_best[0]:",w_best[0])
        print("theta_best:",theta_best)
        # exit()
        theta_final.append(theta_best)

        trained_orig_trees.append(trained_tree)
        trained_pruned_trees.append(pruned_tree)
        # print("trained_pruned_trees:",trained_pruned_trees)
        trained_pruned_projected_trees.append(pruned_projected_tree)

        orig_test_accuracies.append(original_test_acc)

        pruned_test_accuracies.append(pruned_test_acc)

        pruned_projected_test_accuracies.append(pruned_projected_test_acc)
        # the_best_test_acc.append(the_best_test_acc)
        project_infos.append(project_info)
        infos_sample_max.append(eval_info_samplemax)
        infos_greedy.append(eval_info_greedy)
        infos_fidelity.append(info_fidelity)
    sum = 0
    for i in range(all_args.nr_trees_ensemble):
        sum += theta_final[i]
    for p in range(all_args.nr_trees_ensemble):
        theta_final[p] /= sum
    print('theta_final:', theta_final)
    print('pruned_test_accuracies:', pruned_test_accuracies)
    print('pruned_projected_test_accuracies:', pruned_projected_test_accuracies)

    # Weighted soft voting over the class probabilities of all base
    # learners. The normalized learner coefficients are used only at inference.
    testloader = torch.utils.data.DataLoader(testset, batch_size=all_args.batch_size,
                                             shuffle=False, pin_memory=False)
    ensemble_info = eval_ensemble(trained_pruned_projected_trees, testloader, device, log,
                                  weights=theta_final, sampling_strategy='distributed',
                                  progress_prefix='HABPT Ensemble')
    print('HABPT ensemble test accuracy:', ensemble_info['test_accuracy'])
    #     if pt > 4:
    #         # analyse ensemble with > 1 trees:
    #         analyse_ensemble(log, all_args, test_loader, device, trained_orig_trees, trained_pruned_trees,
    #                          trained_pruned_projected_trees, orig_test_accuracies, pruned_test_accuracies,
    #                          pruned_projected_test_accuracies, project_infos, infos_sample_max, infos_greedy,
    #                          infos_fidelity)


if __name__ == '__main__':
    run_ensemble()
