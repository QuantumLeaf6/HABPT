import argparse
from tqdm import tqdm
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from prototree.prototree import ProtoTree
from util.log import Log
from scipy import stats

@torch.no_grad()
def eval(tree: ProtoTree,
        test_loader: DataLoader,
        epoch,
        device,
        log: Log = None,  
        sampling_strategy: str = 'distributed',
        log_prefix: str = 'log_eval_epochs', 
        progress_prefix: str = 'Eval Epoch'
        ) -> dict:
    tree = tree.to(device)

    # Keep an info dict about the procedure
    info = dict()
    if sampling_strategy != 'distributed':
        info['out_leaf_ix'] = []
    # Build a confusion matrix
    cm = np.zeros((tree._num_classes, tree._num_classes), dtype=int)

    # Make sure the model is in evaluation mode
    tree.eval()

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)
    #len(test_loader)=25
    # Iterate through the test set
    # epoch = 100
    for i, (xs, ys) in test_iter:
        # print(i)
        xs, ys = xs.to(device), ys.to(device) # xs.shape: torch.Size([64, 3, 224, 224]) ys.shape:torch.Size([64])
        # Use the model to classify this batch of input data
        out, test_info = tree.forward(xs, sampling_strategy)#out.shape: torch.Size([64, 8])
        ys_pred = torch.argmax(out, dim=1)#ys_pred: torch.Size([64])
        # Update the confusion matrix
        cm_batch = np.zeros((tree._num_classes, tree._num_classes), dtype=int)
        attack_types = ["ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

        for y_pred, y_true in zip(ys_pred, ys):
            # print("y_pred,y_true:",y_pred,y_true)
            cm[y_true][y_pred] += 1
            cm_batch[y_true][y_pred] += 1

        acc = acc_from_cm(cm_batch)
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}], Acc: {acc:.3f}'
        )

        # keep list of leaf indices where test sample ends up when deterministic routing is used.
        if sampling_strategy != 'distributed':
            info['out_leaf_ix'] += test_info['out_leaf_ix']
        del out
        del ys_pred
        del test_info
    # if epoch>=2:
    #     plot_confusion_matrix(cm_batch, classes=attack_types, normalize=False,
    #                       title='Normalized confusion matrix')
    #     info['y_pred'] = ys_pred
    #     info['y_true'] = ys
    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    log.log_message("\nEpoch %s - Test accuracy with %s routing: "%(epoch, sampling_strategy)+str(info['test_accuracy']))
    return info

@torch.no_grad()
def eval_fidelity(tree: ProtoTree,
        test_loader: DataLoader,
        device,
        log: Log = None,  
        progress_prefix: str = 'Fidelity'
        ) -> dict:
    tree = tree.to(device)

    # Keep an info dict about the procedure
    info = dict()

    # Make sure the model is in evaluation mode
    tree.eval()
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix,
                        ncols=0)

    distr_samplemax_fidelity = 0
    distr_greedy_fidelity = 0
    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data, with 3 types of routing
        out_distr, _ = tree.forward(xs, 'distributed')
        ys_pred_distr = torch.argmax(out_distr, dim=1)

        out_samplemax, _ = tree.forward(xs, 'sample_max')
        ys_pred_samplemax = torch.argmax(out_samplemax, dim=1)

        out_greedy, _ = tree.forward(xs, 'greedy')
        ys_pred_greedy = torch.argmax(out_greedy, dim=1)
        
        # Calculate fidelity
        distr_samplemax_fidelity += torch.sum(torch.eq(ys_pred_samplemax, ys_pred_distr)).item()
        distr_greedy_fidelity += torch.sum(torch.eq(ys_pred_greedy, ys_pred_distr)).item()
        # Update the progress bar
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}]'
        )
        del out_distr
        del out_samplemax
        del out_greedy

    distr_samplemax_fidelity = distr_samplemax_fidelity/float(len(test_loader.dataset))
    distr_greedy_fidelity = distr_greedy_fidelity/float(len(test_loader.dataset))
    info['distr_samplemax_fidelity'] = distr_samplemax_fidelity
    info['distr_greedy_fidelity'] = distr_greedy_fidelity
    log.log_message("Fidelity between standard distributed routing and sample_max routing: "+str(distr_samplemax_fidelity))
    log.log_message("Fidelity between standard distributed routing and greedy routing: "+str(distr_greedy_fidelity))
    return info

@torch.no_grad()
# def eval_ensemble(trees: list, test_loader: DataLoader, device, log: Log, args: argparse.Namespace, sampling_strategy: str = 'distributed', progress_prefix: str = 'Eval Ensemble'):
#     # Keep an info dict about the procedure
#     info = dict()
#     # Build a confusion matrix
#     cm = np.zeros((trees[0]._num_classes, trees[0]._num_classes), dtype=int)
#
#     # Show progress on progress bar
#     test_iter = tqdm(enumerate(test_loader),
#                         total=len(test_loader),
#                         desc=progress_prefix,
#                         ncols=0)
#
#     # Iterate through the test set
#     for i, (xs, ys) in test_iter:
#         xs, ys = xs.to(device), ys.to(device)
#         outs = []
#         zuhe = []
#         for tree in trees:
#             # Make sure the model is in evaluation mode
#             tree.eval()
#             tree = tree.to(device)
#             # Use the model to classify this batch of input data
#             out, _ = tree.forward(xs, sampling_strategy)
#             ys_pred_yasuo = torch.argmax(out,dim=1)
#             ys_pred_yasuo = ys_pred_yasuo.unsqueeze(0)
#             # Convert tensor to list
#             # ys_pred_yasuo = ys_pred_yasuo.cpu().numpy().tolist()
#             zuhe.append(ys_pred_yasuo)
#             outs.append(out)
#             del out
#         # print(zuhe)
#         concat = torch.cat((zuhe[0],zuhe[1],zuhe[2],zuhe[3],zuhe[4]),0)#5x64
#         # print('outs:',outs.shape())
#         print('concat:',concat.shape)
#         tiqu = []
#         ys_pred=[]
#         for j in range(64):
#             tiqu.append(concat[:,j])
#             ys_pred.append(stats.mode(tiqu[j].cpu())[0][0])
#         print(tiqu)
#             # tiqu.append(tiqu[j])
#             # ys_pred = stats.mode(tiqu[j])[0][0]
#         # stacked = torch.stack(outs, dim=0)#torch.Size([5, 64, 8])
#         # print(torch.mean(stacked, dim=0))
#         # ys_pred = torch.argmax(torch.mean(stacked, dim=0), dim=1)#torch.Size([64])
#         print('ys_pred:',len(ys_pred))
#         for y_pred, y_true in zip(ys_pred, ys):
#             cm[y_true][y_pred] += 1
#
#         test_iter.set_postfix_str(
#             f'Batch [{i + 1}/{len(test_iter)}]'
#         )
#         del outs
#
#     info['confusion_matrix'] = cm
#     info['test_accuracy'] = acc_from_cm(cm)
#     log.log_message("Ensemble accuracy with %s routing: %s"%(sampling_strategy, str(info['test_accuracy'])))
#     return info


@torch.no_grad()
def eval_ensemble(trees: list, test_loader: DataLoader, device, log: Log,
                  weights=None, sampling_strategy: str = 'distributed',
                  progress_prefix: str = 'Eval Ensemble'):
    """
    Evaluate the HABPT ensemble with weighted soft voting.

    Each base learner first turns its class distribution into a softmax
    probability vector; the ensemble then computes the weighted sum of the
    five probability vectors and assigns the class with the largest score.
    """
    info = dict()
    cm = np.zeros((trees[0]._num_classes, trees[0]._num_classes), dtype=int)

    if weights is None:
        weights = [1.0 / len(trees)] * len(trees)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()  # learner coefficients normalized across the trees

    test_iter = tqdm(enumerate(test_loader),
                     total=len(test_loader),
                     desc=progress_prefix,
                     ncols=0)

    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)
        outs = []
        for tree in trees:
            tree.eval()
            tree = tree.to(device)
            out, _ = tree.forward(xs, sampling_strategy)
            outs.append(torch.softmax(out, dim=1))  # softmax per base learner
            del out
        stacked = torch.stack(outs, dim=0)  # (num_trees, batch_size, num_classes)
        final_score = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
        ys_pred = torch.argmax(final_score, dim=1)

        for y_pred, y_true in zip(ys_pred, ys):
            cm[y_true.item()][y_pred.item()] += 1

        test_iter.set_postfix_str(f'Batch [{i + 1}/{len(test_iter)}]')
        del outs

    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    log.log_message('Ensemble accuracy with %s routing: %s'
                    % (sampling_strategy, str(info['test_accuracy'])))
    return info
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.get_cmap('Blues')):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : values of the computed confusion matrix
    - classes : classes corresponding to each row and column of the confusion matrix
    - normalize : True shows percentages, False shows counts
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    # Keep x and y axes the same length (fix for issue 1)
    plt.axis("equal")
    # Adjust the x axis if there is blank space on either side (fix for issue 2)
    ax = plt.gca()  # Get the current axis
    left, right = plt.xlim()  # Get the x-axis limits
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cm.png')
    plt.show()
def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total
