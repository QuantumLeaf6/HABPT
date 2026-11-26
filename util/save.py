import torch
import os
import argparse
from prototree.prototree import ProtoTree
from util.log import Log

def save_tree(tree: ProtoTree, optimizer, scheduler, epoch: int, log: Log, args: argparse.Namespace):
    tree.eval()
    # Save latest model
    tree.save(f'{log.checkpoint_dir}/latest')
    tree.save_state(f'{log.checkpoint_dir}/latest')
    torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/latest/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/latest/scheduler_state.pth')

    # Save model every 10 epochs
    if epoch == args.epochs or epoch%10==0:
        tree.save(f'{log.checkpoint_dir}/epoch_{epoch}')
        tree.save_state(f'{log.checkpoint_dir}/epoch_{epoch}')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/epoch_{epoch}/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/epoch_{epoch}/scheduler_state.pth')

def save_best_train_tree1(tree: ProtoTree, optimizer, scheduler, w_best:list,theta_best: float,best_train_acc: float, best_train_epoch: int,train_acc: float, log: Log):
    tree.eval()
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        best_train_epoch = best_train_epoch
        w_best = w_best
        theta_best = theta_best
        tree.save(f'{log.checkpoint_dir}/best_train_model')
        tree.save_state(f'{log.checkpoint_dir}/best_train_model')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/best_train_model/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/best_train_model/scheduler_state.pth')
    return best_train_acc,best_train_epoch,w_best,theta_best




def save_best_train_tree(tree: ProtoTree, optimizer, scheduler, best_train_acc: float, train_acc: float, log: Log):
    tree.eval()
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        tree.save(f'{log.checkpoint_dir}/best_train_model')
        tree.save_state(f'{log.checkpoint_dir}/best_train_model')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/best_train_model/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/best_train_model/scheduler_state.pth')
    return best_train_acc

def save_best_test_tree(tree: ProtoTree, optimizer, scheduler, best_test_acc: float, test_acc: float, log: Log):
    tree.eval()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        tree.save(f'{log.checkpoint_dir}/best_test_model')
        tree.save_state(f'{log.checkpoint_dir}/best_test_model')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/best_test_model/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/best_test_model/scheduler_state.pth')
    return best_test_acc

def save_tree_description(tree: ProtoTree, optimizer, scheduler, description: str, log: Log):
    tree.eval()
    # Save model with description
    tree.save(f'{log.checkpoint_dir}/'+description)
    tree.save_state(f'{log.checkpoint_dir}/'+description)
    torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/'+description+'/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/'+description+'/scheduler_state.pth')

####GRASSMAN
import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

