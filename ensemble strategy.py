import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import OrderedDict
import argparse
import torch
from torchvision import transforms, models, datasets
# import model.resnet_cbam as resnet_cbam
import os
from torch.autograd import Variable
from prototree.prototree import ProtoTree
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import cm
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
def load_state_dict(model_dir, is_multi_gpu):# model loading
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']

    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap = plt.get_cmap('Blues')):
        '''
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Input
        - cm : computed confusion matrix values
        - classes : the column corresponding to each row and column in the confusion matrix
        - normalize : True: show percentages, False: show counts
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
        # x- and y-axis lengths are the same (solution to problem 1)
        plt.axis("equal")
        # Handle the x-axis: if there is blank space on either side of the x-axis or y-axis (solution to problem 2)
        ax = plt.gca()  # get the current axis
        left, right = plt.xlim()  # get the maximum and minimum values of the x-axis
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
        plt.show()
def mainboosting(args):
    # gpus = args.gpu.split(',')
    data_transforms = {# data augmentation
        'test': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                # transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, shear=(-2, 2), translate=[0.05, 0.05]),
            ]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    #tensorboard --logdir './runs/Nov13_07-58-39_KOTIN-ASUS365/events.out.tfevents.1605225519' --port 8890
    test_datasets = datasets.ImageFolder(os.path.join(args.data_root, 'test-800'), data_transforms['test'])
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=1,shuffle=False, num_workers=2)
    print("test_dataloaders:", len(test_dataloaders.dataset))
    # print("test_dataloaders:", test_dataloaders)
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    total = 0
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
    conf_matrix = torch.zeros(8,8)
    attack_types = ["ADI", "DEB", "LYM", "MUC", "MUS", "NORM","STR", "TUM"]
    # attack_types = ["Benign", "Complex Stroma", "Debris", "Inflammatory", "Muscle", "Stroma", "Tumor"]
    # attack_types = ["Cancer","Normal"]
    # my_model = resnet_cbam.resnet50_cbam(pretrained=False,num_classes=2).cuda()
    # for i in range (0,4):
    my_model1 = ProtoTree.load(args.prototree1).to(device=device)
    my_model2 = ProtoTree.load(args.prototree2).to(device=device)
    my_model3 = ProtoTree.load(args.prototree3).to(device=device)
    my_model4 = ProtoTree.load(args.prototree4).to(device=device)
    my_model5 = ProtoTree.load(args.prototree5).to(device=device)
    my_model1.eval()
    my_model2.eval()
    my_model3.eval()
    my_model4.eval()
    my_model5.eval()
    sampling_strategy='distributed'
    # a= [0.2177, 0.2072, 0.1987,0.1929,0.1836]# classifier weights
    # a = [0.2227, 0.2205, 0.2112, 0.1584, 0.1871]
    a=[0.2126,0.2124,0.2053, 0.1910, 0.1787]
    img_path_list=[]
    true_category_list=[]
    ys_pred_final_list=[]
    predict_category_1_list=[]
    predict_category_2_list=[]
    predict_category_3_list = []
    predict_category_4_list = []
    predict_category_5_list = []
    predict_category_final_list=[]
    for img_path_ADI in os.listdir(args.image_path_ADI):
        img_path_list.append(img_path_ADI)
    for img_path_DEB in os.listdir(args.image_path_DEB):
        img_path_list.append(img_path_DEB)
    for img_path_LYM in os.listdir(args.image_path_LYM):
        img_path_list.append(img_path_LYM)
    for img_path_MUC in os.listdir(args.image_path_MUC):
        img_path_list.append(img_path_MUC)
    for img_path_MUS in os.listdir(args.image_path_MUS):
        img_path_list.append(img_path_MUS)
    for img_path_NORM in os.listdir(args.image_path_NORM):
        img_path_list.append(img_path_NORM)
    for img_path_STR in os.listdir(args.image_path_STR):
        img_path_list.append(img_path_STR)
    for img_path_TUM in os.listdir(args.image_path_TUM):
        img_path_list.append(img_path_TUM)
    print(img_path_list)
    for xs, ys in  test_dataloaders:
        xs, ys = xs.to(device), ys.to(device)
        # print(xs,ys)
        true_category = ys.cuda()
        # print(true_category)
        # Use the model to classify this batch of input data
        out1, tes_info = my_model1.forward(xs, sampling_strategy)

        out2, test_info = my_model2.forward(xs, sampling_strategy)
        out3, test_info = my_model3.forward(xs, sampling_strategy)
        # print("out1",out1)
        out4, test_info = my_model4.forward(xs, sampling_strategy)
        out5, test_info = my_model5.forward(xs, sampling_strategy)
        print("out:",out1, out2, out3,out4,out5)
        # print("out:", out1, out2, out3)
        F = a[0]*out1+a[1]*out2+a[2]*out3+a[3]*out4+a[4]*out5
        # F = a[0] * out1 + a[1] * out2 + a[2] * out3
        print(F)
        true_category_list.append(true_category)
        ys_pred_1 = torch.argmax(out1, dim=1)
        print("ys_pred_1:",ys_pred_1)
        ys_pred_2 = torch.argmax(out2, dim=1)
        ys_pred_3 = torch.argmax(out3, dim=1)
        ys_pred_4 = torch.argmax(out4, dim=1)
        ys_pred_5 = torch.argmax(out5, dim=1)
        ys_pred_final = torch.argmax(F, dim=1)
        ys_pred_final_list.append(ys_pred_1)
        predict_category_1_list.append(ys_pred_1)
        predict_category_2_list.append(ys_pred_2)
        predict_category_3_list.append(ys_pred_3)
        predict_category_4_list.append(ys_pred_4)
        predict_category_5_list.append(ys_pred_5)
        predict_category_final_list.append(ys_pred_final)
        total += ys.size(0)

        correct1 += (ys_pred_1 == ys.cuda()).sum().cuda()
        correct2 += (ys_pred_2 == ys.cuda()).sum().cuda()
        correct3 += (ys_pred_3 == ys.cuda()).sum().cuda()
        correct4 += (ys_pred_4 == ys.cuda()).sum().cuda()
        correct5 += (ys_pred_5 == ys.cuda()).sum().cuda()
        correct += (ys_pred_final == ys.cuda()).sum().cuda()
        predicted = ys_pred_final
        conf_matrix = confusion_matrix(predicted, labels=ys, conf_matrix=conf_matrix)

    plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False,
                              title='Normalized confusion matrix')
    for i in range(6400):
        if true_category_list[i] != predict_category_1_list[i]:
            print("tree1_wrong:",true_category_list[i], predict_category_1_list[i], img_path_list[i])
            print(img_path_list[i])
    for i in range(6400):
        if true_category_list[i] != predict_category_2_list[i]:
            print("tree2_wrong:",true_category_list[i], predict_category_2_list[i], img_path_list[i])

    for i in range(6400):
        if true_category_list[i] != predict_category_3_list[i]:
            print("tree3_wrong:",true_category_list[i], predict_category_3_list[i], img_path_list[i])
    for i in range(6400):
        if true_category_list[i] != predict_category_4_list[i]:
            print("tree4_wrong:",true_category_list[i], predict_category_4_list[i], img_path_list[i])
    for i in range(6400):
        if true_category_list[i] != predict_category_5_list[i]:
            print("tree5_wrong:",true_category_list[i], predict_category_5_list[i], img_path_list[i])
    for i in range(6400):
        if true_category_list[i] != predict_category_final_list[i]:
            print("treefinal_wrong:",true_category_list[i], predict_category_final_list[i], img_path_list[i])
            print(img_path_list[i])
    print(total)
    print("correct1:", correct1)
    print("correct2:", correct2)
    print("correct3:", correct3)
    print("correct4:", correct4)
    print("correct5:", correct5)
    print("correct:",correct)
    print('Accuracy of the model1 on the test images:{0:0.4''f}'.format(100 * correct1 / total))
    print('Accuracy of the model2 on the test images:{0:0.4''f}'.format(100 * correct2 / total))
    print('Accuracy of the model3 on the test images:{0:0.4''f}'.format(100 * correct3 / total))
    print('Accuracy of the model4 on the test images:{0:0.4''f}'.format(100 * correct4 / total))
    print('Accuracy of the model5 on the test images:{0:0.4''f}'.format(100 * correct5 / total))
    print('Accuracy of the model on the test images:{0:0.4''f}'.format(100 * correct / total))
    # plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False, title=' Confusion Matrix')
    # print('Accuracy of the model on the test images: %.2f ' .format(100 * correct / total))
# def mainbagging(args):
#     # gpus = args.gpu.split(',')
#     data_transforms = {# data augmentation
#         'test': transforms.Compose([
#             # transforms.RandomResizedCrop(224),
#             # transforms.RandomHorizontalFlip(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
#
#     #tensorboard --logdir './runs/Nov13_07-58-39_KOTIN-ASUS365/events.out.tfevents.1605225519' --port 8890
#     test_datasets = datasets.ImageFolder(os.path.join(args.data_root, 'test-800'), data_transforms['test'])
#     test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=1,shuffle=False, num_workers=2)
#     print("test_dataloaders:", len(test_dataloaders.dataset))
#     # print("test_dataloaders:", test_dataloaders)
#     correct = 0
#     correct1 = 0
#     correct2 = 0
#     correct3 = 0
#     correct4 = 0
#     correct5 = 0
#     total = 0
#     if not args.disable_cuda and torch.cuda.is_available():
#         device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
#     else:
#         device = torch.device('cpu')
#     conf_matrix = torch.zeros(7,7)
#     # attack_types = ["ADI", "DEB", "LYM", "MUC", "MUS", "NORM","STR", "TUM"]
#     attack_types = ["Benign", "Complex Stroma", "Debris", "Inflammatory", "Muscle", "Stroma", "Tumor"]
#     # attack_types = ["Cancer","Normal"]
#     # my_model = resnet_cbam.resnet50_cbam(pretrained=False,num_classes=2).cuda()
#     # for i in range (0,4):
#     my_model1 = ProtoTree.load(args.prototree1).to(device=device)
#     my_model2 = ProtoTree.load(args.prototree2).to(device=device)
#     my_model3 = ProtoTree.load(args.prototree3).to(device=device)
#     my_model4 = ProtoTree.load(args.prototree4).to(device=device)
#     my_model5 = ProtoTree.load(args.prototree5).to(device=device)
#     my_model1.eval()
#     my_model2.eval()
#     my_model3.eval()
#     # my_model4.eval()
#     # my_model5.eval()
#     sampling_strategy='distributed'
#     # a= [0.2177, 0.2072, 0.1987,0.1929,0.1836]# classifier weights
#     # a = [0.2227, 0.2205, 0.2112, 0.1584, 0.1871]
#     a=[0.3419,0.336,0.3221]
#     img_path_list=[]
#     true_category_list=[]
#     ys_pred_final_list=[]
#     predict_category_1_list=[]
#     predict_category_2_list=[]
#     predict_category_3_list = []
#     predict_category_4_list = []
#     predict_category_5_list = []
#     predict_category_final_list=[]
#     for img_path_Benign in os.listdir(args.image_path_Benign):
#         img_path_list.append(img_path_Benign)
#     for img_path_ComplexStroma in os.listdir(args.image_path_ComplexStroma):
#         img_path_list.append(img_path_ComplexStroma)
#     for img_path_Debris in os.listdir(args.image_path_Debris):
#         img_path_list.append(img_path_Debris)
#     for img_path_Inflammatory in os.listdir(args.image_path_Inflammatory):
#         img_path_list.append(img_path_Inflammatory)
#     for img_path_Muscle in os.listdir(args.image_path_Muscle):
#         img_path_list.append(img_path_Muscle)
#     for img_path_Stroma in os.listdir(args.image_path_Stroma):
#         img_path_list.append(img_path_Stroma)
#     for img_path_Tumor in os.listdir(args.image_path_Tumor):
#         img_path_list.append(img_path_Tumor)
#     # for img_path_TUM in os.listdir(args.image_path_TUM):
#     #     img_path_list.append(img_path_TUM)
#     # print(img_path_list)
#     for xs, ys in  test_dataloaders:
#         ys_pred_final_list = []
#         xs, ys = xs.to(device), ys.to(device)
#         # print(xs,ys)
#         true_category = ys.cuda()
#         # print(true_category)
#         # Use the model to classify this batch of input data
#         out1, tes_info = my_model1.forward(xs, sampling_strategy)
#
#         out2, test_info = my_model2.forward(xs, sampling_strategy)
#         out3, test_info = my_model3.forward(xs, sampling_strategy)
#         # print("out1",out1)
#         out4, test_info = my_model4.forward(xs, sampling_strategy)
#         out5, test_info = my_model5.forward(xs, sampling_strategy)
#         # print("out:",out1, out2, out3,out4,out5)
#         # print("out:", out1, out2, out3)
#         # F = a[0]*out1+a[1]*out2+a[2]*out3+a[3]*out4+a[4]*out5
#         # F = a[0] * out1 + a[1] * out2 + a[2] * out3
#         # print(F)
#         true_category_list.append(true_category)
#         ys_pred_1 = torch.argmax(out1, dim=1)
#         ys_pred_final_list.append(ys_pred_1)
#         ys_pred_2 = torch.argmax(out2, dim=1)
#         ys_pred_final_list.append(ys_pred_2)
#         ys_pred_3 = torch.argmax(out3, dim=1)
#         ys_pred_final_list.append(ys_pred_3)
#         ys_pred_4 = torch.argmax(out4, dim=1)
#         ys_pred_final_list.append(ys_pred_4)
#         ys_pred_5 = torch.argmax(out5, dim=1)
#         ys_pred_final_list.append(ys_pred_5)
#         # print("ys_pred_final_list",ys_pred_final_list)
#         # print("mode:",stats.mode(ys_pred_final_list)[0][0])
#         ys_pred_final = stats.mode(ys_pred_final_list)[0][0]
#         # ys_pred_final_list.append(ys_pred_1)
#         # predict_category_1_list.append(ys_pred_1)
#         # predict_category_2_list.append(ys_pred_2)
#         # predict_category_3_list.append(ys_pred_3)
#         # predict_category_4_list.append(ys_pred_4)
#         # predict_category_5_list.append(ys_pred_5)
#         predict_category_final_list.append(ys_pred_final)
#         total += ys.size(0)
#
#         correct1 += (ys_pred_1 == ys.cuda()).sum().cuda()
#         correct2 += (ys_pred_2 == ys.cuda()).sum().cuda()
#         correct3 += (ys_pred_3 == ys.cuda()).sum().cuda()
#         correct4 += (ys_pred_4 == ys.cuda()).sum().cuda()
#         correct5 += (ys_pred_5 == ys.cuda()).sum().cuda()
#         correct += (ys_pred_final == ys.cuda()).sum().cuda()
#         predicted = ys_pred_final
#         conf_matrix = confusion_matrix(predicted, labels=ys, conf_matrix=conf_matrix)
#
#     plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False,
#                               title='Normalized confusion matrix')
#     # for i in range(6400):
#     #     if true_category_list[i] != predict_category_1_list[i]:
#     #         # print("tree1_wrong:",true_category_list[i], predict_category_1_list[i], img_path_list[i])
#     #         print(img_path_list[i])
#     # for i in range(6400):
#     #     if true_category_list[i] != predict_category_2_list[i]:
#     #         print("tree2_wrong:",true_category_list[i], predict_category_2_list[i], img_path_list[i])
#     # for i in range(6400):
#     #     if true_category_list[i] != predict_category_3_list[i]:
#     #         print("tree3_wrong:",true_category_list[i], predict_category_3_list[i], img_path_list[i])
#     # for i in range(6400):
#     #     if true_category_list[i] != predict_category_4_list[i]:
#     #         print("tree4_wrong:",true_category_list[i], predict_category_4_list[i], img_path_list[i])
#     # for i in range(6400):
#     #     if true_category_list[i] != predict_category_5_list[i]:
#     #         print("tree5_wrong:",true_category_list[i], predict_category_5_list[i], img_path_list[i])
#     # for i in range(6400):
#     #     if true_category_list[i] != predict_category_final_list[i]:
#     #         # print("treefinal_wrong:",true_category_list[i], predict_category_final_list[i], img_path_list[i])
#     #         print(img_path_list[i])
#     print(total)
#     print("correct1:", correct1)
#     print("correct2:", correct2)
#     print("correct3:", correct3)
#     # print("correct4:", correct4)
#     # print("correct5:", correct5)
#     print("correct:",correct)
#     print('Accuracy of the model1 on the test images:{0:0.4''f}'.format(100 * correct1 / total))
#     print('Accuracy of the model2 on the test images:{0:0.4''f}'.format(100 * correct2 / total))
#     print('Accuracy of the model3 on the test images:{0:0.4''f}'.format(100 * correct3 / total))
#     print('Accuracy of the model4 on the test images:{0:0.4''f}'.format(100 * correct4 / total))
#     print('Accuracy of the model5 on the test images:{0:0.4''f}'.format(100 * correct5 / total))
#     print('Accuracy of the model on the test images:{0:0.4''f}'.format(100 * correct / total))
#     # plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False, title=' Confusion Matrix')
#     # print('Accuracy of the model on the test images: %.2f ' .format(100 * correct / total))
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug',default=True,
                        help='trainer debug flag')
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')
    parser.add_argument('--prototree1',
                        type=str,
                        default=r'.\pretrain_model\P\best_test_model1',)
    parser.add_argument('--prototree2',
                        type=str,
                        default=r'.\pretrain_model\P\best_test_model2', )
    parser.add_argument('--prototree3',
                        type=str,
                        default=r'.\pretrain_model\P\best_test_model3', )
    parser.add_argument('--prototree4',
                        type=str,
                        default=r'.\pretrain_model\P\best_test_model4', )
    parser.add_argument('--prototree5',
                        type=str,
                        default=r'.\pretrain_model\P\best_test_model5', )
    parser.add_argument('--image_path_ADI', type=str, default=r'data/dataset_8class\test-800\001.ADI',
                        help='Input image path')
    parser.add_argument('--image_path_DEB', type=str, default=r'data/dataset_8class\test-800\002.DEB',
                        help='Input image path')
    parser.add_argument('--image_path_LYM', type=str, default=r'data/dataset_8class\test-800\003.LYM',
                        help='Input image path')
    parser.add_argument('--image_path_MUC', type=str, default=r'data/dataset_8class\test-800\004.MUC',
                        help='Input image path')
    parser.add_argument('--image_path_MUS', type=str, default=r'data/dataset_8class\test-800\005.MUS',
                        help='Input image path')
    parser.add_argument('--image_path_NORM', type=str, default=r'data/dataset_8class\test-800\006.NORM',
                        help='Input image path')
    parser.add_argument('--image_path_STR', type=str, default=r'data/dataset_8class\test-800\007.STR',
                        help='Input image path')
    parser.add_argument('--image_path_TUM', type=str, default=r'data/dataset_8class\test-800\008.TUM',
                        help='Input image path')
    parser.add_argument('-d', '--data_root', default=r'data/dataset_8class',
                         type=str, help='data root')

    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    # parser.add_argument('-t', '--train_file', default='./datasets/train.txt',
    #                      type=str, help='train file')
    # parser.add_argument('-v', '--val_file', default='./datasets/val.txt',
    #                      type=str, help='validation file')
    # parser.add_argument('-m', '--model', default='resnet18_cbam',
    #                      type=str, help='model type')
    parser.add_argument('--batch_size', default=64,
                         type=int, help='model train batch size')
    parser.add_argument('--display', action='store_true', dest='display',default=True,
                        help='Use TensorboardX to Display')
    args = parser.parse_args()

    mainboosting(args)