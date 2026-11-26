import os
import subprocess
import numpy as np
import copy
import argparse
from subprocess import check_call
from PIL import Image
import torch
import math
from prototree.prototree import ProtoTree
from prototree.branch import Branch
from prototree.leaf import Leaf
from prototree.node import Node

# Poincare ball model function
def poincare_ball_model(x,c=0.5):
    # Assuming x is a tensor with Poincare coordinates
    norm_sq = x.pow(2).sum(dim=-1, keepdim=True)
    return 2. / (1 - c* norm_sq) * x



def gen_vis(tree: ProtoTree, folder_name: str, args: argparse.Namespace, classes:tuple):
    destination_folder=os.path.join(args.log_dir,folder_name)
    upsample_dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), folder_name)
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.isdir(destination_folder + '/node_vis'):
        os.mkdir(destination_folder + '/node_vis')

    with torch.no_grad():
        s = 'digraph T {margin=0;ranksep=".03";nodesep="0.05";splines="curved";    ranksep=0.4;  nodesep=0.2;\n'#digraph T defines a directed graph; subsequent settings (such as margin, ranksep, nodesep, and splines) are used to adjust the graph display.
           #nodesep="0.5";  // set the horizontal spacing to 0.5;ranksep="0.5";  // set the vertical spacing to 0.5
        s += 'node [shape=rect, label="" ];\n'                #node [shape=rect, label=""]; set the node shape to a rectangle and the label to empty.
        s += _gen_dot_nodes(tree._root, destination_folder, upsample_dir, classes)# called to generate the node information of the tree and append it to string s.
        s += _gen_dot_edges(tree._root, classes)[0]                                 # called to generate the edge information of the tree and append it to string s.
        s += '}\n'

    with open(os.path.join(destination_folder,'treevis.dot'), 'w') as f:
        f.write(s)
   
    from_p = os.path.join(destination_folder,'treevis.dot')
    to_pdf = os.path.join(destination_folder,'treevis.pdf')
    check_call(['dot', '-Tpdf', '-Gmargin=0', from_p, '-o', to_pdf])
    to_pdf = os.path.join(destination_folder, 'treevis.tif')
    check_call(['dot', '-Ttif', '-Gmargin=0', from_p, '-o', to_pdf])
    #check_call('dot -Tpdf -Gmargin=0 %s -o %s'%(from_p, to_pdf), shell=True)

# Modify the _node_vis function
def _node_vis(node: Node, upsample_dir: str, visual_coordinates=None):
    # Convert the coordinates to the Poincare ball model for visualization
    # If visual_coordinates is not provided, convert the node coordinates
    if visual_coordinates is None and hasattr(node, 'coordinates'):
        node_coordinates = torch.tensor(node.coordinates, dtype=torch.float)
        # euclidean_coordinates = poincare_ball_model(node_coordinates).numpy()

    # # visualize using Euclidean coordinates
    # visual_coordinates = euclidean_coordinates

    # Use Visual_coordinates to call the original _node_vis function instead of modifying node.coordinates
    if isinstance(node, Leaf):
        return _leaf_vis(node, visual_coordinates)
    if isinstance(node, Branch):
        return _branch_vis(node, visual_coordinates, upsample_dir)


def _leaf_vis(node: Leaf, visual_coordinates):
    if node._log_probabilities:
        ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
    else:
        ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
    
    ws = np.ones(ws.shape) - ws
    ws *= 255

    height = 24

    if ws.shape[0] < 36:
        img_size = 36
    else:
        img_size = ws.shape[0]
    scaler = math.ceil(img_size/ws.shape[0])

    img = Image.new('F', (ws.shape[0]*scaler, height))
    pixels = img.load()

    for i in range(scaler*ws.shape[0]):
        for j in range(height-10):
            pixels[i,j]=ws[int(i/scaler)]
        for j in range(height-10,height-9):
            pixels[i,j]=0 #set bottom line of leaf distribution black
        for j in range(height-9,height):
            pixels[i,j]=255 #set bottom part of node white such that class label is readable

    if scaler*ws.shape[0]>100:
        img=img.resize((100,height))
    return img



# add three groups of images
def _branch_vis(node: Branch, visual_coordinates, upsample_dir: str):
    branch_id = node.index

    # Base image set (left-right concatenation)
    images = [
        Image.open(os.path.join(upsample_dir, f'{branch_id}_nearest_patch_of_image_0.png')),
        Image.open(os.path.join(upsample_dir, f'{branch_id}_bounding_box_nearest_patch_of_image_0.png')),
        Image.open(os.path.join(upsample_dir, f'{branch_id}_heatmap_original_image_0.png'))
    ]

    # Specific nodes need to load additional images
    extra_images = []
    extra_images1 = []
    if branch_id in [2, 5, 9, 12]:
        # First group of extra images
        for i in range(1, 2):  # only load one group of extra images
            try:
                extra_images.append(Image.open(os.path.join(upsample_dir, f'{branch_id}_nearest_patch_of_image_{i}.png')))
                extra_images.append(Image.open(os.path.join(upsample_dir, f'{branch_id}_bounding_box_nearest_patch_of_image_{i}.png')))
                extra_images.append(Image.open(os.path.join(upsample_dir, f'{branch_id}_heatmap_original_image_{i}.png')))
            except FileNotFoundError:
                print(f"Warning: One or more extra images for branch {branch_id} not found.")

        # Second group of extra images
        for i in range(2, 3):  # only load the second group of extra images
            try:
                extra_images1.append(Image.open(os.path.join(upsample_dir, f'{branch_id}_nearest_patch_of_image_{i}.png')))
                extra_images1.append(Image.open(os.path.join(upsample_dir, f'{branch_id}_bounding_box_nearest_patch_of_image_{i}.png')))
                extra_images1.append(Image.open(os.path.join(upsample_dir, f'{branch_id}_heatmap_original_image_{i}.png')))
            except FileNotFoundError:
                print(f"Warning: One or more extra images for branch {branch_id} not found.")

    # Resize and merge the base images (left-right concatenation)
    resized_images = []
    for im in images:
        w, h = im.size
        if w < 100 and h < 100:
            resized_images.append(im)
        else:
            cs = 100 / w, 100 / h
            min_cs = min(cs)
            im = im.resize(size=(int(min_cs * w), int(min_cs * h)))
            resized_images.append(im)

    # Calculate the total width and maximum height (base images)
    total_width = sum(im.size[0] for im in resized_images)
    max_height = max(im.size[1] for im in resized_images)

    # Create a new canvas for concatenating the base images (left-right)
    new_image = Image.new('RGB', (total_width+50, max_height), color=(255, 255, 255))

    # Concatenate the base images together
    x_offset = 0
    for im in resized_images:
        new_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]+(100-im.size[0])+1

    # If there are extra images, concatenate them vertically
    if extra_images:
        # Resize and merge the extra images (left-right concatenation)
        resized_extra_images = []
        for im in extra_images:
            w, h = im.size
            if w < 100 and h < 100:
                resized_extra_images.append(im)
            else:
                cs = 100 / w, 100 / h
                min_cs = min(cs)
                im = im.resize(size=(int(min_cs * w), int(min_cs * h)))
                resized_extra_images.append(im)

        # Calculate the total width and height of the extra images
        total_extra_width = sum(im.size[0] for im in resized_extra_images)
        total_extra_height = max(im.size[1] for im in resized_extra_images)

        # Create a new canvas for concatenating the extra images (left-right)
        extra_image_combined = Image.new('RGB', (total_extra_width+40, total_extra_height), color=(255, 255, 255))

    # If there are extra images 1, concatenate them vertically
    if extra_images1:
        # Resize and merge the extra images (vertical concatenation)
        resized_extra_images1 = []
        for im in extra_images1:
            w, h = im.size
            if w < 100 and h < 100:

                resized_extra_images1.append(im)
            else:
                cs = 100 / w, 100 / h
                min_cs = min(cs)
                im = im.resize(size=(int(min_cs * w), int(min_cs * h)))
                resized_extra_images1.append(im)

        # Calculate the total width and height of the extra images
        total_extra_width = sum(im.size[0] for im in resized_extra_images1)
        total_extra_height = max(im.size[1] for im in resized_extra_images1)

        # Create a new canvas for concatenating the extra images (vertical)
        extra_image_combined1 = Image.new('RGB', (total_extra_width+40, total_extra_height), color=(255, 255, 255))

        # for index, im in enumerate(resized_extra_images):
        #     # Scale each image to a fixed size
        #     x_offset1 = index * 100  # calculate the horizontal offset of each image
        #     extra_image_combined.paste(im, (x_offset1, 0))  # paste the image into the new image
        # # concatenate the extra images vertically
        x_offset1 = 0
        for im in resized_extra_images:
            extra_image_combined.paste(im, (x_offset1, 0))
            # x_offset1 += im.size[0]+(100-im.size[0])+1
            x_offset1 +=100 + 1

        # concatenate extra images 1 vertically
        x_offset2 = 0
        for im in resized_extra_images1:
            extra_image_combined1.paste(im, (x_offset2, 0))
            x_offset2 += im.size[0]+(100-im.size[0])+1

        # Merge the additionally concatenated images below the base image
        total_combined_height = new_image.height + extra_image_combined.height+extra_image_combined1.height+4
        combined_image = Image.new('RGB', (300, total_combined_height), color=(255, 255, 255))
        combined_image.paste(new_image, (0, 0))  # base image on top
        combined_image.paste(extra_image_combined, (0, new_image.height+2))  # extra images at the bottom
        combined_image.paste(extra_image_combined1, (0, (2*new_image.height)+4))  # extra images at the bottom

    else:
        combined_image = new_image

    return combined_image






def _gen_dot_nodes(node: Node, destination_folder: str, upsample_dir: str, classes:tuple, visual_coordinates=None):
    if hasattr(node, 'coordinates'):
        node_coordinates = torch.tensor(node.coordinates)
        visual_coordinates = poincare_ball_model(node_coordinates).numpy()

    img = _node_vis(node, upsample_dir, visual_coordinates).convert('RGB')
    if isinstance(node, Leaf):
        if node._log_probabilities:
            ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
        else:
            ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = classes[t]
        str_targets = ','.join(str(t) for t in class_targets) if len(class_targets) > 0 else ""
        str_targets = str_targets.replace('_', ' ')
    filename = '{}/node_vis/node_{}_vis.png'.format(destination_folder, node.index)
    img.save(filename)
    if isinstance(node, Leaf):
        s = '{}[imagepos="tc" imagescale=height image="{}" label="{}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'.format(node.index, filename, str_targets)
    else:
        s = '{}[image="{}" xlabel="{}" fontsize=6 labelfontcolor=gray50 fontname=Helvetica];\n'.format(node.index, filename, node.index)
    if isinstance(node, Branch):
        return s\
               + _gen_dot_nodes(node.l, destination_folder, upsample_dir, classes)\
               + _gen_dot_nodes(node.r, destination_folder, upsample_dir, classes)
    if isinstance(node, Leaf):
        return s


def _gen_dot_edges(node: Node, classes:tuple):
    if isinstance(node, Branch):
        edge_l, targets_l = _gen_dot_edges(node.l, classes)
        edge_r, targets_r = _gen_dot_edges(node.r, classes)
        str_targets_l = ','.join(str(t) for t in targets_l) if len(targets_l) > 0 else ""
        str_targets_r = ','.join(str(t) for t in targets_r) if len(targets_r) > 0 else ""
        s = '{} -> {} [label="Absent" fontsize=10 tailport="s" headport="n" fontname=Helvetica color="red" penwidth=2 style="dashed"];\n ' \
            '{} -> {} [label="Present" fontsize=10 tailport="s" headport="n" fontname=Helvetica  color="green" penwidth=2 style="solid"];\n'\
            .format(node.index, node.l.index, node.index, node.r.index)
        return s + edge_l + edge_r, sorted(list(set(targets_l + targets_r)))
    if isinstance(node, Leaf):
        if node._log_probabilities:
            ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
        else:
            ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = classes[t]
        return '', class_targets

