import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from prototree.prototree import ProtoTree
from util.log import Log
import geoopt

def project(tree: ProtoTree,
            project_loader: DataLoader,
            device,
            args: argparse.Namespace,
            log: Log,  
            log_prefix: str = 'log_projection',  # TODO
            progress_prefix: str = 'Projection'
            ) -> dict:
        
    log.log_message("\nProject prototypes to the nearest training patches (without class constraints)...")
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()
    # instantiate the PoincareBall manifold from geoopt
    #manifold = geoopt.PoincareBall(c=1.0)
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes: (D, W1, H1)
    D, W1, H1 = tree.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                            total=len(project_loader),
                            desc=progress_prefix,
                            ncols=0
                            )

    
    with torch.no_grad():
        # Get a batch of data
        xs, ys = next(iter(project_loader))
        batch_size = xs.shape[0]
        for i, (xs, ys) in projection_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features_batch, distances_batch, out_map = tree.forward_partial(xs)
            #distances_batch = manifold.dist2plane(tree.prototype_layer.prototype_vectors, c=1.0).sum(dim=1)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():

                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):

                    # Find the index of the latent patch that is closest to the prototype
                    min_distance = distances.min()
                    min_distance_ix = distances.argmin()
                    # Use the index to get the closest latent patch
                    closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

                    # Check if the latent patch is closest for all data samples seen so far
                    if min_distance < global_min_proto_dist[j]:
                        global_min_proto_dist[j] = min_distance
                        global_min_patches[j] = closest_patch
                        global_min_info[j] = {
                            'input_image_ix': i * batch_size + batch_i,
                            'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
                            'W': W,
                            'H': H,
                            'W1': W1,
                            'H1': H1,
                            'distance': min_distance.item(),
                            'nearest_input': torch.unsqueeze(xs[batch_i],0),
                            'node_ix': node.index,
                        }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

            del features_batch
            del distances_batch
            del out_map
        # Copy the patches to the prototype layer weights
        projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                                dim=0,
                                out=tree.prototype_layer.prototype_vectors)
        del projection

    return global_min_info, tree
#
# def project_with_class_constraints(tree: ProtoTree,
#                                     project_loader: DataLoader,
#                                     device,
#                                     args: argparse.Namespace,
#                                     log: Log,
#                                     log_prefix: str = 'log_projection_with_constraints',  # TODO
#                                     progress_prefix: str = 'Projection'
#                                     ) -> dict:
#
#     log.log_message("\nProject prototypes to the nearest training patches (with class constraints)...")
#     # Set the model to evaluation mode
#     tree.eval()
#     torch.cuda.empty_cache()
#     # instantiate the PoincareBall manifold from geoopt
#     manifold = geoopt.PoincareBall()
#     # The goal is to find the latent patch that minimizes the L2 distance to each prototype
#     # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
#     # Also store info about the image that was used for projection
#     global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
#     global_min_patches = {j: None for j in range(tree.num_prototypes)}
#     global_min_info = {j: None for j in range(tree.num_prototypes)}
#
#     # Get the shape of the prototypes
#     D,W1,H1 = tree.prototype_shape
#
#     # Build a progress bar for showing the status
#     projection_iter = tqdm(enumerate(project_loader),
#                             total=len(project_loader),
#                             desc=progress_prefix,
#                             ncols=0
#                             )
#
#     with torch.no_grad():
#         # Get a batch of data
#         xs, ys = next(iter(project_loader))
#         batch_size = xs.shape[0]
#         # For each internal node, collect the leaf labels in the subtree with this node as root.
#         # Only images from these classes can be used for projection.
#         # "For each internal node, collect the leaf labels in the subtree with this node as root. Only images from these classes can be used for projection."
#         leaf_labels_subtree = dict()
#
#         for branch, j in tree._out_map.items():
#             leaf_labels_subtree[branch.index] = set()
#             for leaf in branch.leaves:
#                 leaf_labels_subtree[branch.index].add(torch.argmax(leaf.distribution()).item())
#
#         for i, (xs, ys) in projection_iter:
#             xs, ys = xs.to(device), ys.to(device)
#             # Get the features and distances
#             # - features_batch: features tensor (shared by all prototypes)
#             #   shape: (batch_size, D, W, H)
#             # - distances_batch: distances tensor (for all prototypes)
#             #   shape: (batch_size, num_prototypes, W, H)
#             # - out_map: a dict mapping decision nodes to distances (indices)
#             features_batch, distances_batch, out_map = tree.forward_partial(xs)
#             #distances_batch = manifold.dist2plane(tree.prototype_layer.prototype_vectors, c=1.0).sum(dim=1)
#             # Get the features dimensions
#             bs, D, W, H = features_batch.shape
#
#             # Get a tensor containing the individual latent patches
#             # Create the patches by unfolding over both the W and H dimensions
#             # TODO -- support for strides in the prototype layer? (corresponds to step size here)
#             patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)
#
#             # Iterate over all decision nodes/prototypes
#             for node, j in out_map.items():
#                 leaf_labels = leaf_labels_subtree[node.index]
#                 # Iterate over all items in the batch
#                 # Select the features/distances that are relevant to this prototype
#                 # - distances: distances of the prototype to the latent patches
#                 #   shape: (W, H)
#                 # - patches: latent patches
#                 #   shape: (D, W, H, W1, H1)
#                 for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):
#                     #Check if label of this image is in one of the leaves of the subtree
#                     if ys[batch_i].item() in leaf_labels:
#                         # Find the index of the latent patch that is closest to the prototype
#                         min_distance = distances.min()
#                         min_distance_ix = distances.argmin()
#                         # Use the index to get the closest latent patch
#                         closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]
#
#                         # Check if the latent patch is closest for all data samples seen so far
#                         if min_distance < global_min_proto_dist[j]:
#                             global_min_proto_dist[j] = min_distance
#                             global_min_patches[j] = closest_patch
#                             global_min_info[j] = {
#                                 'input_image_ix': i * batch_size + batch_i,
#                                 'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
#                                 'W': W,
#                                 'H': H,
#                                 'W1': W1,
#                                 'H1': H1,
#                                 'distance': min_distance.item(),
#                                 'nearest_input': torch.unsqueeze(xs[batch_i],0),
#                                 'node_ix': node.index,
#                             }
#
#             # Update the progress bar if required
#             projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')
#
#             del features_batch
#             del distances_batch
#             del out_map
#
#         # Copy the patches to the prototype layer weights
#         # projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
#         #                         dim=0, out=tree.prototype_layer.prototype_vectors)
#         # del projection
#
#     return global_min_info, tree
# add orthogonality
# def project_with_class_constraints(tree: ProtoTree,
#                                     project_loader: DataLoader,
#                                     device,
#                                     args: argparse.Namespace,
#                                     log: Log,
#                                     log_prefix: str = 'log_projection_with_constraints',  # TODO
#                                     progress_prefix: str = 'Projection'
#                                     ) -> dict:
#
#     log.log_message("\nProject prototypes to the nearest training patches (with class constraints)...")
#     # Set the model to evaluation mode
#     tree.eval()
#     torch.cuda.empty_cache()
#     # instantiate the PoincareBall manifold from geoopt
#     manifold = geoopt.PoincareBall()
#     # The goal is to find the latent patch that minimizes the L2 distance to each prototype
#     # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
#     # Also store info about the image that was used for projection
#     global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
#     global_min_patches = {j: None for j in range(tree.num_prototypes)}
#     global_min_info = {j: None for j in range(tree.num_prototypes)}
#
#     # Get the shape of the prototypes
#     D,W1,H1 = tree.prototype_shape
#
#     # Build a progress bar for showing the status
#     projection_iter = tqdm(enumerate(project_loader),
#                             total=len(project_loader),
#                             desc=progress_prefix,
#                             ncols=0
#                             )
#
#     with torch.no_grad():
#         # Get a batch of data
#         xs, ys = next(iter(project_loader))
#         batch_size = xs.shape[0]
#         # For each internal node, collect the leaf labels in the subtree with this node as root.
#         # Only images from these classes can be used for projection.
#         leaf_labels_subtree = dict()
#
#         for branch, j in tree._out_map.items():
#             leaf_labels_subtree[branch.index] = set()
#             for leaf in branch.leaves:
#                 leaf_labels_subtree[branch.index].add(torch.argmax(leaf.distribution()).item())
#
#         for i, (xs, ys) in projection_iter:
#             xs, ys = xs.to(device), ys.to(device)
#             # Get the features and distances
#             # - features_batch: features tensor (shared by all prototypes)
#             #   shape: (batch_size, D, W, H)
#             # - distances_batch: distances tensor (for all prototypes)
#             #   shape: (batch_size, num_prototypes, W, H)
#             # - out_map: a dict mapping decision nodes to distances (indices)
#             features_batch, distances_batch, out_map = tree.forward_partial(xs)
#             #distances_batch = manifold.dist2plane(tree.prototype_layer.prototype_vectors, c=1.0).sum(dim=1)
#             # Get the features dimensions
#             bs, D, W, H = features_batch.shape
#
#             # Get a tensor containing the individual latent patches
#             # Create the patches by unfolding over both the W and H dimensions
#             # TODO -- support for strides in the prototype layer? (corresponds to step size here)
#             patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)
#
#             # store the features corresponding to the smallest distance of each node
#             chosen_features = {}
#             threshold= 1e-4  # initial threshold
#             # Iterate over all decision nodes/prototypes
#             for node, j in out_map.items():
#                 leaf_labels = leaf_labels_subtree[node.index]
#
#                 for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):
#                     # Check if label of this image is in one of the leaves of the subtree
#                     if ys[batch_i].item() in leaf_labels:
#                         min_distance = distances.min()
#                         min_distance_ix = distances.argmin()
#                         closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]
#
#                         # check whether the current feature is orthogonal to the selected features
#                         is_orthogonal = True
#                         for chosen_node, chosen_patch in chosen_features.items():
#                             if chosen_node != node.index:  #features of previous nodes are not checked for orthogonality against features of other nodes
#                                 # compute the inner product and check whether it is close to zero
#                                 inner_product = (closest_patch.flatten() @ chosen_patch.flatten()).item()
#                                 if abs(inner_product) > threshold:  # threshold can be adjusted as needed
#                                     is_orthogonal = False
#                                     break
#
#                         # if the current feature is orthogonal to all selected features, update the global minimum distance
#                         if is_orthogonal and min_distance < global_min_proto_dist[j]:
#                             global_min_proto_dist[j] = min_distance
#                             global_min_patches[j] = closest_patch
#                             global_min_info[j] = {
#                                 'input_image_ix': i * batch_size + batch_i,
#                                 'patch_ix': min_distance_ix.item(),
#                                 'W': W,
#                                 'H': H,
#                                 'W1': W1,
#                                 'H1': H1,
#                                 'distance': min_distance.item(),
#                                 'nearest_input': torch.unsqueeze(xs[batch_i], 0),
#                                 'node_ix': node.index,
#                             }
#                             '''
#                             Under the current logic, the existence of is_orthogonal is meaningful: it helps determine whether the current feature can be accepted as an orthogonal feature.
#                             As soon as a feature that does not satisfy the condition is found in the loop, is_orthogonal is set to False, thereby effectively filtering out non-conforming features.
#                             If no non-conforming feature is found, is_orthogonal remains True,
#                             '''
#
#                             # store the current feature
#                             chosen_features[node.index] = closest_patch
#
#             # Update the progress bar if required
#             projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')
#
#             del features_batch
#             del distances_batch
#             del out_map
#
#
#
#     return global_min_info, tree
def project_with_class_constraints(tree: ProtoTree,
                                   project_loader: DataLoader,
                                   device,
                                   args: argparse.Namespace,
                                   log: Log,
                                   log_prefix: str = 'log_projection_with_constraints',  # TODO
                                   progress_prefix: str = 'Projection'
                                   ) -> dict:
    log.log_message("\nProject prototypes to the nearest training patches (with class constraints)...")
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()


    # Initialize data structures to store the closest patches and their info
    global_min_proto_dist = {j: [] for j in range(tree.num_prototypes)}
    global_min_patches = {j: [] for j in range(tree.num_prototypes)}
    global_min_info = {j: [] for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    D, W1, H1 = tree.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                           total=len(project_loader),
                           desc=progress_prefix,
                           ncols=0
                           )

    # Get a batch of data to initialize batch_size
    xs, ys = next(iter(project_loader))
    batch_size = xs.shape[0]

    # Create leaf_labels_subtree to store the leaf labels for each subtree
    leaf_labels_subtree = dict()
    for branch, j in tree._out_map.items():
        leaf_labels_subtree[branch.index] = set()
        for leaf in branch.leaves:
            leaf_labels_subtree[branch.index].add(torch.argmax(leaf.distribution()).item())

    with torch.no_grad():
        for i, (xs, ys) in projection_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Get the features and distances
            features_batch, distances_batch, out_map = tree.forward_partial(xs)
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # store the features corresponding to the smallest distance of each node
            chosen_features = {}
            threshold = 1e-4  # initial threshold
            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():
                leaf_labels = leaf_labels_subtree[node.index]

                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):
                    # Check if label of this image is in one of the leaves of the subtree
                    if ys[batch_i].item() in leaf_labels:
                        min_distance = distances.min()
                        min_distance_ix = distances.argmin()
                        closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

                        # check whether the current feature is orthogonal to the selected features
                        is_orthogonal = True
                        for chosen_node, chosen_patch in chosen_features.items():
                            if chosen_node != node.index:  # features of previous nodes are not checked for orthogonality against features of other nodes
                                inner_product = (closest_patch.flatten() @ chosen_patch.flatten()).item()
                                if abs(inner_product) > threshold:  # threshold can be adjusted as needed
                                    is_orthogonal = False
                                    break

                        # if the current feature is orthogonal to all selected features, update the global minimum distance
                        if is_orthogonal:
                            # Check if we have less than 3 unique entries for this prototype
                            if len(global_min_proto_dist[j]) < 3:
                                if min_distance.item() not in global_min_proto_dist[j]:  # ensure the distance is unique
                                    global_min_proto_dist[j].append(min_distance.item())
                                    global_min_patches[j].append(closest_patch)
                                    global_min_info[j].append({
                                        'input_image_ix': i * batch_size + batch_i,
                                        'patch_ix': min_distance_ix.item(),
                                        'W': W,
                                        'H': H,
                                        'W1': W1,
                                        'H1': H1,
                                        'distance': min_distance.item(),
                                        'nearest_input': torch.unsqueeze(xs[batch_i], 0),
                                        'node_ix': node.index,
                                    })
                            else:
                                # If we already have 3 unique entries, check if the new distance is less than the max
                                max_distance = max(global_min_proto_dist[j])
                                if min_distance.item() < max_distance and min_distance.item() not in global_min_proto_dist[j]:
                                    # Replace the entry with the maximum distance
                                    max_index = global_min_proto_dist[j].index(max_distance)
                                    global_min_proto_dist[j][max_index] = min_distance.item()
                                    global_min_patches[j][max_index] = closest_patch
                                    global_min_info[j][max_index] = {
                                        'input_image_ix': i * batch_size + batch_i,
                                        'patch_ix': min_distance_ix.item(),
                                        'W': W,
                                        'H': H,
                                        'W1': W1,
                                        'H1': H1,
                                        'distance': min_distance.item(),
                                        'nearest_input': torch.unsqueeze(xs[batch_i], 0),
                                        'node_ix': node.index,
                                    }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

            del features_batch
            del distances_batch
            del out_map

    return global_min_info, tree

