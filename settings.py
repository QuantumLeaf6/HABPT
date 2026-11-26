img_size = 224
prototype_shape = (24, 256, 1, 1)
num_classes = 8
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

data_path ='data/dataset_8class/'
train_dir = data_path + 'train-6400/'
test_dir = data_path + 'test-800/'
train_push_dir = data_path + 'train-6400/'

# train_dir = data_path + 'train_cropped_augmented/'
# test_dir = data_path + 'test_cropped/'
# train_push_dir = data_path + 'train_cropped/'

train_batch_size = 80
test_batch_size = 100
train_push_batch_size =75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'orth': 1e-4,
    'sub_sep': -1e-7,
}
# for testing only
# num_train_epochs = 10
# num_warm_epochs = 1
#
# push_start = 3
# push_epochs = [i for i in range(num_train_epochs) if i % 3 == 0]

# # for normal use only
num_train_epochs = 41
num_warm_epochs =6

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

#print(push_epochs)
