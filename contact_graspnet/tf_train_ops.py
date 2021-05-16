
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
TF2 = True

from data import farthest_points

def get_learning_rate(step, optimizer_config):
    """
    Return learning rate at training step

    Arguments:
        step {tf.variable} -- training step
        optimizer_config {dict} -- optimizer config_path

    Returns:
        tf.variable -- learning rate
    """

    batch_size = optimizer_config['batch_size'] 
    base_learning_rate = optimizer_config['learning_rate'] 
    decay_step = float(optimizer_config['decay_step'])
    decay_rate = float(optimizer_config['decay_rate'])

    learning_rate = tf.train.exponential_decay(
                        base_learning_rate,  # base learning rate.
                        step * batch_size,  # current index into the dataset.
                        decay_step,          # decay step.
                        decay_rate,          # decay rate.
                        staircase=True)

    learning_rate = tf.maximum(learning_rate, 0.00001) # clip the learning rate!
    return learning_rate        

def get_bn_decay(step, optimizer_config):
    """
    Return batch norm decay at training step.

    Arguments:
        step {tf.variable} -- training step
        optimizer_config {dict} -- optimizer config

    Returns:
        tf.variable -- batch norm decay
    """

    batch_size = optimizer_config['batch_size']
    bn_init_decay = optimizer_config['bn_init_decay']
    bn_decay_decay_step = optimizer_config['bn_decay_decay_step']
    bn_decay_decay_rate = optimizer_config['bn_decay_decay_rate']
    bn_decay_clip = optimizer_config['bn_decay_clip']

    bn_momentum = tf.train.exponential_decay(
                      bn_init_decay,
                      step*batch_size,
                      bn_decay_decay_step,
                      bn_decay_decay_rate,
                      staircase=True)
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
    return bn_decay


def load_labels_and_losses(grasp_estimator, contact_infos, global_config, train=True):
    """
    Loads labels to memory and builds graph for computing losses

    Arguments:
        grasp_estimator {class} -- Grasp Estimator Instance
        contact_infos {list(dicts)} -- Per scene mesh: grasp contact information  
        global_config {dict} -- global config

    Keyword Arguments:
        train {bool} -- training mode (default: {True})

    Returns:
        dict[str:tf.variables] -- tf references to labels and losses
    """

    end_points = grasp_estimator.model_ops['end_points']
    target_point_cloud = end_points['pred_points']
    scene_idx_pl = grasp_estimator.placeholders['scene_idx_pl']
    cam_poses_pl = grasp_estimator.placeholders['cam_poses_pl']

    tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_approaches, \
    tf_pos_finger_diffs, tf_scene_idcs = load_contact_grasps(contact_infos, global_config['DATA'])

    iterator = None
    idx = scene_idx_pl
    if train:
        grasp_dataset = tf.data.Dataset.from_tensor_slices((tf_pos_contact_points, tf_pos_contact_dirs, \
                                                    tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs))

        grasp_dataset = grasp_dataset.repeat()
        grasp_dataset = grasp_dataset.batch(1)
        # grasp_dataset = grasp_dataset.prefetch_to_device(5)
        grasp_dataset = grasp_dataset.prefetch(3)
        grasp_dataset = grasp_dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        iterator = grasp_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, \
        tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx, tf_scene_idcs = next_element
        idx = 0
        
    # Get labels
    dir_labels_pc_cam, offset_labels_pc, \
    grasp_suc_labels_pc, approach_labels_pc = grasp_estimator._model_func.compute_labels(tf_pos_contact_points_idx[idx], tf_pos_contact_dirs_idx[idx], tf_pos_contact_approaches_idx[idx], 
                                                                                            tf_pos_finger_diffs_idx[idx], target_point_cloud, cam_poses_pl, global_config)

    
    if global_config['MODEL']['bin_offsets']:
        orig_offset_labels = offset_labels_pc
        offset_labels_pc = tf.abs(offset_labels_pc)
        offset_labels_pc = grasp_estimator._model_func.multi_bin_labels(offset_labels_pc, global_config['DATA']['labels']['offset_bins'])
        
    # Get losses 
    dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_loss_gt2pred = grasp_estimator._model_func.get_losses(target_point_cloud, end_points, dir_labels_pc_cam, 
                                                                                                                             offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc, 
                                                                                                                             global_config)

    total_loss = 0
    if global_config['MODEL']['pred_contact_base']:
        total_loss += global_config['OPTIMIZER']['dir_cosine_loss_weight'] * dir_loss
    if global_config['MODEL']['pred_contact_success']:
        total_loss += global_config['OPTIMIZER']['score_ce_loss_weight'] * bin_ce_loss
    if global_config['MODEL']['pred_contact_offset']:
        total_loss += global_config['OPTIMIZER']['offset_loss_weight'] * offset_loss
    if global_config['MODEL']['pred_contact_approach']:
        total_loss += global_config['OPTIMIZER']['approach_cosine_loss_weight'] * approach_loss
    if global_config['MODEL']['pred_grasps_adds']:
        total_loss += global_config['OPTIMIZER']['adds_loss_weight'] * adds_loss
    if global_config['MODEL']['pred_grasps_adds_gt2pred']:
        total_loss += global_config['OPTIMIZER']['adds_gt2pred_loss_weight'] * adds_loss_gt2pred

    tf_bin_vals = grasp_estimator._model_func.get_bin_vals(global_config)

    loss_label_ops = {'loss': total_loss,
                    'dir_loss': dir_loss,
                    'bin_ce_loss': bin_ce_loss,
                    'offset_loss': offset_loss,
                    'approach_loss': approach_loss,
                    'adds_loss': adds_loss,
                    'adds_gt2pred_loss': adds_loss_gt2pred,
                    'dir_labels_pc_cam': dir_labels_pc_cam,
                    'offset_labels_pc': offset_labels_pc,
                    'offset_label_idcs_pc': tf.argmax(offset_labels_pc, axis=2) if global_config['MODEL']['bin_offsets'] else None,
                    'offset_orig_labels_vals': orig_offset_labels if global_config['MODEL']['bin_offsets'] else None,
                    'offset_bin_label_vals': tf.gather_nd(tf_bin_vals, tf.expand_dims(tf.argmax(offset_labels_pc, axis=2), axis=2)) if global_config['MODEL']['bin_offsets'] else None,
                    'grasp_suc_labels_pc': grasp_suc_labels_pc,
                    'approach_labels_pc': approach_labels_pc,
                    'tf_bin_vals': tf_bin_vals,
                    'scene_idx': tf_scene_idcs,
                    'iterator': iterator
        }

    return loss_label_ops

def build_train_op(total_loss, step, global_config):
    """
    Initializes optimizer and learning rate scheduler

    Arguments:
        total_loss {tf.variable} -- Weighted sum of all loss terms
        step {tf.variable} -- training step
        global_config {dict} -- global config

    Returns:
        [train_op] -- Operation to run during training
    """

    print("--- Get training operator")
    # Get training operator
    learning_rate = get_learning_rate(step, global_config['OPTIMIZER'])
    tf.summary.scalar('learning_rate', learning_rate)
    if global_config['OPTIMIZER']['optimizer'] == 'momentum':
        momentum = global_config['OPTIMIZER']['momentum']
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif global_config['OPTIMIZER']['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train_op = optimizer.minimize(total_loss, global_step=step, var_list=tf.global_variables())
    if TF2:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, update_ops])

    return train_op

def load_contact_grasps(contact_list, data_config):
    """
    Loads fixed amount of contact grasp data per scene into tf CPU/GPU memory

    Arguments:
        contact_infos {list(dicts)} -- Per scene mesh: grasp contact information  
        data_config {dict} -- data config

    Returns:
        [tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_offsets, 
        tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs, 
        all_obj_paths, all_obj_transforms] -- tf.constants with per scene grasp data, object paths/transforms in scene
    """

    num_pos_contacts = data_config['labels']['num_pos_contacts'] 

    pos_contact_points = []
    pos_contact_dirs = []
    pos_finger_diffs = []
    pos_approach_dirs = []

    for i,c in enumerate(contact_list):
        contact_directions_01 = c['scene_contact_points'][:,0,:] - c['scene_contact_points'][:,1,:]
        all_contact_points = c['scene_contact_points'].reshape(-1,3)
        all_finger_diffs = np.maximum(np.linalg.norm(contact_directions_01,axis=1), np.finfo(np.float32).eps)
        all_contact_directions = np.empty((contact_directions_01.shape[0]*2, contact_directions_01.shape[1],))
        all_contact_directions[0::2] = -contact_directions_01 / all_finger_diffs[:,np.newaxis]
        all_contact_directions[1::2] = contact_directions_01 / all_finger_diffs[:,np.newaxis]
        all_contact_suc = np.ones_like(all_contact_points[:,0])
        all_grasp_transform = c['grasp_transforms'].reshape(-1,4,4)
        all_approach_directions = all_grasp_transform[:,:3,2]

        pos_idcs = np.where(all_contact_suc>0)[0]
        if len(pos_idcs) == 0:
            continue
        print('total positive contact points ', len(pos_idcs))
        
        all_pos_contact_points = all_contact_points[pos_idcs]
        all_pos_finger_diffs = all_finger_diffs[pos_idcs//2]
        all_pos_contact_dirs = all_contact_directions[pos_idcs]
        all_pos_approach_dirs = all_approach_directions[pos_idcs//2]
        
        # Use all positive contacts then mesh_utils with replacement
        if num_pos_contacts > len(all_pos_contact_points)/2:
            pos_sampled_contact_idcs = np.arange(len(all_pos_contact_points))
            pos_sampled_contact_idcs_replacement = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2 - len(all_pos_contact_points) , replace=True) 
            pos_sampled_contact_idcs= np.hstack((pos_sampled_contact_idcs, pos_sampled_contact_idcs_replacement))
        else:
            pos_sampled_contact_idcs = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2, replace=False)

        pos_contact_points.append(all_pos_contact_points[pos_sampled_contact_idcs,:])
        pos_contact_dirs.append(all_pos_contact_dirs[pos_sampled_contact_idcs,:])
        pos_finger_diffs.append(all_pos_finger_diffs[pos_sampled_contact_idcs])
        pos_approach_dirs.append(all_pos_approach_dirs[pos_sampled_contact_idcs])

    device = "/cpu:0" if 'to_gpu' in data_config['labels'] and not data_config['labels']['to_gpu'] else "/gpu:0"
    print("grasp label device: ", device)

    with tf.device(device):
        tf_scene_idcs = tf.constant(np.arange(0,len(pos_contact_points)), tf.int32)
        tf_pos_contact_points = tf.constant(np.array(pos_contact_points), tf.float32)
        tf_pos_contact_dirs =  tf.math.l2_normalize(tf.constant(np.array(pos_contact_dirs), tf.float32),axis=2)
        tf_pos_finger_diffs = tf.constant(np.array(pos_finger_diffs), tf.float32)
        tf_pos_contact_approaches =  tf.math.l2_normalize(tf.constant(np.array(pos_approach_dirs), tf.float32),axis=2)

    return tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs
