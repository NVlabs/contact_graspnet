import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
TF2 = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'pointnet2', 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'pointnet2'))

import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point

import mesh_utils

def placeholder_inputs(batch_size, num_input_points=20000, input_normals=False):
    """
    Creates placeholders for input pointclouds, scene indices, camera poses and training/eval mode 

    Arguments:
        batch_size {int} -- batch size
        num_input_points {int} -- number of input points to the network (default: 20000)

    Keyword Arguments:
        input_normals {bool} -- whether to use normals as input (default: {False})

    Returns:
        dict[str:tf.placeholder] -- dict of placeholders
    """

    pl_dict = {}
    dim = 6 if input_normals else 3
    pl_dict['pointclouds_pl'] = tf.placeholder(tf.float32, shape=(batch_size, num_input_points, dim))
    pl_dict['scene_idx_pl'] = tf.placeholder(tf.int32, ())
    pl_dict['cam_poses_pl'] = tf.placeholder(tf.float32, shape=(batch_size, 4, 4))
    pl_dict['is_training_pl'] = tf.placeholder(tf.bool, shape=())

    return pl_dict

def get_bin_vals(global_config):
    """
    Creates bin values for grasping widths according to bounds defined in config

    Arguments:
        global_config {dict} -- config

    Returns:
        tf.constant -- bin value tensor 
    """
    bins_bounds = np.array(global_config['DATA']['labels']['offset_bins'])
    if global_config['TEST']['bin_vals'] == 'max':
        bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
        bin_vals[-1] = bins_bounds[-1]
    elif global_config['TEST']['bin_vals'] == 'mean':
        bin_vals = bins_bounds[1:]
    else:
        raise NotImplementedError

    if not global_config['TEST']['allow_zero_margin']:
        bin_vals = np.minimum(bin_vals, global_config['DATA']['gripper_width']-global_config['TEST']['extra_opening'])
        
    tf_bin_vals = tf.constant(bin_vals, tf.float32)
    return tf_bin_vals

def get_model(point_cloud, is_training, global_config, bn_decay=None):
    """
    Contact-GraspNet model consisting of a PointNet++ backbone and multiple output heads

    Arguments:
        point_cloud {tf.placeholder} -- batch of point clouds
        is_training {bool} -- train or eval mode
        global_config {dict} -- config

    Keyword Arguments:
        bn_decay {tf.variable} -- batch norm decay (default: {None})

    Returns:
        [dict] -- endpoints of the network
    """

    model_config = global_config['MODEL']
    data_config = global_config['DATA']

    radius_list_0 = model_config['pointnet_sa_modules_msg'][0]['radius_list']
    radius_list_1 = model_config['pointnet_sa_modules_msg'][1]['radius_list']
    radius_list_2 = model_config['pointnet_sa_modules_msg'][2]['radius_list']
    
    nsample_list_0 = model_config['pointnet_sa_modules_msg'][0]['nsample_list']
    nsample_list_1 = model_config['pointnet_sa_modules_msg'][1]['nsample_list']
    nsample_list_2 = model_config['pointnet_sa_modules_msg'][2]['nsample_list']
    
    mlp_list_0 = model_config['pointnet_sa_modules_msg'][0]['mlp_list']
    mlp_list_1 = model_config['pointnet_sa_modules_msg'][1]['mlp_list']
    mlp_list_2 = model_config['pointnet_sa_modules_msg'][2]['mlp_list']
    
    npoint_0 = model_config['pointnet_sa_modules_msg'][0]['npoint']
    npoint_1 = model_config['pointnet_sa_modules_msg'][1]['npoint']
    npoint_2 = model_config['pointnet_sa_modules_msg'][2]['npoint']
    
    fp_mlp_0 = model_config['pointnet_fp_modules'][0]['mlp']
    fp_mlp_1 = model_config['pointnet_fp_modules'][1]['mlp']
    fp_mlp_2 = model_config['pointnet_fp_modules'][2]['mlp']

    input_normals = data_config['input_normals']
    offset_bins = data_config['labels']['offset_bins']
    joint_heads = model_config['joint_heads']

    # expensive, rather use random only
    if 'raw_num_points' in data_config and data_config['raw_num_points'] != data_config['ndataset_points']:
        point_cloud = gather_point(point_cloud, farthest_point_sample(data_config['ndataset_points'], point_cloud))

    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3]) if input_normals else None 

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, npoint_0, radius_list_0, nsample_list_0, mlp_list_0, is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, npoint_1, radius_list_1, nsample_list_1,mlp_list_1, is_training, bn_decay, scope='layer2')

    if 'asymmetric_model' in model_config and model_config['asymmetric_model']:
        l3_xyz, l3_points = pointnet_sa_module_msg(l2_xyz, l2_points, npoint_2, radius_list_2, nsample_list_2,mlp_list_2, is_training, bn_decay, scope='layer3')
        l4_xyz, l4_points, _ = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=model_config['pointnet_sa_module']['mlp'], mlp2=None, group_all=model_config['pointnet_sa_module']['group_all'], is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, fp_mlp_0, is_training, bn_decay, scope='fa_layer1')
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, fp_mlp_1, is_training, bn_decay, scope='fa_layer2')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, fp_mlp_2, is_training, bn_decay, scope='fa_layer3')

        l0_points = l1_points
        pred_points = l1_xyz
    else:
        l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=model_config['pointnet_sa_module']['mlp'], mlp2=None, group_all=model_config['pointnet_sa_module']['group_all'], is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Feature Propagation layers
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, fp_mlp_0, is_training, bn_decay, scope='fa_layer1')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, fp_mlp_1, is_training, bn_decay, scope='fa_layer2')
        l0_points = tf.concat([l0_xyz, l0_points],axis=-1) if input_normals else l0_xyz 
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, fp_mlp_2, is_training, bn_decay, scope='fa_layer3')
        pred_points = l0_xyz

    if joint_heads:
        head = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        head = tf_util.dropout(head, keep_prob=0.7, is_training=is_training, scope='dp1')
        head = tf_util.conv1d(head, 4, 1, padding='VALID', activation_fn=None, scope='fc2')
        grasp_dir_head = tf.slice(head, [0,0,0], [-1,-1,3])
        grasp_dir_head = tf.math.l2_normalize(grasp_dir_head, axis=2)
        binary_seg_head = tf.slice(head, [0,0,3], [-1,-1,1])
    else:
        # Head for grasp direction
        grasp_dir_head = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        grasp_dir_head = tf_util.dropout(grasp_dir_head, keep_prob=0.7, is_training=is_training, scope='dp1')
        grasp_dir_head = tf_util.conv1d(grasp_dir_head, 3, 1, padding='VALID', activation_fn=None, scope='fc3')
        grasp_dir_head_normed = tf.math.l2_normalize(grasp_dir_head, axis=2)

        # Head for grasp approach
        approach_dir_head = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_app', bn_decay=bn_decay)
        approach_dir_head = tf_util.dropout(approach_dir_head, keep_prob=0.7, is_training=is_training, scope='dp1_app')
        approach_dir_head = tf_util.conv1d(approach_dir_head, 3, 1, padding='VALID', activation_fn=None, scope='fc3_app')
        approach_dir_head_orthog = tf.math.l2_normalize(approach_dir_head - tf.reduce_sum(tf.multiply(grasp_dir_head_normed, approach_dir_head), axis=2, keepdims=True)*grasp_dir_head_normed, axis=2)
        
        # Head for grasp width
        if model_config['dir_vec_length_offset']:
            grasp_offset_head = tf.norm(grasp_dir_head, axis=2, keepdims=True)
        elif model_config['bin_offsets']:
            grasp_offset_head = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_off', bn_decay=bn_decay)
            grasp_offset_head = tf_util.conv1d(grasp_offset_head, len(offset_bins)-1, 1, padding='VALID', activation_fn=None, scope='fc2_off')
        else:
            grasp_offset_head = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_off', bn_decay=bn_decay)
            grasp_offset_head = tf_util.dropout(grasp_offset_head, keep_prob=0.7, is_training=is_training, scope='dp1_off')
            grasp_offset_head = tf_util.conv1d(grasp_offset_head, 1, 1, padding='VALID', activation_fn=None, scope='fc2_off')

        # Head for contact points
        binary_seg_head = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_seg', bn_decay=bn_decay)
        binary_seg_head = tf_util.dropout(binary_seg_head, keep_prob=0.5, is_training=is_training, scope='dp1_seg')
        binary_seg_head = tf_util.conv1d(binary_seg_head, 1, 1, padding='VALID', activation_fn=None, scope='fc2_seg')

    end_points['grasp_dir_head'] = grasp_dir_head_normed
    end_points['binary_seg_head'] = binary_seg_head
    end_points['binary_seg_pred'] = tf.math.sigmoid(binary_seg_head)
    end_points['grasp_offset_head'] = grasp_offset_head
    end_points['grasp_offset_pred'] = tf.math.sigmoid(grasp_offset_head) if model_config['bin_offsets'] else grasp_offset_head
    end_points['approach_dir_head'] = approach_dir_head_orthog
    end_points['pred_points'] = pred_points

    return end_points

def build_6d_grasp(approach_dirs, base_dirs, contact_pts, thickness, use_tf=False, gripper_depth = 0.1034):
    """
    Build 6-DoF grasps + width from point-wise network predictions

    Arguments:
        approach_dirs {np.ndarray/tf.tensor} -- Nx3 approach direction vectors
        base_dirs {np.ndarray/tf.tensor} -- Nx3 base direction vectors
        contact_pts {np.ndarray/tf.tensor} -- Nx3 contact points
        thickness {np.ndarray/tf.tensor} -- Nx1 grasp width

    Keyword Arguments:
        use_tf {bool} -- whether inputs and outputs are tf tensors (default: {False})
        gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})

    Returns:
        np.ndarray -- Nx4x4 grasp poses in camera coordinates
    """
    if use_tf:
        grasps_R = tf.stack([base_dirs, tf.linalg.cross(approach_dirs,base_dirs),approach_dirs], axis=3)        
        grasps_t = contact_pts + tf.expand_dims(thickness,2)/2 * base_dirs - gripper_depth * approach_dirs
        ones = tf.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=tf.float32)
        zeros = tf.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=tf.float32)
        homog_vec = tf.concat([zeros, ones], axis=3)
        grasps = tf.concat([tf.concat([grasps_R,  tf.expand_dims(grasps_t, 3)], axis=3), homog_vec], axis=2)
    else:
        grasps = []
        for i in range(len(contact_pts)):
            grasp = np.eye(4)

            grasp[:3,0] = base_dirs[i] / np.linalg.norm(base_dirs[i])
            grasp[:3,2] = approach_dirs[i] / np.linalg.norm(approach_dirs[i])
            grasp_y = np.cross( grasp[:3,2],grasp[:3,0])
            grasp[:3,1] = grasp_y / np.linalg.norm(grasp_y)
            # base_gripper xyz = contact + thickness / 2 * baseline_dir - gripper_d * approach_dir 
            grasp[:3,3] = contact_pts[i] + thickness[i] / 2 * grasp[:3,0] - gripper_depth * grasp[:3,2]
            # grasp[0,3] = finger_width
            grasps.append(grasp)
        grasps = np.array(grasps)

    return grasps

def get_losses(pointclouds_pl, end_points, dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam, global_config):
    """
    Computes loss terms from pointclouds, network predictions and labels 

    Arguments:
        pointclouds_pl {tf.placeholder} -- bxNx3 input point clouds
        end_points {dict[str:tf.variable]} -- endpoints of the network containing predictions
        dir_labels_pc_cam {tf.variable} -- base direction labels in camera coordinates (bxNx3)
        offset_labels_pc {tf.variable} -- grasp width labels (bxNx1) 
        grasp_success_labels_pc {tf.variable} -- contact success labels (bxNx1) 
        approach_labels_pc_cam {tf.variable} -- approach direction labels in camera coordinates (bxNx3)
        global_config {dict} -- config dict 
        
    Returns:
        [dir_cosine_loss, bin_ce_loss, offset_loss, approach_cosine_loss, adds_loss, 
        adds_loss_gt2pred, gt_control_points, pred_control_points, pos_grasps_in_view] -- All losses (not all are used for training)
    """

    grasp_dir_head = end_points['grasp_dir_head']
    grasp_offset_head = end_points['grasp_offset_head']
    approach_dir_head = end_points['approach_dir_head']

    bin_weights = global_config['DATA']['labels']['bin_weights']
    tf_bin_weights = tf.constant(bin_weights)
    
    min_geom_loss_divisor = tf.constant(float(global_config['LOSS']['min_geom_loss_divisor'])) if 'min_geom_loss_divisor' in global_config['LOSS'] else tf.constant(1.)
    pos_grasps_in_view = tf.math.maximum(tf.reduce_sum(grasp_success_labels_pc, axis=1), min_geom_loss_divisor)   

    ### ADS Gripper PC Loss
    if global_config['MODEL']['bin_offsets']:
        thickness_pred = tf.gather_nd(get_bin_vals(global_config), tf.expand_dims(tf.argmax(grasp_offset_head, axis=2), axis=2))
        thickness_gt = tf.gather_nd(get_bin_vals(global_config), tf.expand_dims(tf.argmax(offset_labels_pc, axis=2), axis=2))
    else:
        thickness_pred = grasp_offset_head[:,:,0]
        thickness_gt = offset_labels_pc[:,:,0]
    pred_grasps = build_6d_grasp(approach_dir_head, grasp_dir_head, pointclouds_pl, thickness_pred, use_tf=True) # b x num_point x 4 x 4
    gt_grasps_proj = build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pointclouds_pl, thickness_gt, use_tf=True) # b x num_point x 4 x 4
    pos_gt_grasps_proj = tf.where(tf.broadcast_to(tf.expand_dims(tf.expand_dims(tf.cast(grasp_success_labels_pc, tf.bool),2),3), gt_grasps_proj.shape), gt_grasps_proj, tf.ones_like(gt_grasps_proj)*100000)
    # pos_gt_grasps_proj = tf.reshape(pos_gt_grasps_proj, (global_config['OPTIMIZER']['batch_size'], -1, 4, 4)) 

    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size']) # b x 5 x 3
    sym_gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size'], symmetric=True)

    gripper_control_points_homog =  tf.concat([gripper_control_points, tf.ones((global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], axis=2)  # b x 5 x 4
    sym_gripper_control_points_homog =  tf.concat([sym_gripper_control_points, tf.ones((global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], axis=2)  # b x 5 x 4
    
    # only use per point pred grasps but not per point gt grasps
    control_points = tf.keras.backend.repeat_elements(tf.expand_dims(gripper_control_points_homog,1), gt_grasps_proj.shape[1], axis=1)  # b x num_point x 5 x 4
    sym_control_points = tf.keras.backend.repeat_elements(tf.expand_dims(sym_gripper_control_points_homog,1), gt_grasps_proj.shape[1], axis=1)  # b x num_point x 5 x 4
    pred_control_points = tf.matmul(control_points, tf.transpose(pred_grasps, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_point x 5 x 3

    ### Pred Grasp to GT Grasp ADD-S Loss
    gt_control_points = tf.matmul(control_points, tf.transpose(pos_gt_grasps_proj, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_pos_grasp_point x 5 x 3
    sym_gt_control_points = tf.matmul(sym_control_points, tf.transpose(pos_gt_grasps_proj, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_pos_grasp_point x 5 x 3

    squared_add = tf.reduce_sum((tf.expand_dims(pred_control_points,2)-tf.expand_dims(gt_control_points,1))**2, axis=(3,4)) # b x num_point x num_pos_grasp_point x ( 5 x 3)
    sym_squared_add = tf.reduce_sum((tf.expand_dims(pred_control_points,2)-tf.expand_dims(sym_gt_control_points,1))**2, axis=(3,4)) # b x num_point x num_pos_grasp_point x ( 5 x 3)

    # symmetric ADD-S
    neg_squared_adds = -tf.concat([squared_add,sym_squared_add], axis=2) # b x num_point x 2num_pos_grasp_point
    neg_squared_adds_k = tf.math.top_k(neg_squared_adds, k=1, sorted=False)[0] # b x num_point
    # If any pos grasp exists
    min_adds = tf.minimum(tf.reduce_sum(grasp_success_labels_pc, axis=1, keepdims=True), tf.ones_like(neg_squared_adds_k[:,:,0])) * tf.sqrt(-neg_squared_adds_k[:,:,0])#tf.minimum(tf.sqrt(-neg_squared_adds_k), tf.ones_like(neg_squared_adds_k)) # b x num_point
    adds_loss = tf.reduce_mean(end_points['binary_seg_pred'][:,:,0] * min_adds)

    ### GT Grasp to pred Grasp ADD-S Loss
    gt_control_points = tf.matmul(control_points, tf.transpose(gt_grasps_proj, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_pos_grasp_point x 5 x 3
    sym_gt_control_points = tf.matmul(sym_control_points, tf.transpose(gt_grasps_proj, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_pos_grasp_point x 5 x 3

    neg_squared_adds = -tf.reduce_sum((tf.expand_dims(pred_control_points,1)-tf.expand_dims(gt_control_points,2))**2, axis=(3,4)) # b x num_point x num_pos_grasp_point x ( 5 x 3)
    neg_squared_adds_sym = -tf.reduce_sum((tf.expand_dims(pred_control_points,1)-tf.expand_dims(sym_gt_control_points,2))**2, axis=(3,4)) # b x num_point x num_pos_grasp_point x ( 5 x 3)

    neg_squared_adds_k_gt2pred, pred_grasp_idcs = tf.math.top_k(neg_squared_adds, k=1, sorted=False) # b x num_pos_grasp_point
    neg_squared_adds_k_sym_gt2pred, pred_grasp_sym_idcs = tf.math.top_k(neg_squared_adds_sym, k=1, sorted=False) # b x num_pos_grasp_point
    pred_grasp_idcs_joined = tf.where(neg_squared_adds_k_gt2pred<neg_squared_adds_k_sym_gt2pred, pred_grasp_sym_idcs, pred_grasp_idcs)
    min_adds_gt2pred = tf.minimum(-neg_squared_adds_k_gt2pred, -neg_squared_adds_k_sym_gt2pred) # b x num_pos_grasp_point x 1
    # min_adds_gt2pred = tf.math.exp(-min_adds_gt2pred)
    masked_min_adds_gt2pred = tf.multiply(min_adds_gt2pred[:,:,0], grasp_success_labels_pc)
    
    batch_idcs = tf.meshgrid(tf.range(pred_grasp_idcs_joined.shape[1]), tf.range(pred_grasp_idcs_joined.shape[0]))
    gather_idcs = tf.stack((batch_idcs[1],pred_grasp_idcs_joined[:,:,0]), axis=2)
    nearest_pred_grasp_confidence = tf.gather_nd(end_points['binary_seg_pred'][:,:,0], gather_idcs)
    adds_loss_gt2pred = tf.reduce_mean(tf.reduce_sum(nearest_pred_grasp_confidence*masked_min_adds_gt2pred, axis=1) / pos_grasps_in_view)
 
    ### Grasp baseline Loss
    cosine_distance = tf.constant(1.)-tf.reduce_sum(tf.multiply(dir_labels_pc_cam, grasp_dir_head),axis=2)
    # only pass loss where we have labeled contacts near pc points 
    masked_cosine_loss = tf.multiply(cosine_distance, grasp_success_labels_pc)
    dir_cosine_loss = tf.reduce_mean(tf.reduce_sum(masked_cosine_loss, axis=1) / pos_grasps_in_view)

    ### Grasp Approach Loss
    approach_labels_orthog = tf.math.l2_normalize(approach_labels_pc_cam - tf.reduce_sum(tf.multiply(grasp_dir_head, approach_labels_pc_cam), axis=2, keepdims=True)*grasp_dir_head, axis=2)
    cosine_distance_approach = tf.constant(1.)-tf.reduce_sum(tf.multiply(approach_labels_orthog, approach_dir_head), axis=2)
    masked_approach_loss = tf.multiply(cosine_distance_approach, grasp_success_labels_pc)
    approach_cosine_loss = tf.reduce_mean(tf.reduce_sum(masked_approach_loss, axis=1) / pos_grasps_in_view)

    ### Grasp Offset/Thickness Loss
    if global_config['MODEL']['bin_offsets']:
        if global_config['LOSS']['offset_loss_type'] == 'softmax_cross_entropy':
            offset_loss = tf.losses.softmax_cross_entropy(offset_labels_pc, grasp_offset_head)
        else:
            offset_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=offset_labels_pc, logits=grasp_offset_head)
            
            if 'too_small_offset_pred_bin_factor' in global_config['LOSS'] and global_config['LOSS']['too_small_offset_pred_bin_factor']:
                too_small_offset_pred_bin_factor = tf.constant(global_config['LOSS']['too_small_offset_pred_bin_factor'], tf.float32)
                collision_weight = tf.math.cumsum(offset_labels_pc, axis=2, reverse=True)*too_small_offset_pred_bin_factor + tf.constant(1.)
                offset_loss = tf.multiply(collision_weight, offset_loss)

            offset_loss = tf.reduce_mean(tf.multiply(tf.reshape(tf_bin_weights,(1,1,-1)), offset_loss),axis=2)
    else:
        offset_loss = (grasp_offset_head[:,:,0] - offset_labels_pc[:,:,0])**2
    masked_offset_loss = tf.multiply(offset_loss, grasp_success_labels_pc)        
    offset_loss = tf.reduce_mean(tf.reduce_sum(masked_offset_loss, axis=1) / pos_grasps_in_view)

    ### Grasp Confidence Loss
    bin_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(grasp_success_labels_pc,axis=2), logits=end_points['binary_seg_head'])
    if 'topk_confidence' in global_config['LOSS'] and global_config['LOSS']['topk_confidence']:
        bin_ce_loss,_ = tf.math.top_k(tf.squeeze(bin_ce_loss), k=global_config['LOSS']['topk_confidence'])
    bin_ce_loss = tf.reduce_mean(bin_ce_loss)

    return dir_cosine_loss, bin_ce_loss, offset_loss, approach_cosine_loss, adds_loss, adds_loss_gt2pred

def multi_bin_labels(cont_labels, bin_boundaries):
    """
    Computes binned grasp width labels from continous labels and bin boundaries

    Arguments:
        cont_labels {tf.Variable} -- continouos labels
        bin_boundaries {list} -- bin boundary values

    Returns:
        tf.Variable -- one/multi hot bin labels
    """
    bins = []
    for b in range(len(bin_boundaries)-1):
        bins.append(tf.math.logical_and(tf.greater_equal(cont_labels, bin_boundaries[b]), tf.less(cont_labels,bin_boundaries[b+1])))
    multi_hot_labels = tf.concat(bins, axis=2)
    multi_hot_labels = tf.cast(multi_hot_labels, tf.float32)

    return multi_hot_labels


def compute_labels(pos_contact_pts_mesh, pos_contact_dirs_mesh, pos_contact_approaches_mesh, pos_finger_diffs, pc_cam_pl, camera_pose_pl, global_config):
    """
    Project grasp labels defined on meshes onto rendered point cloud from a camera pose via nearest neighbor contacts within a maximum radius. 
    All points without nearby successful grasp contacts are considered negativ contact points.

    Arguments:
        pos_contact_pts_mesh {tf.constant} -- positive contact points on the mesh scene (Mx3)
        pos_contact_dirs_mesh {tf.constant} -- respective contact base directions in the mesh scene (Mx3)
        pos_contact_approaches_mesh {tf.constant} -- respective contact approach directions in the mesh scene (Mx3)
        pos_finger_diffs {tf.constant} -- respective grasp widths in the mesh scene (Mx1)
        pc_cam_pl {tf.placeholder} -- bxNx3 rendered point clouds
        camera_pose_pl {tf.placeholder} -- bx4x4 camera poses
        global_config {dict} -- global config

    Returns:
        [dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam] -- Per-point contact success labels and per-contact pose labels in rendered point cloud
    """
    label_config = global_config['DATA']['labels']
    model_config = global_config['MODEL']

    nsample = label_config['k']
    radius = label_config['max_radius']
    filter_z = label_config['filter_z']
    z_val = label_config['z_val']

    xyz_cam = pc_cam_pl[:,:,:3]
    pad_homog = tf.ones((xyz_cam.shape[0],xyz_cam.shape[1], 1)) 
    pc_mesh = tf.matmul(tf.concat([xyz_cam, pad_homog], 2), tf.transpose(tf.linalg.inv(camera_pose_pl),perm=[0, 2, 1]))[:,:,:3]

    contact_point_offsets_batch = tf.keras.backend.repeat_elements(tf.expand_dims(pos_finger_diffs,0), pc_mesh.shape[0], axis=0)

    pad_homog2 = tf.ones((pc_mesh.shape[0], pos_contact_dirs_mesh.shape[0], 1)) 
    contact_point_dirs_batch = tf.keras.backend.repeat_elements(tf.expand_dims(pos_contact_dirs_mesh,0), pc_mesh.shape[0], axis=0)
    contact_point_dirs_batch_cam = tf.matmul(contact_point_dirs_batch, tf.transpose(camera_pose_pl[:,:3,:3], perm=[0, 2, 1]))[:,:,:3]

    pos_contact_approaches_batch = tf.keras.backend.repeat_elements(tf.expand_dims(pos_contact_approaches_mesh,0), pc_mesh.shape[0], axis=0)
    pos_contact_approaches_batch_cam = tf.matmul(pos_contact_approaches_batch, tf.transpose(camera_pose_pl[:,:3,:3], perm=[0, 2, 1]))[:,:,:3]
    
    contact_point_batch_mesh = tf.keras.backend.repeat_elements(tf.expand_dims(pos_contact_pts_mesh,0), pc_mesh.shape[0], axis=0)
    contact_point_batch_cam = tf.matmul(tf.concat([contact_point_batch_mesh, pad_homog2], 2), tf.transpose(camera_pose_pl, perm=[0, 2, 1]))[:,:,:3]

    if filter_z:
        dir_filter_passed = tf.keras.backend.repeat_elements(tf.math.greater(contact_point_dirs_batch_cam[:,:,2:3], tf.constant([z_val])), 3, axis=2)
        contact_point_batch_mesh = tf.where(dir_filter_passed, contact_point_batch_mesh, tf.ones_like(contact_point_batch_mesh)*100000)

    squared_dists_all = tf.reduce_sum((tf.expand_dims(contact_point_batch_cam,1)-tf.expand_dims(xyz_cam,2))**2,axis=3)
    neg_squared_dists_k, close_contact_pt_idcs = tf.math.top_k(-squared_dists_all, k=nsample, sorted=False)
    squared_dists_k = -neg_squared_dists_k

    # Nearest neighbor mapping
    grasp_success_labels_pc = tf.cast(tf.less(tf.reduce_mean(squared_dists_k, axis=2), radius*radius), tf.float32) # (batch_size, num_point)

    grouped_dirs_pc_cam = group_point(contact_point_dirs_batch_cam, close_contact_pt_idcs)
    grouped_approaches_pc_cam = group_point(pos_contact_approaches_batch_cam, close_contact_pt_idcs)
    grouped_offsets = group_point(tf.expand_dims(contact_point_offsets_batch,2), close_contact_pt_idcs)

    dir_labels_pc_cam = tf.math.l2_normalize(tf.reduce_mean(grouped_dirs_pc_cam, axis=2),axis=2) # (batch_size, num_point, 3)
    approach_labels_pc_cam = tf.math.l2_normalize(tf.reduce_mean(grouped_approaches_pc_cam, axis=2),axis=2) # (batch_size, num_point, 3)
    offset_labels_pc = tf.reduce_mean(grouped_offsets, axis=2)
        
    return dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam

