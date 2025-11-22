import os
import sys
import numpy as np
import argparse

from os.path import join as pjoin

# Optional torch import (only needed for advanced IK class at end of file)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional IK imports (not needed for basic remove_fs functionality)
try:
    from visualization.InverseKinematics import JacobianInverseKinematics, BasicInverseKinematics
    IK_AVAILABLE = True
except ImportError:
    IK_AVAILABLE = False
    JacobianInverseKinematics = None
    BasicInverseKinematics = None

# from visualization import BVH
# from scripts.motion_process_bvh import *
# from visualization.Animation import *


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r

def remove_fs_old(anim, glb, foot_contact, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True):
    # glb_height = 2.06820832 Not the case, may be use upper leg length
    scale = 1. #glb_height / 1.65 #scale to meter
    # fps = 20 #
    # velocity_thres = 10. # m/s
    height_thres = [0.06, 0.03] #[ankle, toe] meter
    if foot_contact is None:
        def foot_detect(positions, velfactor, heightfactor):
            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l_h = positions[:-1, fid_l, 1]
            feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float64)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r_h = positions[:-1, fid_r, 1]

            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float64)

            return feet_l, feet_r

        # feet_thre = 0.002
        # feet_vel_thre = np.array([velocity_thres**2, velocity_thres**2]) * scale**2 / fps**2
        feet_vel_thre = np.array([0.05, 0.2])
        # height_thre = np.array([0.06, 0.04]) * scale
        feet_h_thre = np.array(height_thres) * scale
        feet_l, feet_r = foot_detect(glb, velfactor=feet_vel_thre, heightfactor=feet_h_thre)
        foot = np.concatenate([feet_l, feet_r], axis=-1).transpose(1, 0)  # [4, T-1]
        foot = np.concatenate([foot, foot[:, -1:]], axis=-1)
    else:
        foot = foot_contact.transpose(1, 0)

    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
    # print(foot_heights)
    # floor_height = softmin(foot_heights, softness=0.03, axis=0)
    sort_height = np.sort(foot_heights)
    temp_len = len(sort_height)
    floor_height = np.mean(sort_height[int(0.25*temp_len):int(0.5*temp_len)])
    if floor_height > 0.5: # for motion like swim
        floor_height = 0
    # print(floor_height)
    # floor_height = foot_heights.min()
    # print(floor_height)
    # print(foot)
    # print(foot_heights.min())
    # print(floor_height)
    glb[:, :, 1] -= floor_height
    anim.positions[:, 0, 1] -= floor_height
    for i, fidx in enumerate(fid):
        fixed = foot[i]  # [T]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    targetmap = {}
    for j in range(glb.shape[1]):
        targetmap[j] = glb[:, j]

    # ik = BasicInverseKinematics(anim, glb, iterations=5,
    #                             silent=True)

    # slightly larger loss, but better visual
    ik = JacobianInverseKinematics(anim, targetmap, iterations=30, damping=5, recalculate=False, silent=True)

    anim = ik()
    return anim



def compute_adaptive_thresholds(glb, fid_l=(3, 4), fid_r=(7, 8), percentile=15):
    """
    Compute adaptive velocity thresholds based on motion statistics.

    Args:
        glb: Joint positions (T, J, 3)
        fid_l: Left foot joint indices (ankle, toe)
        fid_r: Right foot joint indices (ankle, toe)
        percentile: Percentile for threshold (lower = more strict)

    Returns:
        Adaptive velocity thresholds for [ankle, toe]
    """
    velocities = np.sqrt(np.sum((glb[1:] - glb[:-1]) ** 2, axis=-1))

    # Get foot velocities
    ankle_vels = np.concatenate([velocities[:, fid_l[0]], velocities[:, fid_r[0]]])
    toe_vels = np.concatenate([velocities[:, fid_l[1]], velocities[:, fid_r[1]]])

    # Use percentile-based thresholds
    ankle_threshold = np.percentile(ankle_vels, percentile)
    toe_threshold = np.percentile(toe_vels, percentile)

    # Clamp to reasonable ranges to avoid extreme values
    ankle_threshold = np.clip(ankle_threshold, 0.02, 0.15)
    toe_threshold = np.clip(toe_threshold, 0.05, 0.30)

    return np.array([ankle_threshold, toe_threshold])


def remove_fs(glb, foot_contact, fid_l=(3, 4), fid_r=(7, 8), interp_length=5,
              force_on_floor=True, adaptive_thresholds=True, velocity_percentile=15):
    """
    Remove foot sliding artifacts from motion data.

    Args:
        glb: Global joint positions (T, J, 3)
        foot_contact: Optional binary foot contact labels (4, T)
        fid_l: Left foot joint indices (ankle, toe)
        fid_r: Right foot joint indices (ankle, toe)
        interp_length: Interpolation window size for smoothing
        force_on_floor: If True, force contacted feet to y=0
        adaptive_thresholds: If True, compute thresholds from motion statistics
        velocity_percentile: Percentile for adaptive threshold computation

    Returns:
        Joint positions with reduced foot sliding
    """
    scale = 1.0
    height_thres = [0.06, 0.03]  # [ankle, toe] meter - default values

    if foot_contact is None:
        def foot_detect(positions, velfactor, heightfactor):
            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l_h = positions[:-1, fid_l, 1]
            feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float64)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r_h = positions[:-1, fid_r, 1]

            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float64)

            return feet_l, feet_r

        # Adaptive or fixed thresholds
        if adaptive_thresholds:
            feet_vel_thre = compute_adaptive_thresholds(glb, fid_l, fid_r, percentile=velocity_percentile)
        else:
            feet_vel_thre = np.array([0.05, 0.2])

        feet_h_thre = np.array(height_thres) * scale
        feet_l, feet_r = foot_detect(glb, velfactor=feet_vel_thre, heightfactor=feet_h_thre)
        foot = np.concatenate([feet_l, feet_r], axis=-1).transpose(1, 0)  # [4, T-1]
        foot = np.concatenate([foot, foot[:, -1:]], axis=-1)
    else:
        foot = foot_contact.transpose(1, 0)

    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
    # print(foot_heights)
    # floor_height = softmin(foot_heights, softness=0.03, axis=0)
    sort_height = np.sort(foot_heights)
    temp_len = len(sort_height)
    floor_height = np.mean(sort_height[int(0.25*temp_len):int(0.5*temp_len)])
    if floor_height > 0.5: # for motion like swim
        floor_height = 0
    # print(floor_height)
    # floor_height = foot_heights.min()
    # print(floor_height)
    # print(foot)
    # print(foot_heights.min())
    # print(floor_height)
    glb[:, :, 1] -= floor_height
    # anim.positions[:, 0, 1] -= floor_height
    for i, fidx in enumerate(fid):
        fixed = foot[i]  # [T]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    # Note: IK optimization was previously disabled here (commented out).
    # The foot-fixed positions should be returned as-is, and IK should be
    # applied separately by the caller (e.g., in joints2bvh.py) after
    # creating an animation object with proper skeleton structure.
    # This maintains backward compatibility while still fixing foot sliding.
    return glb


def compute_foot_sliding(foot_data, traj_qpos, offseth):
    foot = np.array(foot_data).copy()
    offseth = np.mean(foot[:10, 1])
    foot[:, 1] -= offseth  # Grounding it
    foot_disp = np.linalg.norm(foot[1:, [0, 2]] - foot[:-1, [0, 2]], axis=1)
    traj_qpos[:, 1] -= offseth
    seq_len = len(traj_qpos)
    H = 0.05
    y_threshold = 0.65  # yup system
    y = traj_qpos[1:, 1]

    foot_avg = (foot[:-1, 1] + foot[1:, 1]) / 2
    subset = np.logical_and(foot_avg < H, y > y_threshold)
    # import pdb; pdb.set_trace()

    sliding_stats = np.abs(foot_disp * (2 - 2 ** (foot_avg / H)))[subset]
    sliding = np.sum(sliding_stats) / seq_len * 1000
    return sliding, sliding_stats