# credit: https://github.com/cliport/cliport

"""Miscellaneous utilities."""

import cv2
import random
import matplotlib
import matplotlib.pyplot as plt
import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf

import yaml
import numpy as np
from transforms3d import euler

import hydra
from hydra.experimental import compose, initialize

import pybullet as p
import kornia
from omegaconf import OmegaConf

import os
import torch
import json

from pathlib import Path
from numpy.random import normal
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------
# HEIGHTMAP UTILS
# -----------------------------------------------------------------------------


def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.

    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.

    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = np.array(config["intrinsics"]).reshape(3, 3)
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)
    return heightmaps, colormaps


def pix_to_xyz(pixel, height, bounds, pixel_size, skip_height=False):
    """Convert from pixel location on heightmap to 3D position."""
    u, v = pixel
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 0] + u * pixel_size
    if not skip_height:
        z = bounds[2, 0] + height[u, v]
    else:
        z = 0.0
    return (x, y, z)


def xyz_to_pix(position, bounds, pixel_size):
    """Convert from 3D position to pixel location on heightmap."""
    u = int(np.round((position[1] - bounds[1, 0]) / pixel_size))
    v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
    return (u, v)


def unproject_vectorized(uv_coordinates, depth_values, intrinsic, distortion):
    """Vectorized version of unproject(), for N points.

    Args:
      uv_coordinates: pixel coordinates to unproject of shape (n, 2).
      depth_values: depth values corresponding index-wise to the uv_coordinates of
        shape (n).
      intrinsic: array of shape (3, 3). This is typically the return value of
        intrinsics_to_matrix.
      distortion: camera distortion parameters of shape (5,).

    Returns:
      xyz coordinates in camera frame of shape (n, 3).
    """
    cam_mtx = intrinsic  # shape [3, 3]
    cam_dist = np.array(distortion)  # shape [5]

    # shape of points_undistorted is [N, 2] after the squeeze().
    points_undistorted = cv2.undistortPoints(uv_coordinates.reshape((-1, 1, 2)), cam_mtx, cam_dist).squeeze()

    x = points_undistorted[:, 0] * depth_values
    y = points_undistorted[:, 1] * depth_values

    xyz = np.vstack((x, y, depth_values)).T
    return xyz


def unproject_depth_vectorized(im_depth, depth_dist, camera_mtx, camera_dist):
    """Unproject depth image into 3D point cloud, using calibration.

    Args:
      im_depth: raw depth image, pre-calibration of shape (height, width).
      depth_dist: depth distortion parameters of shape (8,)
      camera_mtx: intrinsics matrix of shape (3, 3). This is typically the return
        value of intrinsics_to_matrix.
      camera_dist: camera distortion parameters shape (5,).

    Returns:
      numpy array of shape [3, H*W]. each column is xyz coordinates
    """
    h, w = im_depth.shape

    # shape of each u_map, v_map is [H, W].
    u_map, v_map = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))

    adjusted_depth = depth_dist[0] + im_depth * depth_dist[1]

    # shape after stack is [N, 2], where N = H * W.
    uv_coordinates = np.stack((u_map.reshape(-1), v_map.reshape(-1)), axis=-1)

    return unproject_vectorized(uv_coordinates, adjusted_depth.reshape(-1), camera_mtx, camera_dist)


# -----------------------------------------------------------------------------
# MATH UTILS
# -----------------------------------------------------------------------------


def sample_distribution(prob, n_samples=1):
    """Sample data point from a custom distribution."""
    flat_prob = prob.flatten() / np.sum(prob)
    rand_ind = np.random.choice(np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False)
    rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
    return np.int32(rand_ind_coords.squeeze())


# -------------------------------------------------------------------------
# Transformation Helper Functions
# -------------------------------------------------------------------------


def invert(pose):
    return p.invertTransform(pose[0], pose[1])


def multiply(pose0, pose1):
    return p.multiplyTransforms(pose0[0], pose0[1], pose1[0], pose1[1])


def apply(pose, position):
    position = np.float32(position)
    position_shape = position.shape
    position = np.float32(position).reshape(3, -1)
    rotation = np.float32(p.getMatrixFromQuaternion(pose[1])).reshape(3, 3)
    translation = np.float32(pose[0]).reshape(3, 1)
    position = rotation @ position + translation
    return tuple(position.reshape(position_shape))


def is_in_container(container_pose, container_size, object_pose):
    """Check if object is within container."""
    container_size = np.float32(container_size).reshape(3, 1)
    rotation = np.float32(p.getMatrixFromQuaternion(container_pose[1])).reshape(3, 3)
    translation = np.float32(object_pose[0]).reshape(3, 1) - np.float32(container_pose[0]).reshape(3, 1)
    relative_pos = rotation.transpose() @ translation
    return np.all(np.abs(relative_pos[:2]) < np.asarray(container_size[:2]) / 2)


def eulerXYZ_to_quatXYZW(rotation):  # pylint: disable=invalid-name
    """Abstraction for converting from a 3-parameter rotation to quaterion.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      rotation: a 3-parameter rotation, in xyz order tuple of 3 floats

    Returns:
      quaternion, in xyzw order, tuple of 4 floats
    """
    euler_zxy = (rotation[2], rotation[0], rotation[1])
    quaternion_wxyz = euler.euler2quat(*euler_zxy, axes="szxy")
    q = quaternion_wxyz
    quaternion_xyzw = (q[1], q[2], q[3], q[0])
    return quaternion_xyzw


def quatXYZW_to_eulerXYZ(quaternion_xyzw):  # pylint: disable=invalid-name
    """Abstraction for converting from quaternion to a 3-parameter toation.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      quaternion_xyzw: in xyzw order, tuple of 4 floats

    Returns:
      rotation: a 3-parameter rotation, in xyz order, tuple of 3 floats
    """
    q = quaternion_xyzw
    quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
    euler_zxy = euler.quat2euler(quaternion_wxyz, axes="szxy")
    euler_xyz = (euler_zxy[1], euler_zxy[2], euler_zxy[0])
    return euler_xyz


def apply_transform(transform_to_from, points_from):
    r"""Transforms points (3D) into new frame.

    Using transform_to_from notation.

    Args:
      transform_to_from: numpy.ndarray of shape [B,4,4], SE3
      points_from: numpy.ndarray of shape [B,3,N]

    Returns:
      points_to: numpy.ndarray of shape [B,3,N]
    """
    num_points = points_from.shape[-1]

    # non-batched
    if len(transform_to_from.shape) == 2:
        ones = np.ones((1, num_points))

        # makes these each into homogenous vectors
        points_from = np.vstack((points_from, ones))  # [4,N]
        points_to = transform_to_from @ points_from  # [4,N]
        return points_to[0:3, :]  # [3,N]

    # batched
    else:
        assert len(transform_to_from.shape) == 3
        batch_size = transform_to_from.shape[0]
        zeros = np.ones((batch_size, 1, num_points))
        points_from = np.concatenate((points_from, zeros), axis=1)
        assert points_from.shape[1] == 4
        points_to = transform_to_from @ points_from
        return points_to[:, 0:3, :]


# -----------------------------------------------------------------------------
# IMAGE UTILS
# -----------------------------------------------------------------------------


def preprocess(img, dist="transporter"):
    """Pre-process input (subtract mean, divide by std)."""

    transporter_color_mean = [0.18877631, 0.18877631, 0.18877631]
    transporter_color_std = [0.07276466, 0.07276466, 0.07276466]
    transporter_depth_mean = 0.00509261
    transporter_depth_std = 0.00903967

    franka_color_mean = [0.622291933, 0.628313992, 0.623031488]
    franka_color_std = [0.168154213, 0.17626014, 0.184527364]
    franka_depth_mean = 0.872146842
    franka_depth_std = 0.195743116

    clip_color_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_color_std = [0.26862954, 0.26130258, 0.27577711]

    # choose distribution
    if dist == "clip":
        color_mean = clip_color_mean
        color_std = clip_color_std
    elif dist == "franka":
        color_mean = franka_color_mean
        color_std = franka_color_std
    else:
        color_mean = transporter_color_mean
        color_std = transporter_color_std

    if dist == "franka":
        depth_mean = franka_depth_mean
        depth_std = franka_depth_std
    else:
        depth_mean = transporter_depth_mean
        depth_std = transporter_depth_std

    # convert to pytorch tensor (if required)
    if type(img) == torch.Tensor:

        def cast_shape(stat, img):
            tensor = torch.from_numpy(np.array(stat)).to(device=img.device, dtype=img.dtype)
            tensor = tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            tensor = tensor.repeat(img.shape[0], 1, img.shape[-2], img.shape[-1])
            return tensor

        color_mean = cast_shape(color_mean, img)
        color_std = cast_shape(color_std, img)
        depth_mean = cast_shape(depth_mean, img)
        depth_std = cast_shape(depth_std, img)

        # normalize
        img = img.clone()
        img[:, :3, :, :] = (img[:, :3, :, :] / 255 - color_mean) / color_std
        img[:, 3:, :, :] = (img[:, 3:, :, :] - depth_mean) / depth_std
    else:
        # normalize
        img[:, :, :3] = (img[:, :, :3] / 255 - color_mean) / color_std
        img[:, :, 3:] = (img[:, :, 3:] - depth_mean) / depth_std

    # if dist == 'franka' or dist == 'transporter':
    #     print(np.mean(img[:,:3,:,:].detach().cpu().numpy(), axis=(0,2,3)),
    #           np.mean(img[:,3,:,:].detach().cpu().numpy()))

    return img


def deprocess(img):
    color_mean = 0.18877631
    depth_mean = 0.00509261
    color_std = 0.07276466
    depth_std = 0.00903967

    img[:, :, :3] = np.uint8(((img[:, :, :3] * color_std) + color_mean) * 255)
    img[:, :, 3:] = np.uint8(((img[:, :, 3:] * depth_std) + depth_mean) * 255)
    return img


def get_fused_heightmap(obs, configs, bounds, pix_size):
    """Reconstruct orthographic heightmaps with segmentation masks."""
    heightmaps, colormaps = reconstruct_heightmaps(obs["color"], obs["depth"], configs, bounds, pix_size)
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    cmap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
    cmap = np.uint8(np.round(cmap))
    hmap = np.max(heightmaps, axis=0)  # Max to handle occlusions.
    return cmap, hmap


def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1.0, 0.0, -pivot[0]], [0.0, 1.0, -pivot[1]], [0.0, 0.0, 1.0]])
    image_t_pivot = np.array([[1.0, 0.0, pivot[0]], [0.0, 1.0, pivot[1]], [0.0, 0.0, 1.0]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]], [np.sin(theta), np.cos(theta), trans[1]], [0.0, 0.0, 1.0]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))


def check_transform(image, pixel, transform):
    """Valid transform only if pixel locations are still in FoV after transform."""
    new_pixel = np.flip(
        np.int32(np.round(np.dot(transform, np.float32([pixel[1], pixel[0], 1.0]).reshape(3, 1))))[:2].squeeze()
    )
    valid = np.all(new_pixel >= 0) and new_pixel[0] < image.shape[0] and new_pixel[1] < image.shape[1]
    return valid, new_pixel


def get_se3_from_image_transform(theta, trans, pivot, heightmap, bounds, pixel_size):
    """Calculate SE3 from image transform."""
    position_center = pix_to_xyz(np.flip(np.int32(np.round(pivot))), heightmap, bounds, pixel_size, skip_height=False)
    new_position_center = pix_to_xyz(
        np.flip(np.int32(np.round(pivot + trans))), heightmap, bounds, pixel_size, skip_height=True
    )
    # Don't look up the z height, it might get augmented out of frame
    new_position_center = (new_position_center[0], new_position_center[1], position_center[2])

    delta_position = np.array(new_position_center) - np.array(position_center)

    t_world_center = np.eye(4)
    t_world_center[0:3, 3] = np.array(position_center)

    t_centernew_center = np.eye(4)
    euler_zxy = (-theta, 0, 0)
    t_centernew_center[0:3, 0:3] = euler.euler2mat(*euler_zxy, axes="szxy")[0:3, 0:3]

    t_centernew_center_tonly = np.eye(4)
    t_centernew_center_tonly[0:3, 3] = -delta_position
    t_centernew_center = t_centernew_center @ t_centernew_center_tonly

    t_world_centernew = t_world_center @ np.linalg.inv(t_centernew_center)
    return t_world_center, t_world_centernew


def get_random_image_transform_params(image_size, theta_sigma=60):
    theta = np.random.normal(0, np.deg2rad(theta_sigma))

    trans_sigma = np.min(image_size) / 6
    trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return (w, x, y, z)


def perturb(input_image, pixels, theta_sigma=60, add_noise=False, action_noise=None):
    """Data augmentation on images."""
    image_size = input_image.shape[:2]

    # Compute random rigid transform.
    while True:
        theta, trans, pivot = get_random_image_transform_params(image_size, theta_sigma=theta_sigma)
        transform = get_image_transform(theta, trans, pivot)
        transform_params = theta, trans, pivot

        # Ensure pixels remain in the image after transform.
        is_valid = True
        new_pixels = []
        new_rounded_pixels = []
        for pixel in pixels:
            pixel = np.float32([pixel[1], pixel[0], 1.0]).reshape(3, 1)

            if action_noise is not None:
                noise = normal(scale=action_noise, size=2).reshape(2, 1)

                if np.all(np.abs(noise) < 3):
                    pixel[:2] += noise

            rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
            rounded_pixel = np.flip(rounded_pixel)

            pixel = (transform @ pixel)[:2].squeeze()
            pixel = np.flip(pixel)

            in_fov_rounded = rounded_pixel[0] < image_size[0] and rounded_pixel[1] < image_size[1]
            in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]

            is_valid = is_valid and np.all(rounded_pixel >= 0) and np.all(pixel >= 0) and in_fov_rounded and in_fov

            new_pixels.append(pixel)
            new_rounded_pixels.append(rounded_pixel)
        if is_valid:
            break

    # Apply rigid transform to image and pixel labels.
    input_image = cv2.warpAffine(input_image, transform[:2, :], (image_size[1], image_size[0]), flags=cv2.INTER_LINEAR)

    # Apply noise
    color = np.int32(input_image[:, :, :3])
    depth = np.float32(input_image[:, :, 3:])

    if add_noise:
        color += np.int32(np.random.normal(0, 3, image_size + (3,)))
        color = np.uint8(np.clip(color, 0, 255))

        depth += np.float32(np.random.normal(0, 0.003, image_size + (3,)))

    input_image = np.concatenate((color, depth), axis=2)

    return input_image, new_pixels, new_rounded_pixels, transform_params


def apply_perturbation(input_image, transform_params):
    """Apply data augmentation with specific transform params"""
    image_size = input_image.shape[:2]

    # Apply rigid transform to image and pixel labels.
    theta, trans, pivot = transform_params
    transform = get_image_transform(theta, trans, pivot)

    input_image = cv2.warpAffine(input_image, transform[:2, :], (image_size[1], image_size[0]), flags=cv2.INTER_LINEAR)
    return input_image


class ImageRotator:
    """Rotate for n rotations."""

    # Reference: https://kornia.readthedocs.io/en/latest/tutorials/warp_affine.html?highlight=rotate

    def __init__(self, n_rotations):
        self.angles = []
        for i in range(n_rotations):
            theta = i * 2 * 180 / n_rotations
            self.angles.append(theta)

    def __call__(self, x_list, pivot_list, reverse=False):
        assert len(x_list) == len(self.angles)
        B = x_list[0].shape[0]
        assert all([x.shape[0] == B for x in x_list])
        if pivot_list[0].dim() == 1:
            pivot_list = [pivot.repeat(B, 1) for pivot in pivot_list]
        else:
            assert all([pivot.shape[0] == B for pivot in pivot_list])
        rot_x_list = []
        for i, angle in enumerate(self.angles):
            # x = x_list[i].unsqueeze(0)
            x = x_list[i]  # B, C, H, W
            pivot = pivot_list[i]  # B, 2

            # create transformation (rotation)
            alpha: float = angle if not reverse else (-1.0 * angle)  # in degrees
            # angle: torch.tensor = torch.ones(1) * alpha
            angle: torch.tensor = torch.ones(B).to(x.device) * alpha

            # define the rotation center
            # center: torch.tensor = torch.ones(1, 2)
            # center[..., 0] = pivot[1]
            # center[..., 1] = pivot[0]
            center = torch.stack([pivot[:, 1], pivot[:, 0]], dim=1).float()  # B, 2

            # define the scale factor
            # scale: torch.tensor = torch.ones(1, 2)
            scale: torch.tensor = torch.ones(B, 2).to(x.device)

            # compute the transformation matrix
            M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale)

            # apply the transformation to original image
            _, _, h, w = x.shape
            x_warped: torch.tensor = kornia.warp_affine(x.float(), M.to(x.device), dsize=(h, w))
            x_warped = x_warped
            rot_x_list.append(x_warped)

        return rot_x_list


# -----------------------------------------------------------------------------
# COLOR AND PLOT UTILS
# -----------------------------------------------------------------------------

# Colors (Tableau palette).
COLORS = {
    "blue": [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
    "red": [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0],
    "green": [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0],
    "orange": [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0],
    "yellow": [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0],
    "purple": [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
    "pink": [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
    "cyan": [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
    "brown": [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0],
    "white": [255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0],
    "gray": [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0],
}

TRAIN_COLORS = ["blue", "red", "green", "yellow", "brown", "gray", "cyan"]
EVAL_COLORS = ["blue", "red", "green", "orange", "purple", "pink", "white"]


TRAIN_GOOGLE_OBJECTS = [
    "alarm clock",
    "android toy",
    "black boot with leopard print",
    "black fedora",
    "black razer mouse",
    "black sandal",
    "black shoe with orange stripes",
    "bull figure",
    "butterfinger chocolate",
    "c clamp",
    "can opener",
    "crayon box",
    "dog statue",
    "frypan",
    "green and white striped towel",
    "grey soccer shoe with cleats",
    "hard drive",
    "honey dipper",
    "magnifying glass",
    "mario figure",
    "nintendo 3ds",
    "nintendo cartridge",
    "office depot box",
    "orca plush toy",
    "pepsi gold caffeine free box",
    "pepsi wild cherry box",
    "porcelain cup",
    "purple tape",
    "red and white flashlight",
    "rhino figure",
    "rocket racoon figure",
    "scissors",
    "silver tape",
    "spatula with purple head",
    "spiderman figure",
    "tablet",
    "toy school bus",
]
VAL_GOOGLE_OBJECTS = [
    "ball puzzle",
    "black and blue sneakers",
    "black shoe with green stripes",
    "brown fedora",
    "dinosaur figure",
    "hammer",
    "light brown boot with golden laces",
    "lion figure",
    "pepsi max box",
    "pepsi next box",
    "porcelain salad plate",
    "porcelain spoon",
    "red and white striped towel",
    "red cup",
    "screwdriver",
    "toy train",
    "unicorn toy",
    "white razer mouse",
    "yoshi figure",
]
TEST_GOOGLE_OBJECTS = [
    "ball puzzle",
    "black and blue sneakers",
    "black shoe with green stripes",
    "brown fedora",
    "dinosaur figure",
    "hammer",
    "light brown boot with golden laces",
    "lion figure",
    "pepsi max box",
    "pepsi next box",
    "porcelain salad plate",
    "porcelain spoon",
    "red and white striped towel",
    "red cup",
    "screwdriver",
    "toy train",
    "unicorn toy",
    "white razer mouse",
    "yoshi figure",
]
GOOGLE_OBJECTS = list(set(TRAIN_GOOGLE_OBJECTS) | set(VAL_GOOGLE_OBJECTS) | set(TEST_GOOGLE_OBJECTS))


def get_color_name(color):
    for name, rgb in COLORS.items():
        if rgb == color:
            return name


def plot(
    fname,  # pylint: disable=dangerous-default-value
    title,
    ylabel,
    xlabel,
    data,
    xlim=[-np.inf, 0],
    xticks=None,
    ylim=[np.inf, -np.inf],
    show_std=True,
):
    """Plot frame data."""
    # Data is a dictionary that maps experiment names to tuples with 3
    # elements: x (size N array) and y (size N array) and y_std (size N array)

    # Get data limits.
    for name, (x, y, _) in data.items():
        del name
        y = np.array(y)
        xlim[0] = max(xlim[0], np.min(x))
        xlim[1] = max(xlim[1], np.max(x))
        ylim[0] = min(ylim[0], np.min(y))
        ylim[1] = max(ylim[1], np.max(y))

    # Draw background.
    plt.title(title, fontsize=14)
    plt.ylim(ylim)
    plt.ylabel(ylabel, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xlim)
    plt.xlabel(xlabel, fontsize=14)
    plt.grid(True, linestyle="-", color=[0.8, 0.8, 0.8])
    ax = plt.gca()
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_color("#000000")
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["mathtext.default"] = "regular"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    # Draw data.
    color_iter = 0
    for name, (x, y, std) in data.items():
        del name
        x, y, std = np.float32(x), np.float32(y), np.float32(std)
        upper = np.clip(y + std, ylim[0], ylim[1])
        lower = np.clip(y - std, ylim[0], ylim[1])
        color = COLORS[list(COLORS.keys())[color_iter]]
        if show_std:
            plt.fill_between(x, upper, lower, color=color, linewidth=0, alpha=0.3)
        plt.plot(x, y, color=color, linewidth=2, marker="o", alpha=1.0)
        color_iter += 1

    if xticks:
        plt.xticks(ticks=range(len(xticks)), labels=xticks, fontsize=14)
    else:
        plt.xticks(fontsize=14)
    plt.legend([name for name, _ in data.items()], loc="lower right", fontsize=14)
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()


# -----------------------------------------------------------------------------
# MESHCAT UTILS
# -----------------------------------------------------------------------------


def create_visualizer(clear=True):
    print("Waiting for meshcat server... have you started a server?")
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    if clear:
        vis.delete()
    return vis


def make_frame(vis, name, h, radius, o=1.0):
    """Add a red-green-blue triad to the Meschat visualizer.

    Args:
      vis (MeshCat Visualizer): the visualizer
      name (string): name for this frame (should be unique)
      h (float): height of frame visualization
      radius (float): radius of frame visualization
      o (float): opacity
    """
    vis[name]["x"].set_object(
        g.Cylinder(height=h, radius=radius), g.MeshLambertMaterial(color=0xFF0000, reflectivity=0.8, opacity=o)
    )
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]["x"].set_transform(rotate_x)

    vis[name]["y"].set_object(
        g.Cylinder(height=h, radius=radius), g.MeshLambertMaterial(color=0x00FF00, reflectivity=0.8, opacity=o)
    )
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]["y"].set_transform(rotate_y)

    vis[name]["z"].set_object(
        g.Cylinder(height=h, radius=radius), g.MeshLambertMaterial(color=0x0000FF, reflectivity=0.8, opacity=o)
    )
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]["z"].set_transform(rotate_z)


def meshcat_visualize(vis, obs, act, info):
    """Visualize data using meshcat."""

    for key in sorted(info.keys()):
        pose = info[key]
        pick_transform = np.eye(4)
        pick_transform[0:3, 3] = pose[0]
        quaternion_wxyz = np.asarray([pose[1][3], pose[1][0], pose[1][1], pose[1][2]])
        pick_transform[0:3, 0:3] = mtf.quaternion_matrix(quaternion_wxyz)[0:3, 0:3]
        label = "obj_" + str(key)
        make_frame(vis, label, h=0.05, radius=0.0012, o=1.0)
        vis[label].set_transform(pick_transform)

    for cam_index in range(len(act["camera_config"])):
        verts = unproject_depth_vectorized(
            obs["depth"][cam_index],
            np.array([0, 1]),
            np.array(act["camera_config"][cam_index]["intrinsics"]).reshape(3, 3),
            np.zeros(5),
        )

        # switch from [N,3] to [3,N]
        verts = verts.T

        cam_transform = np.eye(4)
        cam_transform[0:3, 3] = act["camera_config"][cam_index]["position"]
        quaternion_xyzw = act["camera_config"][cam_index]["rotation"]
        quaternion_wxyz = np.asarray([quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]])
        cam_transform[0:3, 0:3] = mtf.quaternion_matrix(quaternion_wxyz)[0:3, 0:3]
        verts = apply_transform(cam_transform, verts)

        colors = obs["color"][cam_index].reshape(-1, 3).T / 255.0

        vis["pointclouds/" + str(cam_index)].set_object(g.PointCloud(position=verts, color=colors))


# -----------------------------------------------------------------------------
# CONFIG UTILS
# -----------------------------------------------------------------------------


def set_seed(seed, torch=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch:
        import torch

        torch.manual_seed(seed)


def load_cfg(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def load_hydra_config(config_path):
    return OmegaConf.load(config_path)


# -----------------------------------------------------------------------------
# MISC UTILS
# -----------------------------------------------------------------------------


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def get_mean_of_last_k_percentile(values, k=0.25):
    """Get mean of last k percentile."""
    values = np.array(values)
    k = int(k * len(values))
    k = max([k, 1])
    return np.mean(values[:k])


def get_train_cfg(icfg):
    train_path = Path(icfg["train_config"]).parent.parent
    train_config_path = Path(icfg["train_config"])

    if not train_config_path.exists() and icfg["train_demos"] == 0:
        # If train config doesn't exist and train_demos is 0, train from scratch.
        # We then use the train configuration from current train.yaml with some overrides.
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        os.chdir(os.environ["ASKDAGGER_ROOT"])
        with initialize(config_path="../cfg"):
            tcfg = compose(
                config_name="train",
                overrides=[
                    f"train.iteration={icfg['iteration']}",
                    f"train.drop_prob={icfg['train_interactive']['drop_prob']}",
                    f"train.N={icfg['train_interactive']['N']}",
                    f"train.gpu={icfg['train_interactive']['gpu']}",
                    f"train.agent={icfg['agent']}",
                    f"train.task={icfg['train_interactive_task']}",
                    f"train.lr={icfg['train_interactive']['lr']}",
                    f"train.batch_size={icfg['train_interactive']['batch_size']}",
                    f"train.n_workers={icfg['train_interactive']['n_workers']}",
                ],
            )
        OmegaConf.set_struct(tcfg, False)
    else:
        tcfg = load_hydra_config(icfg["train_config"])
    tcfg["train_interactive"] = icfg["train_interactive"]
    tcfg["train"]["train_dir"] = str(train_path)
    return tcfg


def get_heightmap_dummy(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.

    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint64)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap


def reconstruct_heightmaps_dummy(color, depth, configs, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = np.array(config["intrinsics"]).reshape(3, 3)
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap_dummy(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)
    return heightmaps, colormaps


def get_projection(obs, cam_config, bounds, pix_size, in_shape):
    cmap, hmap = get_fused_heightmap(obs, cam_config, bounds, pix_size)
    img = np.concatenate((cmap, hmap[Ellipsis, None], hmap[Ellipsis, None], hmap[Ellipsis, None]), axis=2)
    assert img.shape == in_shape, img.shape
    return img


def set_plot_style():
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1)

    plt.rcParams.update(
        {
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "font.size": 8,
            "grid.linewidth": 0.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.pad": -2.0,
            "ytick.major.pad": -2.0,
            "lines.linewidth": 1.3,
            "axes.xmargin": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": True,
            "font.family": "Helvetica",
        }
    )


def get_interactive_train_dir(train_dir, vcfg):
    interactive_train_dir = (
        train_dir
        / "interactive_training"
        / f"fier_{vcfg['relabeling_demos']}_pier_{vcfg['eval']['pier']}_sensitivity_{vcfg['eval']['s_desired']}"
    )
    return interactive_train_dir


def list_ckpts_to_eval(vcfg, existing_results, verbose=True):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg["checkpoint_type"] == "last":
        last_ckpt = "last.ckpt"
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg["checkpoint_type"] == "val_missing":
        checkpoints = sorted([c for c in os.listdir(vcfg["model_path"]) if "steps=" in c])
        if len(checkpoints) == 0:
            checkpoints = sorted([c for c in os.listdir(vcfg["model_path"]) if "interactive=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg["checkpoint_type"] == "test_best":
        result_jsons = [c for c in os.listdir(vcfg["results_path"]) if "results-val" in c]
        result_jsons = [r for r in result_jsons]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg["results_path"], result_json), "r") as f:
                eval_res = json.load(f)
            best_checkpoint = "last.ckpt"
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res["mean_reward"] > best_success:
                    best_checkpoint = ckpt
                    best_success = res["mean_reward"]
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = "last.ckpt"
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        if verbose:
            print(f"Looking for: {vcfg['checkpoint_type']}")
        checkpoints = [c for c in os.listdir(vcfg["model_path"]) if vcfg["checkpoint_type"] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    return ckpts_to_eval


def get_train_results(vcfg):
    train_dir = Path(str(vcfg["model_path"]).split("checkpoints")[0])
    interactive_train_dir = get_interactive_train_dir(train_dir, vcfg)
    train_demos = vcfg["train_demos"]
    eval_task = vcfg["eval_task"]
    name = "{}-{}-n{}".format(eval_task, vcfg["agent"], train_demos)
    # Save path for results.
    json_name = "results-train.json"
    save_json = os.path.join(interactive_train_dir, f"{name}-{json_name}")
    if os.path.exists(save_json):
        with open(save_json, "r") as f:
            results = json.load(f)
        results = results[name]
    else:
        print(f"Could not find {save_json}, continuing...")
        results = None
    return results


def get_eval_results(vcfg):
    train_dir = Path(str(vcfg["model_path"]).split("checkpoints")[0])
    interactive_train_dir = get_interactive_train_dir(train_dir, vcfg)
    save_path = Path(interactive_train_dir) / "checkpoints"
    eval_task = vcfg["eval_task"]
    # Choose eval mode and task.
    mode = vcfg["mode"]
    eval_task = vcfg["eval_task"]
    if mode not in {"train", "val", "test"}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    name = "{}-{}-n{}".format(eval_task, vcfg["agent"], vcfg["n_demos"])
    json_name = f"multi-results-{mode}.json" if "multi" in vcfg["model_path"] else f"results-{mode}.json"
    save_json = os.path.join(save_path, f"{name}-{json_name}")
    if os.path.exists(save_json):
        with open(save_json, "r") as f:
            results = json.load(f)
    else:
        print(f"Could not find {save_json}, continuing...")
        results = None
    return results
