import torch
import math
from typing import Tuple
# TODO : modify the metrics to fit into the metrics_base format
def mse(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Computes MSE for a set of trajectories with respect to ground truth.

    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return: errs: errors, shape [batch_size, num_modes]
    """

    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / torch.sum((1 - masks_rpt), dim=2)
    return err


def max_dist(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Computes max distance of a set of trajectories with respect to ground truth trajectory.

    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return dist: shape [batch_size, num_modes]
    """
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    dist = traj_gt_rpt - traj[:, :, :, 0:2]
    dist = torch.pow(dist, exponent=2)
    dist = torch.sum(dist, dim=3)
    dist = torch.pow(dist, exponent=0.5)
    dist[masks_rpt.bool()] = -math.inf
    dist, _ = torch.max(dist, dim=2)

    return dist


def min_mse(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes MSE for the best trajectory is a set, with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape [batch_size]
    """

    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / torch.sum((1 - masks_rpt), dim=2)
    err, inds = torch.min(err, dim=1)

    return err, inds


def min_ade(traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes average displacement error for the best trajectory is a set, with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape [batch_size]
    """
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.pow(err, exponent=0.5)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / torch.sum((1 - masks_rpt), dim=2)
    err, inds = torch.min(err, dim=1)

    return err, inds