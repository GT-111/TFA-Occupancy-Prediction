import einops
import torch
import math
from typing import Tuple
from torchmetrics import MeanMetric
from evaluation.metrics.metrics_base import Metrics

class TrajectoryMetrics(Metrics):

    def __init__(self, device):

        self.min_mse = MeanMetric().to(device)
        self.min_ade = MeanMetric().to(device)
        self.min_fde = MeanMetric().to(device)


    def compute(self, predicted_trajectories, groundtruth_trajectories, valid_mask):
        """
        Compute the metrics for the predicted trajectories
        :param predicted_trajectories: shape [batch_size, num_modes, sequence_length, 2]
        :param groundtruth_trajectories: shape [batch_size, sequence_length, 2]
        :param valid_mask: shape [batch_size, sequence_length]
        :return:
        """
        # mse = self.compute_mse(predicted_trajectories, groundtruth_trajectories, valid_mask)
        # max_dist = self.compute_max_dist(predicted_trajectories, groundtruth_trajectories, valid_mask)
        # min_mse = self.compute_min_mse(predicted_trajectories, groundtruth_trajectories, valid_mask)
        min_ade = self.compute_min_ade(predicted_trajectories, groundtruth_trajectories, valid_mask)
        min_fed = self.compute_min_fde(predicted_trajectories, groundtruth_trajectories, valid_mask)
        metrics_dict = {
            'trajectories_min_ade': min_ade,
            'trajectories_min_fde': min_fed,
        }

        return metrics_dict

    def update(self, metrics_dict):

        self.min_ade.update(metrics_dict['trajectories_min_ade'])
        self.min_fde.update(metrics_dict['trajectories_min_fde'])

    def reset(self):

        self.min_ade.reset()
        self.min_fde.reset()

    def get_result(self):

        res_dict = {}
        res_dict['trajectories_min_ade'] = self.min_ade.compute()
        res_dict['trajectories_min_fde'] = self.min_fde.compute()
        self.reset()

        return res_dict



    def compute_mse(self, traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Computes MSE for a set of trajectories with respect to ground truth.

        :param traj: predictions, shape [batch_size, num_agents, num_modes, sequence_length, 2]
        :param traj_gt: ground truth trajectory, shape [batch_size, num_agents, sequence_length, 2]
        :param masks: masks for varying length ground truth, shape [batch_size, num_agents, sequence_length]
        :return: errs: errors, shape [batch_size, num_agents, num_modes]
        """

        num_modes = traj.shape[2]

        traj_gt_rpt = traj_gt.unsqueeze(2).repeat(1, 1, num_modes, 1, 1)
        masks_rpt = masks.unsqueeze(2).repeat(1, 1, num_modes, 1)

        err = traj_gt_rpt - traj[:, :, :, :, 0:2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=4)  # Sum over x, y dimensions
        err = torch.sum(err * (1 - masks_rpt), dim=3) / torch.sum((1 - masks_rpt), dim=3)

        return err  # Shape: [batch_size, num_agents, num_modes]


    def compute_max_dist(self, traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Computes max distance of a set of trajectories with respect to ground truth trajectory.

        :param traj: predictions, shape [batch_size, num_agents, num_modes, sequence_length, 2]
        :param traj_gt: ground truth trajectory, shape [batch_size, num_agents, sequence_length, 2]
        :param masks: masks for varying length ground truth, shape [batch_size, num_agents, sequence_length]
        :return dist: shape [batch_size, num_agents, num_modes]
        """

        num_modes = traj.shape[2]

        traj_gt_rpt = traj_gt.unsqueeze(2).repeat(1, 1, num_modes, 1, 1)
        masks_rpt = masks.unsqueeze(2).repeat(1, 1, num_modes, 1)

        dist = traj_gt_rpt - traj[:, :, :, :, 0:2]
        dist = torch.pow(dist, exponent=2)
        dist = torch.sum(dist, dim=4)  # Sum over x, y dimensions
        dist = torch.pow(dist, exponent=0.5)

        dist[masks_rpt.bool()] = -math.inf  # Mask out invalid distances
        dist, _ = torch.max(dist, dim=3)  # Max over sequence length

        return dist  # Shape: [batch_size, num_agents, num_modes]


    def compute_min_mse(self, traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes MSE for the best trajectory in a set, with respect to ground truth.

        :param traj: predictions, shape [batch_size, num_agents, num_modes, sequence_length, 2]
        :param traj_gt: ground truth trajectory, shape [batch_size, num_agents, sequence_length, 2]
        :param masks: masks for varying length ground truth, shape [batch_size, num_agents, sequence_length]
        :return errs, inds: errors and indices for modes with min error, shape [batch_size, num_agents]
        """

        mse_values = self.compute_mse(traj, traj_gt, masks)  # [batch_size, num_agents, num_modes]
        err, inds = torch.min(mse_values, dim=2)  # Get min MSE along num_modes

        return err, inds  # Shape: [batch_size, num_agents]

    # // [ ] Fix the following function, get NaN values
    def compute_min_ade(self, traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Computes average displacement error for the best trajectory in a set, with respect to ground truth.

        :param traj: predictions, shape [batch_size, num_agents, num_modes, sequence_length, 2]
        :param traj_gt: ground truth trajectory, shape [batch_size, num_agents, sequence_length, 2]
        :param masks: masks for varying length ground truth, shape [batch_size, num_agents, sequence_length]
        :return errs, inds: errors and indices for modes with min ADE, shape [batch_size, num_agents]
        """
        
        num_modes = traj.shape[2]

        traj_gt_rpt = traj_gt.unsqueeze(2).repeat(1, 1, num_modes, 1, 1)
        masks_rpt = masks.unsqueeze(2).repeat(1, 1, num_modes, 1)

        err = traj_gt_rpt - traj[:, :, :, :, 0:2]
        err = torch.norm(err, dim=4)  # Compute Euclidean distance per timestep

        # Correct mask usage
        err = torch.sum(err * masks_rpt, dim=3) / torch.sum(masks_rpt, dim=3)

        err, _ = torch.min(err, dim=2)  # Min ADE over num_modes

        return err.mean()  # Compute final mean ADE


    def compute_min_fde(self, traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Computes final displacement error (FDE) for the best trajectory in a set, with respect to ground truth.

        :param traj: predictions, shape [batch_size, num_agents, num_modes, sequence_length, 2]
        :param traj_gt: ground truth trajectory, shape [batch_size, num_agents, sequence_length, 2]
        :param masks: masks for varying length ground truth, shape [batch_size, num_agents, sequence_length]
        :return errs, inds: errors and indices for modes with min final displacement error, shape [batch_size, num_agents]
        """

        batch_size, num_agents, num_modes, seq_len, _ = traj.shape

        # Find last valid index per agent
        last_valid_idx = (torch.sum(masks, dim=2) - 1).clamp(min=0, max=seq_len - 1).to(dtype=torch.int64)

        # Gather final predicted positions correctly
        last_valid_idx_expanded = last_valid_idx[:, :, None, None, None].expand(-1, -1, num_modes, -1, 2)
        final_preds = traj.gather(dim=3, index=last_valid_idx_expanded).squeeze(3)

        # Gather final ground truth positions
        final_gt = traj_gt.gather(dim=2, index=last_valid_idx[:, :, None, None].expand(-1, -1, -1, 2)).squeeze(2)

        # Compute Euclidean distance
        err = torch.norm(final_preds - final_gt.unsqueeze(2), dim=3)  # Shape: [batch_size, num_agents, num_modes]

        # Get min FDE across num_modes
        err, _ = torch.min(err, dim=2)

        return err.mean()  # Compute final mean FDE