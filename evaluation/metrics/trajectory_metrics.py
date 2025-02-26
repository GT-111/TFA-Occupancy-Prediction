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
        mse = self.compute_mse(predicted_trajectories, groundtruth_trajectories, valid_mask)
        max_dist = self.compute_max_dist(predicted_trajectories, groundtruth_trajectories, valid_mask)
        min_mse = self.compute_min_mse(predicted_trajectories, groundtruth_trajectories, valid_mask)
        min_ade = self.compute_min_ade(predicted_trajectories, groundtruth_trajectories, valid_mask)
        min_fed = self.compute_min_fde(predicted_trajectories, groundtruth_trajectories, valid_mask)
        metrics_dict = {
            'trajectories_min_mse': min_mse,
            'trajectories_min_ade': min_ade,
            'trajectories_min_fde': min_fed,
        }

        return metrics_dict
    
    def update(self, metrics_dict):

        self.min_mse.update(metrics_dict['trajectories_min_mse'])
        self.min_ade.update(metrics_dict['trajectories_min_ade'])
        self.min_fde.update(metrics_dict['trajectories_min_fde'])

    def reset(self):

        self.min_mse.reset()
        self.min_ade.reset()
        self.min_fde.reset()

    def get_result(self):

        res_dict = {}
        res_dict['trajectories_min_mse'] = self.min_mse.compute()
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

        mse_values = self.mse(traj, traj_gt, masks)  # [batch_size, num_agents, num_modes]
        err, inds = torch.min(mse_values, dim=2)  # Get min MSE along num_modes

        return err, inds  # Shape: [batch_size, num_agents]


    def compute_min_ade(self, traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=4)
        err = torch.pow(err, exponent=0.5)

        err = torch.sum(err * (1 - masks_rpt), dim=3) / torch.sum((1 - masks_rpt), dim=3)
        err, inds = torch.min(err, dim=2)  # Get min ADE along num_modes

        return err, inds  # Shape: [batch_size, num_agents]


    def compute_min_fde(self, traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes final displacement error (FDE) for the best trajectory in a set, with respect to ground truth.

        :param traj: predictions, shape [batch_size, num_agents, num_modes, sequence_length, 2]
        :param traj_gt: ground truth trajectory, shape [batch_size, num_agents, sequence_length, 2]
        :param masks: masks for varying length ground truth, shape [batch_size, num_agents, sequence_length]
        :return errs, inds: errors and indices for modes with min final displacement error, shape [batch_size, num_agents]
        """

        num_modes = traj.shape[2]
        batch_size, num_agents, _, seq_len, _ = traj.shape

        # Find last valid index for each agent
        last_valid_idx = torch.sum(1 - masks, dim=2) - 1  # Shape: [batch_size, num_agents]

        # Gather final predicted positions across all modes
        final_preds = traj[
            torch.arange(batch_size).unsqueeze(1).unsqueeze(2),  # Batch index
            torch.arange(num_agents).unsqueeze(0).unsqueeze(2),  # Agent index
            torch.arange(num_modes).unsqueeze(0).unsqueeze(1),  # Mode index
            last_valid_idx.unsqueeze(2),  # Last valid timestep index
            :2  # Only take x, y
        ]  # Shape: [batch_size, num_agents, num_modes, 2]

        # Gather final ground truth positions
        final_gt = traj_gt[torch.arange(batch_size).unsqueeze(1), torch.arange(num_agents).unsqueeze(0), last_valid_idx, :2]  # Shape: [batch_size, num_agents, 2]

        # Compute Euclidean distance
        err = torch.norm(final_preds - final_gt.unsqueeze(2), dim=3)  # Shape: [batch_size, num_agents, num_modes]

        # Get minimum FDE and corresponding mode index
        err, inds = torch.min(err, dim=2)  # Shape: [batch_size, num_agents]

        return err, inds
