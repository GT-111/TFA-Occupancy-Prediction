import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric
from evaluation.losses.losses_base import Loss

class TrajectoryLoss(Loss):
    def __init__(self, device, config):
        
        self.device = device
        self.trajectory_regression_loss = MeanMetric().to(self.device)
        self.trajectory_classification_loss = MeanMetric().to(self.device)

    def update(self, loss_dict):
        """
        Update the loss metrics with the provided loss dictionary.

        Args:
            loss_dict: Dictionary containing loss values.
        """
        self.trajectory_regression_loss.update(loss_dict['trajectory_regression_loss'])
        self.trajectory_classification_loss.update(loss_dict['trajectory_classification_loss'])
    
    def get_result(self):
        """
        Get the computed loss metrics.

        Returns:
            Dictionary containing the computed loss metrics.
        """
        result_dict = {
            'trajectory_regression_loss': self.trajectory_regression_loss.compute(),
            'trajectory_classification_loss': self.trajectory_classification_loss.compute()
        }
        
        # Reset the metrics after computation
        self.trajectory_regression_loss.reset()
        self.trajectory_classification_loss.reset()
        
        return result_dict
    
    def compute(self, pred_trajectories: torch.Tensor,
                 pred_scores: torch.Tensor,
                 gt_trajectories: torch.Tensor,
                 gt_mask: torch.Tensor):
        """
        Compute loss for trajectory prediction.

        Args:
            pred_trajectories: Tensor of shape [B, A, M, T, 2], where B=batch size,
                               A=number of agents, M=number of prediction modes,
                               T=time stamps, 2=(x,y).
            pred_scores: Tensor of shape [B, A, M] with predicted scores for each mode.
            gt_trajectories: Tensor of shape [B, A, T, 2] with ground truth trajectories.
            gt_mask: Tensor of shape [B, A] indicating valid agents (1 for valid, 0 for invalid).

        Returns:
            loss_dict: Dictionary containing total loss, regression loss, and classification loss.
        """
        B, A, M, T, _ = pred_trajectories.shape

        # Expand ground truth trajectories to allow broadcasting against modes.
        gt_traj_exp = gt_trajectories.unsqueeze(2)
        
        # Compute L2 error over the (x,y) dimensions, resulting in shape [B, A, M, T]
        error = torch.norm(pred_trajectories - gt_traj_exp, dim=-1)
        
        # Average error per mode over the time dimension: [B, A, M]
        avg_error = error.mean(dim=-1)
        
        # Find the best mode (i.e., mode with the smallest average displacement error) for each agent.
        best_mode_idx = torch.argmin(avg_error, dim=-1)
        
        # Gather the best predicted trajectory for each agent.
        best_mode_idx_expanded = best_mode_idx.unsqueeze(-1).unsqueeze(-1).expand(B, A, 1, T, 2)
        pred_best = torch.gather(pred_trajectories, dim=2, index=best_mode_idx_expanded).squeeze(2)
        
        # -------------------- Regression Loss --------------------
        reg_loss_per_agent = F.smooth_l1_loss(pred_best, gt_trajectories, reduction='none')
        reg_loss_per_agent = reg_loss_per_agent.mean(dim=[2, 3])
        reg_loss = (reg_loss_per_agent * gt_mask).sum() / (gt_mask.sum() + 1e-6)
        
        # ------------------ Classification Loss ------------------
        pred_scores_flat = pred_scores.view(-1, M)
        best_mode_labels = best_mode_idx.view(-1)
        gt_mask_flat = gt_mask.view(-1)
        cls_loss_per_agent = F.cross_entropy(pred_scores_flat, best_mode_labels, reduction='none')
        cls_loss = (cls_loss_per_agent * gt_mask_flat).sum() / (gt_mask_flat.sum() + 1e-6)
        
        # Combine losses into a dictionary
        loss_dict = {
            'trajectory_regression_loss': reg_loss,
            'trajectory_classification_loss': cls_loss
        }
        
        return loss_dict

    def reset(self):
        """
        Reset the loss metrics.
        """
        self.trajectory_regression_loss.reset()
        self.trajectory_classification_loss.reset()