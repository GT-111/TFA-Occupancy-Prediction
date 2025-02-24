import torch
import torch.nn.functional as F

def trajectory_loss(pred_trajectories: torch.Tensor,
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
        total_loss: Scalar tensor combining regression and classification losses.
        reg_loss: Regression loss component (Huber loss).
        cls_loss: Classification loss component (cross-entropy loss).
    """
    B, A, M, T, _ = pred_trajectories.shape

    # Expand ground truth trajectories to allow broadcasting against modes.
    # gt_traj_exp: [B, A, 1, T, 2]
    gt_traj_exp = gt_trajectories.unsqueeze(2)
    
    # Compute L2 error over the (x,y) dimensions, resulting in shape [B, A, M, T]
    error = torch.norm(pred_trajectories - gt_traj_exp, dim=-1)
    
    # Average error per mode over the time dimension: [B, A, M]
    avg_error = error.mean(dim=-1)
    
    # Find the best mode (i.e., mode with the smallest average displacement error) for each agent.
    # best_mode_idx: [B, A]
    best_mode_idx = torch.argmin(avg_error, dim=-1)
    
    # Gather the best predicted trajectory for each agent.
    # We need to index along the mode dimension.
    # First, reshape best_mode_idx so it can be used with torch.gather.
    # best_mode_idx_expanded: [B, A, 1, T, 2]
    best_mode_idx_expanded = best_mode_idx.unsqueeze(-1).unsqueeze(-1).expand(B, A, 1, T, 2)
    # Gather along the mode dimension (dim=2) and squeeze the mode dimension out.
    # pred_best: [B, A, T, 2]
    pred_best = torch.gather(pred_trajectories, dim=2, index=best_mode_idx_expanded).squeeze(2)
    
    # -------------------- Regression Loss --------------------
    # Compute the Huber (smooth L1) loss between the best predicted trajectory and ground truth.
    # We set reduction='none' to apply the mask later.
    # reg_loss_per_agent: [B, A, T, 2]
    reg_loss_per_agent = F.smooth_l1_loss(pred_best, gt_trajectories, reduction='none')
    # Average over time and spatial dimensions -> [B, A]
    reg_loss_per_agent = reg_loss_per_agent.mean(dim=[2, 3])
    # Apply the ground truth mask (only valid agents contribute to the loss)
    reg_loss = (reg_loss_per_agent * gt_mask).sum() / (gt_mask.sum() + 1e-6)
    
    # ------------------ Classification Loss ------------------
    # Use cross-entropy loss on the predicted scores.
    # Flatten the batch and agent dimensions.
    pred_scores_flat = pred_scores.view(-1, M)      # [B*A, M]
    best_mode_labels = best_mode_idx.view(-1)         # [B*A]
    gt_mask_flat = gt_mask.view(-1)                   # [B*A]
    # Compute the cross-entropy loss per agent (without reduction).
    cls_loss_per_agent = F.cross_entropy(pred_scores_flat, best_mode_labels, reduction='none')
    # Average over valid agents.
    cls_loss = (cls_loss_per_agent * gt_mask_flat).sum() / (gt_mask_flat.sum() + 1e-6)
    
    # Sum both losses.
    total_loss = reg_loss + cls_loss
    
    return total_loss, reg_loss, cls_loss
