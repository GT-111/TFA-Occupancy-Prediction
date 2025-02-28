import torch
from typing import Dict
from functools import partial
import torch.nn.functional as F
from torchmetrics import MeanMetric
from utils.occupancy_flow_map_utils import sample
from torchmetrics.functional.classification import binary_average_precision
from evaluation.losses.losses_base import Loss

class OccupancyFlowMapLoss(Loss):
    
    def __init__(self, device, config):
        """Initializes the loss module with weights and configuration options."""

        self.device = device
        self.ogm_weight = config.ogm_weight  # Weight for observed occupancy loss
        self.flow_weight = config.flow_weight  # Weight for flow loss
        self.occ_weight = config.occ_weight  # Weight for occluded occupancy loss
        self.replica = config.replica  # Scaling factor

        # Loss functions
        self.focal_loss = partial(sigmoid_focal_loss, from_logits=True)
        self.occlude_focal_loss = partial(sigmoid_focal_loss, from_logits=True)
        self.flow_focal_loss = partial(sigmoid_focal_loss, from_logits=False)
        self.bce = batch_binary_cross_entropy

        # Flags for optional behaviors
        self.no_use_warp = config.no_use_warp
        self.use_focal_loss = config.use_focal_loss
        self.use_pred = config.use_pred
        self.flow_origin_weight = config.flow_origin_weight
        self.use_gt = config.use_gt

        self.observed_occupancy_cross_entropy = MeanMetric().to(self.device)
        self.occluded_occupancy_cross_entropy = MeanMetric().to(self.device)
        self.flow_norm = MeanMetric().to(self.device)
        self.flow_wrap_occupancy_cross_entropy = MeanMetric().to(self.device)
        
    def update(self, loss_dict):
        
        self.observed_occupancy_cross_entropy.update(loss_dict['observed_occupancy_cross_entropy'])
        self.occluded_occupancy_cross_entropy.update(loss_dict['occluded_occupancy_cross_entropy'])
        self.flow_norm.update(loss_dict['flow_norm'])
        self.flow_wrap_occupancy_cross_entropy.update(loss_dict['flow_wrap_occupancy_cross_entropy'])
        
    def compute(self, pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits,
                 gt_observed_occupancy_logits, gt_occluded_occupancy_logits, gt_flow_logits,
                 flow_origin_occupancy_logits, gt_mask) -> Dict[str, torch.Tensor]:
        """
        Calculate the occupancy flow map loss.

        Args:
            pred_observed_occupancy_logits: Predicted logits for observed occupancy [B, H, W, T ,1].
            pred_occluded_occupancy_logits: Predicted logits for occluded occupancy [B, H, W, T ,1].
            pred_flow_logits: Predicted logits for flow [B, H, W, T , 2].
            gt_observed_occupancy_logits: Ground truth observed occupancy [B, H, W, T ,1].
            gt_occluded_occupancy_logits: Ground truth occluded occupancy [B, H, W, T ,1].
            gt_flow_logits: Ground truth flow [B, H, W, T ,2].
            flow_origin_occupancy_logits: Flow origin occupancy logits.
            gt_mask: Ground truth mask to indicate valid timesteps. [B, T]

        Returns:
            A dictionary of loss components.
        """
        

        loss_dict = {'observed_occupancy_cross_entropy': [], 'occluded_occupancy_cross_entropy': [], 'flow_norm': [], 'flow_wrap_occupancy_cross_entropy': []}
        batch_size, occupancy_flow_map_height, occupancy_flow_map_width, n_waypoints,_ = gt_flow_logits.shape

        # Generate identity indices for warping operations
        h = torch.arange(0, occupancy_flow_map_height, dtype=torch.float32)
        w = torch.arange(0, occupancy_flow_map_width, dtype=torch.float32)
        h_idx, w_idx = torch.meshgrid(h, w, indexing="xy")
        identity_indices = torch.stack((w_idx.T, h_idx.T), dim=-1).detach().to(self.device)

        # Iterate over each waypoint to calculate losses
        f_c = []  # Track flow correction factor
        for k in range(n_waypoints):
            # Extract predicted and ground truth values for current waypoint
            pred_observed_occupancy = pred_observed_occupancy_logits[..., k, :]
            pred_occluded_occupancy = pred_occluded_occupancy_logits[..., k, :]
            pred_flow = pred_flow_logits[..., k, :]

            true_observed_occupancy = gt_observed_occupancy_logits[..., k, :]
            true_occluded_occupancy = gt_occluded_occupancy_logits[..., k, :]
            true_flow = gt_flow_logits[..., k, :]

            flow_origin_occupancy = flow_origin_occupancy_logits[..., k, :]
            mask = gt_mask[..., k].view(-1, 1, 1, 1)
            # Calculate observed occupancy loss
            loss_dict['observed_occupancy_cross_entropy'].append(
                self._sigmoid_xe_loss(true_observed_occupancy * mask, pred_observed_occupancy * mask, self.ogm_weight))

            # Calculate occluded occupancy loss
            loss_dict['occluded_occupancy_cross_entropy'].append(
                self._sigmoid_occ_loss(true_occluded_occupancy * mask, pred_occluded_occupancy * mask, self.occ_weight))

            # Combine observed and occluded occupancy
            true_all_occupancy = torch.clamp(true_observed_occupancy + true_occluded_occupancy, 0, 1)

            # Flow loss with optional warping correction
            if self.use_gt:
                warped_indices = identity_indices + true_flow
                wp_org = sample(image=flow_origin_occupancy, warp=warped_indices, pixel_type=0)
                res = binary_average_precision(preds=wp_org * true_all_occupancy, target=true_all_occupancy.to(torch.int8), thresholds=100)
                res = (1 - res < 1.0).to(torch.float32)
            else:
                res = 1.0

            f_c.append(res)
            loss_dict['flow_norm'].append((k + 1) * res * self._flow_loss(true_flow * mask, pred_flow * mask))

            # Warp loss
            if not self.no_use_warp:
                warped_indices = identity_indices + pred_flow
                wp_origin = sample(image=flow_origin_occupancy, warp=warped_indices, pixel_type=0)

                if self.use_pred:
                    loss_dict['flow_wrap_occupancy_cross_entropy'].append(res * self._sigmoid_xe_warp_loss_pred(
                        true_all_occupancy * mask, pred_observed_occupancy * mask, pred_occluded_occupancy * mask, wp_origin, self.flow_origin_weight))
                else:
                    loss_dict['flow_wrap_occupancy_cross_entropy'].append(res * self._sigmoid_xe_warp_loss(
                        true_all_occupancy * mask, true_observed_occupancy * mask, true_occluded_occupancy * mask, wp_origin, self.flow_origin_weight))

        # Compute the final loss as the average across waypoints
        n_dict = {
            'observed_occupancy_cross_entropy': sum(loss_dict['observed_occupancy_cross_entropy']) / n_waypoints,
            'occluded_occupancy_cross_entropy': sum(loss_dict['occluded_occupancy_cross_entropy']) / n_waypoints,
            'flow_norm': sum(loss_dict['flow_norm']) / sum(f_c)
        }

        if not self.no_use_warp:
            n_dict['flow_wrap_occupancy_cross_entropy'] = sum(loss_dict['flow_wrap_occupancy_cross_entropy']) / sum(f_c)
        else:
            n_dict['flow_wrap_occupancy_cross_entropy'] = 0.0

        return n_dict
    def get_result(self):
        """Returns the computed loss values."""
        result_dict = {}
        
        result_dict['observed_occupancy_cross_entropy'] = self.observed_occupancy_cross_entropy.compute()
        result_dict['occluded_occupancy_cross_entropy'] = self.occluded_occupancy_cross_entropy.compute()
        result_dict['flow_norm'] = self.flow_norm.compute()
        result_dict['flow_wrap_occupancy_cross_entropy'] = self.flow_wrap_occupancy_cross_entropy.compute()
        
        # Reset the metrics after computation
        self.observed_occupancy_cross_entropy.reset()
        self.occluded_occupancy_cross_entropy.reset()
        self.flow_norm.reset()
        self.flow_wrap_occupancy_cross_entropy.reset()
        # Clear the loss dictionary
        
        return result_dict
    def reset(self):
        """Resets the loss metrics."""
        self.observed_occupancy_cross_entropy.reset()
        self.occluded_occupancy_cross_entropy.reset()
        self.flow_norm.reset()
        self.flow_wrap_occupancy_cross_entropy.reset()
    
    # Utility function to flatten tensors
    def _batch_flatten(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.reshape(input_tensor, [input_tensor.size(0), -1])

    def _sigmoid_xe_loss(
        self,
        true_occupancy: torch.Tensor,
        pred_occupancy: torch.Tensor,
        loss_weight: float = 1000,
    ) -> torch.Tensor:
        """Computes sigmoid cross-entropy loss over all grid cells."""
        # Since the mean over per-pixel cross-entropy values can get very small,
        # we compute the sum and multiply it by the loss weight before computing
        # the mean.
        if self.use_focal_loss:
            xe_sum = torch.sum(
                self.focal_loss(
                    targets=self._batch_flatten(true_occupancy),
                    inputs=self._batch_flatten(pred_occupancy)
                )) + torch.sum(
            F.binary_cross_entropy_with_logits(
                target=self._batch_flatten(true_occupancy),
                input=self._batch_flatten(pred_occupancy),
                reduction="none"
            ))
        else:
            xe_sum = torch.sum(
            F.binary_cross_entropy_with_logits(
                target=self._batch_flatten(true_occupancy),
                input=self._batch_flatten(pred_occupancy),
                reduction="none"
            ))
        # Return mean.
        return loss_weight * xe_sum / (torch.numel(pred_occupancy)*self.replica)
    
    def _sigmoid_occ_loss(
        self,
        true_occupancy: torch.Tensor,
        pred_occupancy: torch.Tensor,
        loss_weight: float = 1000,
    ) -> torch.Tensor:
        """Computes sigmoid cross-entropy loss over all grid cells."""
        # Since the mean over per-pixel cross-entropy values can get very small,
        # we compute the sum and multiply it by the loss weight before computing
        # the mean.
        if self.use_focal_loss:
            xe_sum = torch.sum(
                self.occlude_focal_loss(
                    targets=self._batch_flatten(true_occupancy),
                    inputs=self._batch_flatten(pred_occupancy)
                )) +torch.sum(
            F.binary_cross_entropy_with_logits(
                target=self._batch_flatten(true_occupancy),
                input=self._batch_flatten(pred_occupancy),
                reduction="none"
            ))
        else:
            xe_sum = torch.sum(
            F.binary_cross_entropy_with_logits(
                target=self._batch_flatten(true_occupancy),
                input=self._batch_flatten(pred_occupancy),
                reduction="none"
            ))
        # Return mean.
        return loss_weight * xe_sum / (torch.numel(pred_occupancy)*self.replica)
    
    def _sigmoid_xe_warp_loss(
        self,
        true_occupancy: torch.Tensor,
        pred_occupancy_obs: torch.Tensor,
        pred_occupancy_occ: torch.Tensor,
        warped_origin: torch.Tensor,
        loss_weight: float = 1000,
    ) -> torch.Tensor:
        labels=self._batch_flatten(true_occupancy)
        sig_logits = self._batch_flatten(torch.sigmoid(pred_occupancy_obs)+ torch.sigmoid(pred_occupancy_occ))
        sig_logits = torch.clamp(sig_logits,0,1)
        joint_flow_occ_logits = sig_logits * self._batch_flatten(warped_origin)
        # joint_flow_occ_logits = tf.clip_by_value(joint_flow_occ_logits,0,1)
        if self.use_focal_loss:
            joint_flow_occ_logits = torch.clamp(joint_flow_occ_logits,1e-7,1 - 1e-7) # for numerical stability
            xe_sum = torch.sum(self.flow_focal_loss(targets=labels,inputs=joint_flow_occ_logits)) + torch.sum(self.bce(input=joint_flow_occ_logits, target=labels))
        else:
            xe_sum =torch.sum(F.binary_cross_entropy_with_logits(target=labels,input=joint_flow_occ_logits, reduction="none"))

        # Return mean.
        return loss_weight * xe_sum / (torch.numel(true_occupancy)*self.replica)
    
    def _sigmoid_xe_warp_loss_pred(
        self,
        true_occupancy: torch.Tensor,
        pred_occupancy_obs: torch.Tensor,
        pred_occupancy_occ: torch.Tensor,
        warped_origin: torch.Tensor,
        loss_weight: float = 1000,
    ) -> torch.Tensor:
        labels=self._batch_flatten(true_occupancy)
        sig_logits = self._batch_flatten(torch.sigmoid(pred_occupancy_obs)+torch.sigmoid(pred_occupancy_occ))
        sig_logits = torch.clip_by_value(sig_logits,0,1)
        joint_flow_occ_logits =  self._batch_flatten(warped_origin)*sig_logits
        if self.use_focal_loss:
            xe_sum = torch.sum(self.flow_focal_loss(targets=labels,inputs=joint_flow_occ_logits)) + torch.sum(self.bce(target=labels,input=joint_flow_occ_logits))
        else:
            xe_sum =torch.sum(F.binary_cross_entropy_with_logits(target=labels,input=joint_flow_occ_logits,reduction="none"))
        xe_sum = torch.sum(self.bce(target=labels,input=joint_flow_occ_logits) )

        # Return mean.
        return loss_weight * xe_sum / (torch.numel(true_occupancy)*self.replica)

    # Flow loss based on L1 norm
    def _flow_loss(
        self, 
        true_flow: torch.Tensor, 
        pred_flow: torch.Tensor, 
        loss_weight: float = 1,
    ) -> torch.Tensor:
        
        """Computes L1 flow loss."""
        diff = true_flow - pred_flow
        # Ignore predictions in areas where ground-truth flow is zero.
        # [batch_size, height, width, 1], [batch_size, height, width, 1]
        true_flow_dx, true_flow_dy = torch.chunk(true_flow, 2, dim=-1)

        # [batch_size, height, width, 1]
        flow_exists = torch.logical_or(
            torch.not_equal(true_flow_dx, 0.0),
            torch.not_equal(true_flow_dy, 0.0),
        )
        flow_exists = flow_exists.to(torch.float32)
        diff = diff * flow_exists
        diff_norm = torch.linalg.norm(diff, ord=1, dim=-1)  # L1 norm.
        diff_norm_sum = torch.sum(diff_norm)
        flow_exists_sum = torch.sum(flow_exists) * self.replica / 2 # / 2 since (dx, dy) is counted twice.
        if torch.is_nonzero(flow_exists_sum):
            mean_diff = torch.div(diff_norm_sum, flow_exists_sum)
        else:
            mean_diff = 0
        return loss_weight * mean_diff
    
# Sigmoid focal loss function with optional alpha and gamma parameters.
def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, from_logits=False, alpha: float = 0.25, gamma: float = 2):
    if from_logits:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    else:
        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return torch.sum(loss, dim=-1)

# Batch-wise binary cross-entropy loss.
def batch_binary_cross_entropy(input, target):
    return torch.mean(F.binary_cross_entropy(input=input, target=target, reduction="none"), dim=-1)

