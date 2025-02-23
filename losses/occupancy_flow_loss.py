import torch
import torch.nn.functional as F
from utils.metrics_utils import sample
from typing import Dict
from functools import partial
from torchmetrics.functional.classification import binary_average_precision

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


class OccupancyFlowMapLoss:
    def __init__(self, config, ogm_weight=1000.0, occ_weight=1000.0, flow_weight=1.0, flow_origin_weight=1000.0,
                 replica=1.0, no_use_warp=False, use_pred=False, use_focal_loss=True, use_gt=False):
        """Initializes the loss module with weights and configuration options."""

        self.config = config
        self.ogm_weight = ogm_weight  # Weight for observed occupancy loss
        self.flow_weight = flow_weight  # Weight for flow loss
        self.occ_weight = occ_weight  # Weight for occluded occupancy loss
        self.replica = replica  # Scaling factor

        # Loss functions
        self.focal_loss = partial(sigmoid_focal_loss, from_logits=True)
        self.occlude_focal_loss = partial(sigmoid_focal_loss, from_logits=True)
        self.flow_focal_loss = partial(sigmoid_focal_loss, from_logits=False)
        self.bce = batch_binary_cross_entropy

        # Flags for optional behaviors
        self.no_use_warp = no_use_warp
        self.use_focal_loss = use_focal_loss
        self.use_pred = use_pred
        self.flow_origin_weight = flow_origin_weight
        self.use_gt = use_gt

    def __call__(self, pred_observed_occupancy_logits, pred_occluded_occupancy_logits, pred_flow_logits,
                 gt_observed_occupancy_logits, gt_occluded_occupancy_logits, gt_flow_logits,
                 flow_origin_occupancy_logits, gt_mask) -> Dict[str, torch.Tensor]:
        """
        Calculate the occupancy flow map loss.

        Args:
            pred_observed_occupancy_logits: Predicted logits for observed occupancy.
            pred_occluded_occupancy_logits: Predicted logits for occluded occupancy.
            pred_flow_logits: Predicted logits for flow.
            gt_observed_occupancy_logits: Ground truth observed occupancy.
            gt_occluded_occupancy_logits: Ground truth occluded occupancy.
            gt_flow_logits: Ground truth flow.
            flow_origin_occupancy_logits: Flow origin occupancy logits.
            gt_mask: Ground truth mask to indicate valid regions.

        Returns:
            A dictionary of loss components.
        """
        device = flow_origin_occupancy_logits.device

        loss_dict = {'observed_xe': [], 'occluded_xe': [], 'flow': [], 'flow_warp_xe': []}
        n_waypoints = self.config.task_config.num_waypoints

        # Generate identity indices for warping operations
        h = torch.arange(0, self.config.occupancy_flow_map.grid_size.y, dtype=torch.float32)
        w = torch.arange(0, self.config.occupancy_flow_map.grid_size.x, dtype=torch.float32)
        h_idx, w_idx = torch.meshgrid(h, w, indexing="xy")
        identity_indices = torch.stack((w_idx.T, h_idx.T), dim=-1).detach().to(device)

        # Iterate over each waypoint to calculate losses
        f_c = []  # Track flow correction factor
        for k in range(n_waypoints):
            # Extract predicted and ground truth values for current waypoint
            pred_observed_occupancy = pred_observed_occupancy_logits[:, k]
            pred_occluded_occupancy = pred_occluded_occupancy_logits[:, k]
            pred_flow = pred_flow_logits[:, k]

            true_observed_occupancy = gt_observed_occupancy_logits[..., k]
            true_occluded_occupancy = gt_occluded_occupancy_logits[..., k]
            true_flow = gt_flow_logits[..., k, :]

            flow_origin_occupancy = flow_origin_occupancy_logits[..., k][..., None]
            mask = gt_mask[..., k]

            # Calculate observed occupancy loss
            loss_dict['observed_xe'].append(
                self._sigmoid_xe_loss(true_observed_occupancy * mask, pred_observed_occupancy * mask, self.ogm_weight))

            # Calculate occluded occupancy loss
            loss_dict['occluded_xe'].append(
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
            loss_dict['flow'].append((k + 1) * res * self._flow_loss(true_flow * mask.unsqueeze(-1), pred_flow * mask.unsqueeze(-1)))

            # Warp loss
            if not self.no_use_warp:
                warped_indices = identity_indices + pred_flow
                wp_origin = sample(image=flow_origin_occupancy, warp=warped_indices, pixel_type=0)

                if self.use_pred:
                    loss_dict['flow_warp_xe'].append(res * self._sigmoid_xe_warp_loss_pred(
                        true_all_occupancy * mask, pred_observed_occupancy * mask, pred_occluded_occupancy * mask, wp_origin, self.flow_origin_weight))
                else:
                    loss_dict['flow_warp_xe'].append(res * self._sigmoid_xe_warp_loss(
                        true_all_occupancy * mask, true_observed_occupancy * mask, true_occluded_occupancy * mask, wp_origin, self.flow_origin_weight))

        # Compute the final loss as the average across waypoints
        n_dict = {
            'observed_xe': sum(loss_dict['observed_xe']) / n_waypoints,
            'occluded_xe': sum(loss_dict['occluded_xe']) / n_waypoints,
            'flow': sum(loss_dict['flow']) / sum(f_c)
        }

        if not self.no_use_warp:
            n_dict['flow_warp_xe'] = sum(loss_dict['flow_warp_xe']) / sum(f_c)
        else:
            n_dict['flow_warp_xe'] = 0.0

        return n_dict

    # Utility function to flatten tensors
    def _batch_flatten(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.reshape(input_tensor, [input_tensor.size(0), -1])

    # Sigmoid cross-entropy loss for observed occupancy
    def _sigmoid_xe_loss(self, true_occupancy, pred_occupancy, loss_weight):
        if self.use_focal_loss:
            xe_sum = torch.sum(self.focal_loss(targets=self._batch_flatten(true_occupancy), inputs=self._batch_flatten(pred_occupancy)))
        else:
            xe_sum = torch.sum(F.binary_cross_entropy_with_logits(target=self._batch_flatten(true_occupancy), input=self._batch_flatten(pred_occupancy), reduction="none"))
        return loss_weight * xe_sum / (torch.numel(pred_occupancy) * self.replica)

    # Sigmoid cross-entropy loss for occluded occupancy
    def _sigmoid_occ_loss(self, true_occupancy, pred_occupancy, loss_weight):
        if self.use_focal_loss:
            xe_sum = torch.sum(self.occlude_focal_loss(targets=self._batch_flatten(true_occupancy), inputs=self._batch_flatten(pred_occupancy)))
        else:
            xe_sum = torch.sum(F.binary_cross_entropy_with_logits(target=self._batch_flatten(true_occupancy), input=self._batch_flatten(pred_occupancy), reduction="none"))
        return loss_weight * xe_sum / (torch.numel(pred_occupancy) * self.replica)

    # Flow loss based on L1 norm
    def _flow_loss(self, true_flow, pred_flow, loss_weight=1):
        diff = (true_flow - pred_flow) * (torch.logical_or(true_flow[..., 0] != 0.0, true_flow[..., 1] != 0.0).float())
        diff_norm_sum = torch.sum(torch.linalg.norm(diff, ord=1, dim=-1))
        flow_exists_sum = torch.sum(diff != 0) * self.replica / 2
        mean_diff = diff_norm_sum / flow_exists_sum if flow_exists_sum else 0
        return loss_weight * mean_diff
