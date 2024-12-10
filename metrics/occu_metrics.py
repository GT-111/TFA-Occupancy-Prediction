import torch
from typing import List
from torchmetrics.functional.classification import binary_average_precision
from utils.metrics_utils import sample, _mean



def compute_occupancy_flow_metrics(
    config,
    pred_observed_occupancy_logits,
    pred_occluded_occupancy_logits,
    pred_flow_logits,
    gt_observed_occupancy_logits,
    gt_occluded_occupancy_logits,
    gt_flow_logits,
    flow_origin_occupancy_logits,
    no_warp: bool=False
):
  """Computes occupancy (observed, occluded) and flow metrics.

  Args:
    config: OccupancyFlowTaskConfig proto message.
    true_waypoints: Set of num_waypoints ground truth labels.
    pred_waypoints: Predicted set of num_waypoints occupancy and flow topdowns.

  Returns:
    OccupancyFlowMetrics proto message containing mean metric values averaged
      over all waypoints.
  """
  # Accumulate metric values for each waypoint and then compute the mean.
  metrics_dict = {
      'vehicles_observed_auc': [],
      'vehicles_occluded_auc': [],
      'vehicles_observed_iou': [],
      'vehicles_occluded_iou': [],
      'vehicles_flow_epe': [],
      'vehicles_flow_warped_occupancy_auc': [],
      'vehicles_flow_warped_occupancy_iou': [],
  }

  has_true_observed_occupancy = {-1: True}
  has_true_occluded_occupancy = {-1: True}



  # Warp flow-origin occupancies according to predicted flow fields.
  if not no_warp:
    warped_flow_origins = _flow_warp(
        config=config,
        pred_flow_logits=pred_flow_logits,
        flow_origin_occupancy_logits=flow_origin_occupancy_logits,
    )

  # Iterate over waypoints.
  for k in range(config.task_config.num_waypoints):
    pred_observed_occupancy = pred_observed_occupancy_logits[:,k]
    pred_occluded_occupancy = pred_occluded_occupancy_logits[:,k]
    pred_flow = pred_flow_logits[:,k]
    true_observed_occupancy = gt_observed_occupancy_logits[...,k][...,None]
    true_occluded_occupancy = gt_occluded_occupancy_logits[...,k][...,None]
    true_flow = gt_flow_logits[...,k,:]
    # adding this CAUSE DISTRIBUTE ERROR!!!!
    # has_true_observed_occupancy[k] = tf.reduce_max(true_observed_occupancy) > 0
    # has_true_occluded_occupancy[k] = tf.reduce_max(true_occluded_occupancy) > 0
    # has_true_flow = (has_true_observed_occupancy[k] and
    #                   has_true_observed_occupancy[k - 1]) or (
    #                       has_true_occluded_occupancy[k] and
    #                       has_true_occluded_occupancy[k - 1])

    # Compute occupancy metrics.
    if True:#:has_true_observed_occupancy[k]:
      metrics_dict['vehicles_observed_auc'].append(
          _compute_occupancy_auc(true_observed_occupancy,
                                pred_observed_occupancy))
      metrics_dict['vehicles_observed_iou'].append(
        _compute_occupancy_soft_iou(true_observed_occupancy,
                                    pred_observed_occupancy))
    if True:#has_true_occluded_occupancy[k]:                       
      metrics_dict['vehicles_occluded_auc'].append(
          _compute_occupancy_auc(true_occluded_occupancy,
                                pred_occluded_occupancy))
      
      metrics_dict['vehicles_occluded_iou'].append(
          _compute_occupancy_soft_iou(true_occluded_occupancy,
                                      pred_occluded_occupancy))
      
    # Compute flow metrics.
    if True:#has_true_flow:
      metrics_dict['vehicles_flow_epe'].append(
          _compute_flow_epe(true_flow, pred_flow))

      # Compute flow-warped occupancy metrics.
      # First, construct ground-truth occupancy of all observed and occluded
      # vehicles.
      
      
      true_all_occupancy = torch.clamp(
          true_observed_occupancy + true_occluded_occupancy, 0, 1)
      # Construct predicted version of same value.
      pred_all_occupancy = torch.clamp(
          pred_observed_occupancy + pred_occluded_occupancy, 0, 1)
      # We expect to see the same results by warping the flow-origin occupancies.
      if not no_warp:
        flow_warped_origin_occupancy = warped_flow_origins[k]
        # Construct quantity that requires both prediction paths to be correct.
        flow_grounded_pred_all_occupancy = (
            pred_all_occupancy * flow_warped_origin_occupancy)
        # Now compute occupancy metrics between this quantity and ground-truth.
        # reverse the order of true and pred
        metrics_dict['vehicles_flow_warped_occupancy_auc'].append(
            _compute_occupancy_auc(true_all_occupancy, flow_grounded_pred_all_occupancy))
        metrics_dict['vehicles_flow_warped_occupancy_iou'].append(
            _compute_occupancy_soft_iou(flow_grounded_pred_all_occupancy,
                                        true_all_occupancy))

  # Compute means and return as proto message.
  metrics_dict = {k: _mean(v) for k, v in metrics_dict.items()}
    
  return metrics_dict


def _compute_occupancy_auc(
    true_occupancy: torch.Tensor,
    pred_occupancy: torch.Tensor,
) -> torch.Tensor:
  """Computes the AUC between the predicted and true occupancy grids.

  Args:
    true_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].
    pred_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].

  Returns:
    AUC: float32 scalar.
  """
  return binary_average_precision(preds=pred_occupancy, target=true_occupancy.to(torch.int8), thresholds=100)


def _compute_occupancy_soft_iou(
    true_occupancy: torch.Tensor,
    pred_occupancy: torch.Tensor,
) -> torch.Tensor:
  """Computes the soft IoU between the predicted and true occupancy grids.

  Args:
    true_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].
    pred_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].

  Returns:
    Soft IoU score: float32 scalar.
  """
  true_occupancy = torch.reshape(true_occupancy, [-1])
  pred_occupancy = torch.reshape(pred_occupancy, [-1])

  intersection = torch.mean(torch.multiply(pred_occupancy, true_occupancy))
  true_sum = torch.mean(true_occupancy)
  pred_sum = torch.mean(pred_occupancy)
  # Scenes with empty ground-truth will have a score of 0.
  score = torch.nan_to_num(torch.div(
            intersection,
            pred_sum + true_sum - intersection), posinf=0, neginf=0)
  return score


def _compute_flow_epe(
    true_flow: torch.Tensor,
    pred_flow: torch.Tensor,
) -> torch.Tensor:
  """Computes average end-point-error between predicted and true flow fields.

  Flow end-point-error measures the Euclidean distance between the predicted and
  ground-truth flow vector endpoints.

  Args:
    true_flow: float32 Tensor shaped [batch_size, height, width, 2].
    pred_flow: float32 Tensor shaped [batch_size, height, width, 2].

  Returns:
    EPE averaged over all grid cells: float32 scalar.
  """
  # [batch_size, height, width, 2]
  diff = true_flow - pred_flow
  # [batch_size, height, width, 1], [batch_size, height, width, 1]
  true_flow_dx, true_flow_dy = torch.chunk(true_flow, 2, dim=-1)
  # [batch_size, height, width, 1]
  flow_exists = torch.logical_or(
      torch.not_equal(true_flow_dx, 0.0),
      torch.not_equal(true_flow_dy, 0.0),
  )
  flow_exists = flow_exists.to(torch.float32)

  diff = diff * flow_exists
  # [batch_size, height, width, 1]
  epe = torch.linalg.norm(diff, ord=2, dim=-1, keepdim=True)
  # Scalar.
  sum_epe = torch.sum(epe)
  # Scalar.
  sum_flow_exists = torch.sum(flow_exists)
  # Scalar.
  mean_epe = torch.nan_to_num(torch.div(
            sum_epe,
            sum_flow_exists), posinf=0, neginf=0)

  return mean_epe


def _flow_warp(
    config,
    pred_flow_logits,
    flow_origin_occupancy_logits,
) -> List[torch.Tensor]:
  """Warps ground-truth flow-origin occupancies according to predicted flows.

  Performs bilinear interpolation and samples from 4 pixels for each flow
  vector.

  Args:
    config: OccupancyFlowTaskConfig proto message.
    true_waypoints: Set of num_waypoints ground truth labels.
    pred_waypoints: Predicted set of num_waypoints occupancy and flow topdowns.

  Returns:
    List of `num_waypoints` occupancy grids for vehicles as float32
      [batch_size, height, width, 1] tensors.
  """

  device = flow_origin_occupancy_logits.device

  h = torch.arange(0, config.occupancy_flow_map.grid_size.y, dtype=torch.float32, device=device)
  w = torch.arange(0, config.occupancy_flow_map.grid_size.x, dtype=torch.float32, device=device)
  h_idx, w_idx = torch.meshgrid(h, w, indexing="xy")
  # These indices map each (x, y) location to (x, y).
  # [height, width, 2] but storing x, y coordinates.
  identity_indices = torch.stack(
      (
          w_idx.T,
          h_idx.T,
      ),
      dim=-1,
  )

  warped_flow_origins = []
  for k in range(config.task_config.num_waypoints):
    # [batch_size, height, width, 1]
    # [batch_size, height, width, 2]
    pred_flow = pred_flow_logits[:,k]
    flow_origin_occupancy = flow_origin_occupancy_logits[...,k][...,None]
    # Shifting the identity grid indices according to predicted flow tells us
    # the source (origin) grid cell for each flow vector.  We simply sample
    # occupancy values from these locations.
    # [batch_size, height, width, 2]
    warped_indices = identity_indices + pred_flow
    # Pad flow_origin with a blank (zeros) boundary so that flow vectors
    # reaching outside the grid bring in zero values instead of producing edge
    # artifacts.
    # flow_origin_occupancy = tf.pad(flow_origin_occupancy,
    #                                [[0, 0], [1, 1], [1, 1], [0, 0]])
    # Shift warped indices as well to map to the padded origin.
    # warped_indices = warped_indices + 1
    # NOTE: tensorflow graphics expects warp to contain (x, y) as well.
    # [batch_size, height, width, 2]
    warped_origin = sample(
        image=flow_origin_occupancy,
        warp=warped_indices,
        pixel_type=0,
    )
    warped_flow_origins.append(warped_origin)

  return warped_flow_origins