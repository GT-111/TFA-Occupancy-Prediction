import torch
import torch.nn.functional as F
from utils.metrics_utils import sample
from google.protobuf import text_format
from typing import Dict
from functools import partial
from torchmetrics.functional.classification import binary_average_precision
from utils.file_utils import get_config

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    from_logits = False,
    alpha: float = 0.25,
    gamma: float = 2,
):
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

def batch_binary_cross_entropy(input, target):
    return torch.mean(F.binary_cross_entropy(input=input, target=target, reduction="none"), dim=-1)


class OGMFlow_loss():

    def __init__(self, config, ogm_weight=1000.0,occ_weight=1000.0,flow_weight=1.0,flow_origin_weight=1000.0,replica=1.0,no_use_warp=False,use_pred=False,
    use_focal_loss=True,use_gt=False):

        self.config = config
        self.ogm_weight = ogm_weight
        self.flow_weight = flow_weight
        self.occ_weight = occ_weight
        self.replica = replica
        self.focal_loss = partial(sigmoid_focal_loss, from_logits=True)
        self.occlude_focal_loss = partial(sigmoid_focal_loss, from_logits=True)
        self.no_use_warp = no_use_warp
        self.use_focal_loss = use_focal_loss
        self.use_pred = use_pred
        self.flow_origin_weight = flow_origin_weight
        self.flow_focal_loss = partial(sigmoid_focal_loss, from_logits=False)
        self.bce = batch_binary_cross_entropy
        self.use_gt = use_gt
    
    def warping_preparation(self):
        pass

    def __call__(self,
        pred_observed_occupancy_logits,
        pred_occluded_occupancy_logits,
        pred_flow_logits,
        gt_observed_occupancy_logits,
        gt_occluded_occupancy_logits,
        gt_flow_logits,
        flow_origin_occupancy_logits,
        

    ) -> Dict[str, torch.Tensor]:
        """Loss function.

        Args:
            

        Returns:
            A dict containing different loss tensors:
            observed_xe: Observed occupancy cross-entropy loss.
            occluded_xe: Occluded occupancy cross-entropy loss.
            flow: Flow loss.
        """

        device = flow_origin_occupancy_logits.device

        loss_dict = {}
        # Store loss tensors for each waypoint and average at the end.
        loss_dict['observed_xe'] = []
        loss_dict['occluded_xe'] = []
        loss_dict['flow'] = []
        loss_dict['flow_warp_xe'] = []


        #Preparation for flow warping:
        h = torch.arange(0, self.config.occupancy_flow_map.grid_size.y, dtype=torch.float32)
        w = torch.arange(0, self.config.occupancy_flow_map.grid_size.x, dtype=torch.float32)
        h_idx, w_idx = torch.meshgrid(h, w, indexing="xy")
        # These indices map each (x, y) location to (x, y).
        # [height, width, 2] but storing x, y coordinates.
        identity_indices = torch.stack(
        (
            w_idx.T,
            h_idx.T,
        ),dim=-1)
        identity_indices = identity_indices.detach().to(device)
        # print(identity_indices.shape)
        # Iterate over waypoints.
        # flow_origin_occupancy = curr_ogm[:,128:128+256,128:128+256,tf.newaxis]
        n_waypoints = self.config.task_config.num_waypoints
        has_true_observed_occupancy = {-1: True}
        has_true_occluded_occupancy = {-1: True}
        true_obs_cnt,true_occ_cnt,true_flow_cnt = [],[],[]
        f_c = []
        for k in range(n_waypoints):
            # Occupancy cross-entropy loss.
            pred_observed_occupancy = pred_observed_occupancy_logits[:,k]
            pred_occluded_occupancy = pred_occluded_occupancy_logits[:,k]
            pred_flow = pred_flow_logits[:,k]

            true_observed_occupancy = gt_observed_occupancy_logits[:,k]
            true_occluded_occupancy = gt_occluded_occupancy_logits[:,k]
            true_flow = gt_flow_logits[:,k]

            flow_origin_occupancy = flow_origin_occupancy_logits[:,k]

            # Accumulate over waypoints.
            loss_dict['observed_xe'].append(
                self._sigmoid_xe_loss(
                    true_occupancy=true_observed_occupancy,
                    pred_occupancy=pred_observed_occupancy,
                    loss_weight=self.ogm_weight)) 
            loss_dict['occluded_xe'].append(
                self._sigmoid_occ_loss(
                    true_occupancy=true_occluded_occupancy,
                    pred_occupancy=pred_occluded_occupancy,
                    loss_weight=self.occ_weight))
            
            true_all_occupancy = torch.clamp(true_observed_occupancy + true_occluded_occupancy, 0, 1)
            if self.use_gt:
                warped_indices = identity_indices + true_flow
                wp_org = sample(
                        image=flow_origin_occupancy,
                        warp=warped_indices,
                        pixel_type=0,
                    )
                res = binary_average_precision(preds=wp_org*true_all_occupancy, target=true_all_occupancy.to(torch.int8), thresholds=100)
                res = (1 - res<1.0).to(torch.float32)
            else:
                res = 1.0
            f_c.append(res)
            loss_dict['flow'].append((k+1)*res*self._flow_loss(true_flow,pred_flow))

            # flow warp_loss:
            if not self.no_use_warp:
                warped_indices = identity_indices + pred_flow
                wp_origin = sample(
                    image=flow_origin_occupancy,
                    warp=warped_indices,
                    pixel_type=0,
                )
                if self.use_pred:
                    loss_dict['flow_warp_xe'].append(res*self._sigmoid_xe_warp_loss_pred(true_all_occupancy,
                pred_observed_occupancy, pred_occluded_occupancy, wp_origin,
                loss_weight=self.flow_origin_weight))
                else:
                    loss_dict['flow_warp_xe'].append(res*self._sigmoid_xe_warp_loss(true_all_occupancy,
                    true_observed_occupancy, true_occluded_occupancy, wp_origin,
                    loss_weight=self.flow_origin_weight))
            
        # Mean over waypoints.
        n_dict = {}
        n_dict['observed_xe'] = sum(loss_dict['observed_xe']) / n_waypoints
        n_dict['occluded_xe'] = sum(loss_dict['occluded_xe']) / n_waypoints
        n_dict['flow'] = sum(loss_dict['flow']) / sum(f_c)

        if not self.no_use_warp:
            n_dict['flow_warp_xe'] = sum(loss_dict['flow_warp_xe']) / sum(f_c)
        else:
            n_dict['flow_warp_xe'] = 0.0
        return n_dict


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
        sig_logits = torch.clamp(sig_logits,0,1)
        joint_flow_occ_logits =  self._batch_flatten(warped_origin)*sig_logits
        if self.use_focal_loss:
            xe_sum = torch.sum(self.flow_focal_loss(targets=labels,inputs=joint_flow_occ_logits)) + torch.sum(self.bce(target=labels,input=joint_flow_occ_logits))
        else:
            xe_sum =torch.sum(F.binary_cross_entropy_with_logits(target=labels,input=joint_flow_occ_logits,reduction="none"))
        xe_sum = torch.sum(self.bce(target=labels,input=joint_flow_occ_logits) )

        # Return mean.
        return loss_weight * xe_sum / (torch.numel(true_occupancy)*self.replica)

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
        flow_exists_sum = torch.sum(flow_exists)*self.replica / 2 # / 2 since (dx, dy) is counted twice.
        if torch.is_nonzero(flow_exists_sum):
            mean_diff = torch.div(diff_norm_sum, flow_exists_sum)
        else:
            mean_diff = 0
        return loss_weight * mean_diff

    def _batch_flatten(self,input_tensor: torch.Tensor) -> torch.Tensor:
        """Flatten tensor to a shape [batch_size, -1]."""
        image_shape = input_tensor.size()
        return torch.reshape(input_tensor, [*image_shape[0:1], -1])


def test_loss(config):
    his_len = config.task_config.history_length
    pred_len = config.task_config.prediction_length
    batch_size = config.dataloader_config.batch_size
    grid_size_x = config.occupancy_flow_map.grid_size.x
    grid_size_y = config.occupancy_flow_map.grid_size.y
    num_waypoints = config.task_config.num_waypoints
    
    dummy_pred_observed_occupancy = cur_ogm = torch.rand((batch_size, num_waypoints, grid_size_x, grid_size_y, 1))
    dummy_pred_occluded_occupancy = torch.rand((batch_size, num_waypoints,  grid_size_x, grid_size_y, 1))
    dummy_pred_flow = torch.rand((batch_size, num_waypoints, grid_size_x, grid_size_y, 2))
    
    dummy_gt_observed_occupancy = cur_ogm = torch.rand((batch_size, num_waypoints, grid_size_x, grid_size_y, 1))
    dummy_gt_occluded_occupancy = torch.rand((batch_size, num_waypoints,  grid_size_x, grid_size_y, 1))
    dummy_gt_flow = torch.rand((batch_size, num_waypoints, grid_size_x, grid_size_y, 2))
    flow_origin_occupancy = torch.rand((batch_size, grid_size_x, grid_size_y, 1))
    # pred_observed_occupancy_logits,
    # pred_occluded_occupancy_logits,
    # pred_flow_logits,
    # gt_observed_occupancy_logits,
    # gt_occluded_occupancy_logits,
    # gt_flow_logits,
    # flow_origin_occupancy,

    loss_fn = OGMFlow_loss(config, replica=1, no_use_warp=False, use_pred=False, use_gt=True, use_focal_loss=True)

    # values = loss_fn(dummy_pred_observed_occupancy, dummy_pred_occluded_occupancy, dummy_pred_flow, dummy_gt_observed_occupancy, dummy_gt_occluded_occupancy, dummy_gt_flow, flow_origin_occupancy)
    values = loss_fn(dummy_gt_observed_occupancy, dummy_gt_occluded_occupancy, dummy_gt_flow, dummy_gt_observed_occupancy, dummy_gt_occluded_occupancy, dummy_gt_flow, flow_origin_occupancy)
    print(values)


if __name__ == "__main__":
    config = get_config("./config.yaml")
    test_loss(config)