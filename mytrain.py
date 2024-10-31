from utils.file_utils import get_config, get_last_checkpoint
from utils.dataset_utils import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from tqdm import tqdm
import utils.training_utils as training_utils
import torch
import os



def setup(config, gup_id):
    """
    """
    ## TODO: Implement baseline model
    model = None

    loss_fn = None
    
    optimizer = torch.optim.NAdam(params=model.parameters(), 
                                  lr=config.training.optimizer.lr,
                                  weight_decay=config.training.optimizer.weight_decay) 
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=config.training.scheduler.step_size,
                                                gamma=config.training.scheduler.gamma) 
    return model, loss_fn, optimizer, scheduler



def model_training(config, gpu_id, logger:SummaryWriter):
    
    train_dataloader, val_dataloader, _ = get_dataloader(config)
    if get_last_checkpoint(config.paths.checkpoints) is not None:
        
        checkpoint = torch.load(get_last_checkpoint(config.paths.checkpoints))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        continue_ep = checkpoint['epoch'] + 1
        
        if gpu_id == 0:
            print(f'Continue_training...ep:{continue_ep+1}')
    else:
        continue_ep = 0
    
    
    model, loss_fn, optimizer, scheduler = setup(config, gpu_id)
    
    for epoch in range(config.training.epochs):
        if epoch<continue_ep:
            if gpu_id == 0:
                print("\nskip epoch {}/{}".format(epoch+1, config.training.epochs))
            continue
        
        model.train()
        train_loss = MeanMetric().to(gpu_id)
        train_loss_occ = MeanMetric().to(gpu_id)
        train_loss_flow = MeanMetric().to(gpu_id)
        train_loss_warp = MeanMetric().to(gpu_id)
        
        loop = tqdm(enumerate(train_dataloader))
        for batch, data in loop:
            input_dict, ground_truth_dict = training_utils.parse_data(data, gpu_id)
            
            output_dict = model(input_dict)
            loss_dict = loss_fn(output_dict, ground_truth_dict)
            loss_value = torch.sum(sum(loss_dict.values()))
            
            # backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            
            train_loss.update(loss_dict['observed_xe'])
            train_loss_occ.update(loss_dict['occluded_xe'])
            train_loss_flow.update(loss_dict['flow'])
            train_loss_warp.update(loss_dict['flow_warp_xe'])
            
            if batch % 10 == 0:
                logger.add_custom_scalars_multiline(
                    {
                        "train_loss": {
                            "observed_xe": train_loss.compute(),
                            "occluded_xe": train_loss_occ.compute(),
                            "flow": train_loss_flow.compute(),
                            "flow_warp_xe": train_loss_warp.compute()
                        }
                    }
                )
        
        scheduler.step()
        
        ## validate
        
        valid_loss      = MeanMetric().to(gpu_id)
        valid_loss_occ  = MeanMetric().to(gpu_id)
        valid_loss_flow = MeanMetric().to(gpu_id)
        valid_loss_warp = MeanMetric().to(gpu_id)

        # valid_metrics = OGMFlowMetrics(gpu_id, no_warp=False)


        model.eval()
        
        with torch.no_grad():
            loop = tqdm(enumerate(val_dataloader))
            for batch, data in loop:
                # inputs: will automatically be put on right device when passed to model 
                map_img = data['map_image']
                centerlines = data['centerlines']
                actors = data['actors']
                occl_actors = data['occl_actors']
                ogm = data['ogm']
                flow = data['vec_flow']

                # ground truths directly put on device for loss / metrics
                gt_obs_ogm  = data['gt_obs_ogm'].to(gpu_id)
                gt_occ_ogm  = data['gt_occ_ogm'].to(gpu_id)
                gt_flow     = data['gt_flow'].to(gpu_id)
                origin_flow = data['origin_flow'].to(gpu_id)


                # forward pass
                outputs = model(ogm,map_img,obs=actors,occ=occl_actors,mapt=centerlines,flow=flow)

                # compute losses
                loss_dict = loss_fn(outputs, gt_flow, gt_occ_ogm, gt_obs_ogm)
                loss_value = torch.sum(sum(loss_dict.values()))
                
                valid_loss.update(loss_dict['observed_xe'])
                valid_loss_occ.update(loss_dict['occluded_xe'])
                valid_loss_flow.update(loss_dict['flow'])
                valid_loss_warp.update(loss_dict['flow_warp_xe'])
                
                logger.add_custom_scalars_multiline(
                    {
                        "val_loss": {
                            "observed_xe": valid_loss.compute(),
                            "occluded_xe": valid_loss_occ.compute(),
                            "flow": valid_loss_flow.compute(),
                            "flow_warp_xe": valid_loss_warp.compute()
                        }
                    }
                )
            #     metrics = 
            #     valid_metrics.update(metrics)
            # val_res_dict = valid_metrics.compute()
        
        if (epoch+1) % config.training.checkpoint_interval == 0:
            training_utils.save_checkpoint(model, optimizer, scheduler, epoch, config.paths.checkpoints)










config = get_config('./config.json')
checkpoints_path = config.paths.checkpoints
os.path.exists(checkpoints_path) or os.makedirs(checkpoints_path)

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    # add tensor board
    logger = SummaryWriter(log_dir=config.paths.logs)
    
    model_training(gpu_id, config, logger)
else:
    print('No GPU available')