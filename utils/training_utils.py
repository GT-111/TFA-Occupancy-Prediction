import os
import torch
from utils.file_utils import get_last_file_with_extension

def save_checkpoint(model, optimizer, scheduler, epoch, proj_exp_dir, global_step):
    """
    Save model checkpoint
    """
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(proj_exp_dir + f'/epoch_{epoch+1}.pth'))


def load_checkpoint(model, optimizer, scheduler, proj_exp_dir, gpu_id):
    """
    Loads the latest checkpoint if available and updates the model, optimizer, and scheduler.

    Args:
        model: The model to load state dict into.
        optimizer: The optimizer to load state dict into.
        scheduler: The scheduler to load state dict into.
        proj_exp_dir: Directory to search for the checkpoint file.
        gpu_id: The GPU ID, used for printing messages if gpu_id == 0.

    Returns:
        continue_ep: The next epoch to continue training from.
        global_step: The global step loaded from the checkpoint.
    """
    checkpoint_path = get_last_file_with_extension(proj_exp_dir, '.pth')
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        continue_ep = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        if gpu_id == 0:
            print(f'Continue_training...ep:{continue_ep + 1}')
    else:
        continue_ep = 0
        global_step = 0

    return continue_ep, global_step
