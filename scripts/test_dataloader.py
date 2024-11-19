import torch
from dataloader import get_dataloader
import matplotlib.pyplot as plt
import logging

def visualize_trajectory():
    # Get dataloader in debug mode
    dataloader = get_dataloader(debug=True)
    
    # Get a single batch
    batch = next(iter(dataloader))
    
    # Take the first trajectory from the batch
    trajectory_frames = batch['frames'][0]  # This is a list of tensors
    subtask = batch['subtask_name'][0]
    label = batch['label'][0]
    num_frames = batch['num_frames'][0].item()  # Convert tensor to integer
    
    # Log information about the trajectory
    print(f"\nTrajectory Information:")
    print(f"Subtask: {subtask}")
    print(f"Success: {bool(label)}")
    print(f"Number of frames: {num_frames}")
    print(f"First frame shape: {trajectory_frames[0].shape}")
    
    # Create a grid of frames
    num_cols = 4
    num_rows = (num_frames + num_cols - 1) // num_cols
    
    plt.figure(figsize=(15, 3*num_rows))
    for i in range(num_frames):
        plt.subplot(num_rows, num_cols, i + 1)
        # Convert from [C, H, W] to [H, W, C]
        frame = trajectory_frames[i].permute(1, 2, 0).numpy()
        plt.imshow(frame)
        plt.title(f"Frame {i}")
        plt.axis('off')
    
    plt.suptitle(f"Trajectory for task: {subtask}\nSuccess: {bool(label)}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        visualize_trajectory()
    except Exception as e:
        logging.error(f"Error visualizing trajectory: {str(e)}", exc_info=True)