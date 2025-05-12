import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_graph_frames(scalars, num_frames, frame_size=(128, 128), max_val=0.3):
    """Generate frames of a changing graph using Matplotlib."""
    graph_frames = []
    for i in range(num_frames):
        fig, ax = plt.subplots()
        ax.plot(scalars[:i+1], linewidth=4)
        ax.set_xlim(0, num_frames)
        ax.set_ylim(0, max_val)
        ax.axis('off')  # Turn off the axis
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        fig.canvas.draw()
        
        # Convert Matplotlib figure to numpy array
        graph_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_frame = graph_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize to match frame size
        graph_frame = cv2.resize(graph_frame, frame_size)
        
        # Convert to tensor
        graph_frame = torch.tensor(graph_frame).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        
        graph_frames.append(graph_frame)
        plt.close(fig)
    return torch.stack(graph_frames)


def concat_uncertainty_with_video(data, video_pred, disagreement_ensemble, thr=3.0, max_val=3.0):

    is_first = torch.tensor(data["is_first"][:6], device="cuda")
    # Initialize the index tensor
    valid_tensor = torch.ones_like(is_first, dtype=torch.bool)

    # Iterate over each row to find the second True and create the index tensor
    for i in range(is_first.shape[0]):
        true_indices = torch.nonzero(is_first[i])
        if len(true_indices) > 1:
            second_true_index = true_indices[1].item()
            valid_tensor[i, second_true_index:] = False

    disagreement_ensemble = torch.clamp(disagreement_ensemble, min=thr) -thr
    disagreement_ensemble[~valid_tensor] = 0.

    # Generate graph frames for each sequence
    graph_frames_list = []
    for i in range(disagreement_ensemble.shape[0]):
        scalars = disagreement_ensemble[i].squeeze(-1).cpu().numpy()
        graph_frames = generate_graph_frames(scalars, disagreement_ensemble.shape[1], frame_size=(128, 128), max_val=max_val)
        graph_frames_list.append(graph_frames)

    # Concatenate graph frames with video frames
    video_with_graph = []
    
    for i in range(video_pred.shape[0]):
        
        video_frame = torch.tensor(video_pred[i]).cpu()  # Shape: (32, H, W, 3)
        frames_to_concatenate = [video_frame]  # Start with the mandatory video_frame
    
        graph_frame = graph_frames_list[i]  # Shape: (32, 3, 128, 128)
        graph_frame = graph_frame.permute(0, 2, 3, 1)  # Shape: (32, 128, 128, 3)
        frames_to_concatenate.append(graph_frame)
        combined_frame = torch.cat(frames_to_concatenate, dim=1)
        video_with_graph.append(combined_frame)

    video_with_graph = torch.stack(video_with_graph).numpy()

    return video_with_graph
