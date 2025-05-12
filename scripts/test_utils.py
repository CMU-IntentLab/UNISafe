import torch
import numpy as np
from pathlib import Path

def load_model(learner, model_path):
    checkpoint = torch.load(model_path, map_location="cuda")
    learner.agent.load_state_dict(checkpoint['agent_state_dict'])

def predict_and_compare_video_with_logger(learner, data, n, name):
    """
    Perform video prediction for a single data sample `n` times, 
    compare the results by concatenating them vertically, and log the video using the provided logger.

    Parameters:
        agent: The agent containing the world model (_wm).
        data: A single data sample (dataset entry).
        n: Number of prediction runs to compare.
        logger: Logger instance for saving the video.
        log_tag: Tag to associate with the logged video.
        fps: Frames per second for the output video.
    """

    predictions = []

    # Perform predictions `n` times
    for _ in range(n):
        video_pred = learner.agent._wm.video_pred(data)  # Expected shape: (6, 32, H, W, 3)
        predictions.append(video_pred.cpu().numpy())

    concatenated_video = np.concatenate(predictions, axis=2)

    # Log the video
    learner.logger.video(name, concatenated_video)
    learner.logger.write(step=learner.logger.step)
    learner.logger.step += 1

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


def construct_data_for_video_pred(env, initial_state, action_sequences):
    """
    Construct data for the video_pred method with a given initial state and action sequences.

    Args:
        env (gym.Env): The environment to interact with.
        initial_state (np.ndarray): The initial state to set in the environment.
        action_sequences (np.ndarray): The sequence of actions to take in the environment.

    Returns:
        dict: A dictionary containing the constructed data for video_pred.
    """
    obs_list = []
    action_list = []
    is_first_list = []
    is_terminal_list = []

    initial_state = np.array(initial_state)
    state = initial_state

    u_val = env.car.discrete_controls

    # Step through the environment using the given action sequences
    for t, action in enumerate(action_sequences):
        img = env.car.capture_image(state)
        obs_list.append(img)

        action_onethot = np.zeros(3)
        action_onethot[action.astype(np.uint8)] = 1 
        action_list.append(action_onethot)
        is_first_list.append(t == 0)
        is_terminal_list.append(t >= (len(action_sequences)-1))

        state = env.car.integrate_forward(state, u_val[action.astype(np.uint8)])

    # Convert lists to numpy arrays
    obs_array = np.array(obs_list)[np.newaxis, :]
    action_array = np.array(action_list)[np.newaxis, :]
    is_first_array = np.array(is_first_list)[np.newaxis, :]
    is_terminal_array = np.array(is_terminal_list)[np.newaxis, :]

    # Construct the data dictionary
    data = {
        "image": obs_array,
        "action": action_array,
        "is_first": is_first_array,
        "is_terminal": is_terminal_array,
    }

    return data

def predict_and_measure_disagreement(learner, data, n, name, thr=-3.0):
    video_pred, disagreement = learner.agent._wm.video_pred_ensemble(data, learner.agent._disag_ensemble)  # Expected shape: (6, 32, H, W, 3)
 
    _, disagreement_density = learner.agent._wm.video_pred_density(data, learner.agent._density_estimator)  # Expected shape: (6, 32, H, W, 3)    
    is_first = torch.tensor(data["is_first"][:6], device=disagreement.device)
    # Initialize the index tensor
    valid_tensor = torch.ones_like(is_first, dtype=torch.bool)

    # Iterate over each row to find the second True and create the index tensor
    for i in range(is_first.shape[0]):
        true_indices = torch.nonzero(is_first[i])
        if len(true_indices) > 1:
            second_true_index = true_indices[1].item()
            valid_tensor[i, second_true_index:] = False

    # Adjust disagreement shape from (6, 31, 1) to (6, 32, 1) by adding zeros
    print(disagreement.min(), disagreement.max())
    disagreement = torch.clamp(disagreement, min=thr) -thr
    disagreement[~valid_tensor] = 0.
    disagreement_density[~valid_tensor] = 1.
    max_val = disagreement.max().item()

    # Generate graph frames for each sequence
    graph_frames_list = []
    graph_frames_density_list = []
    for i in range(disagreement.shape[0]):
        scalars = disagreement[i].squeeze(-1).cpu().numpy()
        graph_frames = generate_graph_frames(scalars, disagreement.shape[1], frame_size=(128, 128), max_val=3)
        graph_frames_list.append(graph_frames)

        # Density
        scalars = disagreement_density[i].squeeze(-1).cpu().numpy()
        graph_frames = generate_graph_frames(scalars, disagreement_density.shape[1], frame_size=(128, 128), max_val=1)
        graph_frames_density_list.append(graph_frames)
    
    # Concatenate graph frames with video frames
    video_with_graph = []
    
    for i in range(video_pred.shape[0]):
        video_frame = torch.tensor(video_pred[i]).cpu()  # Shape: (32, H, W, 3)
        graph_frame = graph_frames_list[i]  # Shape: (32, 3, 128, 128)
        graph_frame = graph_frame.permute(0, 2, 3, 1)  # Shape: (32, 128, 128, 3)

        graph_frames_density = graph_frames_density_list[i]  # Shape: (32, 3, 128, 128)
        graph_frames_density = graph_frames_density.permute(0, 2, 3, 1)  # Shape: (32, 128, 128, 3)
        combined_frame = torch.cat([video_frame, graph_frame, graph_frames_density], dim=1)  # Concatenate along width
        video_with_graph.append(combined_frame)

    video_with_graph = torch.stack(video_with_graph).numpy()
    # Log the video
    learner.logger.video(name, video_with_graph)
    learner.logger.write(step=learner.logger.step)
    learner.logger.step += 1


def predict_and_measure_disagreement_ensemble(learner, data, n, name, thr=-3.0):
    video_pred, disagreement = learner.agent._wm.video_pred_ensemble(data, 
                                                                     ensemble=learner.agent._disag_ensemble if learner.config.use_ensemble else None,
                                                                     density=learner.agent._density_estimator if learner.config.use_density_estimator else None,
                                                                     rssm_ensemble=learner.config.use_ensemble_RSSM)  # Expected shape: (6, 32, H, W, 3)
 
    # _, disagreement_density = learner.agent._wm.video_pred_density(data, learner.agent._density_estimator)  # Expected shape: (6, 32, H, W, 3)    
    is_first = torch.tensor(data["is_first"][:6], device="cuda")
    # Initialize the index tensor
    valid_tensor = torch.ones_like(is_first, dtype=torch.bool)

    # Iterate over each row to find the second True and create the index tensor
    for i in range(is_first.shape[0]):
        true_indices = torch.nonzero(is_first[i])
        if len(true_indices) > 1:
            second_true_index = true_indices[1].item()
            valid_tensor[i, second_true_index:] = False

    # if learner.config.use_ensemble :

    disagreement_ensemble = disagreement["ensemble"] #disagreement["ensemble"]
    # Adjust disagreement shape from (6, 31, 1) to (6, 32, 1) by adding zeros
    print("ensemble", disagreement_ensemble.min(), disagreement_ensemble.max())
    disagreement_ensemble = torch.clamp(disagreement_ensemble, min=thr) -thr
    disagreement_ensemble[~valid_tensor] = 0.

    valid_disagreement = disagreement_ensemble[valid_tensor] + thr
    max_val = valid_disagreement.max()
    quantile_90 = torch.quantile(valid_disagreement, 0.9)
    quantile_95 = torch.quantile(valid_disagreement, 0.95)
    print("Max of valid disagreement:", max_val.item(), "90th quantile:", quantile_90.item(), "95th quantile:", quantile_95.item())

    # Generate graph frames for each sequence
    graph_frames_list = []
    graph_frames_density_list = []
    graph_frames_rssm_list = []
    for i in range(disagreement_ensemble.shape[0]):
        # if learner.config.use_ensemble :
        if True :
            scalars = disagreement_ensemble[i].squeeze(-1).cpu().numpy()
            graph_frames = generate_graph_frames(scalars, disagreement_ensemble.shape[1], frame_size=(128, 128), max_val=0.1)
            graph_frames_list.append(graph_frames)
    
    # Concatenate graph frames with video frames
    video_with_graph = []
    
    for i in range(video_pred.shape[0]):
        
        video_frame = torch.tensor(video_pred[i]).cpu()  # Shape: (32, H, W, 3)
        frames_to_concatenate = [video_frame]  # Start with the mandatory video_frame
        
        # if learner.config.use_ensemble :
        if True:
            graph_frame = graph_frames_list[i]  # Shape: (32, 3, 128, 128)
            graph_frame = graph_frame.permute(0, 2, 3, 1)  # Shape: (32, 128, 128, 3)
            frames_to_concatenate.append(graph_frame)

        combined_frame = torch.cat(frames_to_concatenate, dim=1)
        video_with_graph.append(combined_frame)

    video_with_graph = torch.stack(video_with_graph).numpy()
    # Log the video
    learner.logger.video(name, video_with_graph)
    learner.logger.write(step=learner.logger.step)
    learner.logger.step += 1


import tqdm
def predict_and_measure_disagreement_states(learner, thr=-3.0):

    def plot_angle(x_coords, y_coords, theta, disagreement_post, disagreement_imagine, target_angle, angle_tolerance):

        # Step 1: Filter states based on the angle
        angle_diff = np.abs((theta - target_angle + np.pi) % (2 * np.pi) - np.pi)  # Circular difference
        angle_mask = angle_diff <= angle_tolerance
        # angle_mask = np.abs(theta - target_angle) <= angle_tolerance
        filtered_x = x_coords[angle_mask]
        filtered_y = y_coords[angle_mask]
        filtered_imagine_uncertainty = disagreement_imagine[angle_mask]
        filtered_post_uncertainty = disagreement_post[angle_mask]

        # Step 2: Create subplots for Imagine and Post Disagreements
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

        # Plot Imagine Disagreements
        sc1 = axes[0].scatter(filtered_x, filtered_y, c=filtered_imagine_uncertainty, cmap='seismic', s=50)
        axes[0].set_title(f"Imagine Disagreement (Angle ~ {target_angle}, Tolerance = {angle_tolerance})")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].set_aspect('equal', adjustable='box')
        cbar1 = plt.colorbar(sc1, ax=axes[0])
        cbar1.set_label("Disagreement Imagine")

        # Plot Post Disagreements
        sc2 = axes[1].scatter(filtered_x, filtered_y, c=filtered_post_uncertainty, cmap='seismic', s=50)
        axes[1].set_title(f"Post Disagreement (Angle ~ {target_angle}, Tolerance = {angle_tolerance})")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        axes[0].set_aspect('equal', adjustable='box')
        cbar2 = plt.colorbar(sc2, ax=axes[1])
        cbar2.set_label("Disagreement Post")
        
        plt.tight_layout()
        plt.savefig("uncertainty_{}.png".format(target_angle))
        plt.close()


    all_states = []
    all_disagreements_imagine = []
    all_disagreements_post = []
    # Processing loop
    idx = 0
    for data in tqdm.tqdm(learner.expert_dataset):
        states_post, disagreement_imagine, disagreement_post = learner.agent._wm.measure_disagreement(data, learner.agent._disag_ensemble)
        
        # Filter based on "is_first"
        is_first = torch.tensor(data["is_first"], device=disagreement_post.device)
        valid_tensor = torch.ones_like(is_first, dtype=torch.bool)

        for i in range(is_first.shape[0]):
            true_indices = torch.nonzero(is_first[i])
            if len(true_indices) > 1:
                second_true_index = true_indices[1].item()
                valid_tensor[i, second_true_index:] = False

        # Filter valid states and disagreements
        state = states_post[valid_tensor].cpu()
        disagreement_imagine = disagreement_imagine[valid_tensor].cpu()
        disagreement_post = disagreement_post[valid_tensor].cpu()
        
        all_states.append(state)
        all_disagreements_post.append(disagreement_post)
        all_disagreements_imagine.append(disagreement_imagine)

        idx +=1

        if idx > 1000 :
            break

        # print("{} states accumulated".format(len(heatmap_accumulator)), end='\r')

    # Extract components
    # Combine all results after the loop
    all_states = torch.cat(all_states, dim=0)  # Combine along the first dimension
    all_disagreements_post = torch.cat(all_disagreements_post, dim=0)
    all_disagreements_imagine = torch.cat(all_disagreements_imagine, dim=0)

    x_coords, y_coords, theta = all_states[:, 0].numpy(), all_states[:, 1].numpy(), all_states[:, 2].numpy()
    disagreement_post = all_disagreements_post.numpy().flatten()
    disagreement_imagine = all_disagreements_imagine.numpy().flatten()

    plot_angle(x_coords, y_coords, theta, disagreement_post, disagreement_imagine, target_angle=0, angle_tolerance=0.2)
    plot_angle(x_coords, y_coords, theta, disagreement_post, disagreement_imagine, target_angle=np.pi/4, angle_tolerance=0.2)