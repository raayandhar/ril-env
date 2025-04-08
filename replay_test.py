from ril_env.replay_buffer import ReplayBuffer

# Load the replay buffer
buffer = ReplayBuffer.create_from_path('./recordings/replay_buffer.zarr')

# Print the number of episodes
print(f"Number of episodes: {buffer.n_episodes}")

# If there are episodes, examine the first one
if buffer.n_episodes > 0:
    # Get the first episode
    episode = buffer.get_episode(0)
    
    # Print the keys (what types of data are stored)
    print(f"Data keys: {list(episode.keys())}")
    
    # For each key, print some information
    for key, value in episode.items():
        print(f"\n{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Data type: {value.dtype}")
        
        # Print some sample values
        if len(value.shape) == 1:  # 1D array
            print(f"  First few values: {value[:190]}")
        else:  # multi-dimensional array
            print(f"  First value: {value[0]}")
            if len(value) > 1:
                print(f"  Second value: {value[1]}")
