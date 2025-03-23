import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from cycler import cycler

def visualize_timeline(data, clip_id=None, save_path=None):
    """
    Visualize speaker activity timelines using horizontal bars.
    
    Args:
        data (dict): Dictionary with speaker IDs as keys and lists of 
                    (start_time, end_time) tuples as values
        clip_id (str): Optional identifier for the title
        save_path (str): Optional path to save the figure
    """
    # Sort speakers numerically if possible, otherwise lexicographically
    try:
        speakers = sorted(data.keys(), key=lambda x: int(x))
    except ValueError:
        speakers = sorted(data.keys())
    
    n_speakers = len(speakers)
    if n_speakers == 0:
        return  # No data to plot
    
    # Setup figure with dynamic height
    fig_height = max(3, n_speakers * 0.7)  # Adjust height based on speaker count
    fig, ax = plt.subplots(figsize=(18, fig_height))
    
    # Create a dictionary of colors for each speaker ID
    colors = {speaker: plt.cm.rainbow(i / n_speakers) for i, speaker in enumerate(speakers)}

    # Plot each speaker's timeline
    for y, speaker in enumerate(speakers):
        intervals = data[speaker]
        # Convert to lists for vectorized plotting
        starts = [s for s, _ in intervals]
        ends = [e for _, e in intervals]
        color = colors[speaker]
        for s,e in zip(starts,ends):
            ax.hlines(
                y=y, 
                xmin=s, 
                xmax=e, 
                linewidth=50, 
                label=speaker,
                color=color
            )
    
    # Configure axes
    ax.set_yticks(range(n_speakers))
    ax.set_yticklabels(speakers)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Speaker ID')
    ax.set_title(f'Speaker Timeline - {clip_id}' if clip_id else 'Speaker Timeline')
    
    # Set axis limits with padding
    ax.set_ylim(-0.5, n_speakers - 0.5)
    if data:
        all_times = [t for times in data.values() for t, _ in times] + \
                   [t for times in data.values() for _, t in times]
        ax.set_xlim(min(all_times) - 1, max(all_times) + 1)
    
    # Add grid and style
    ax.grid(True, linestyle=':', alpha=0.7, axis='x')
    
    # Save or display
    if save_path:
        plt.savefig(f"{save_path}/{clip_id}_timeline.png" if clip_id else f"{save_path}/timeline.png", 
                    bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

