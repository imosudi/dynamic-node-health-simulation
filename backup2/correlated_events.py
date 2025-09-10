import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Generate sample data
np.random.seed(42)
n_events = 10
metrics = ['CPU', 'Memory', 'Disk', 'Network']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# Create event data
events = []
start_time = datetime(2024, 1, 15, 8, 0)

for i in range(n_events):
    event_start = start_time + timedelta(hours=i*2 + np.random.randint(-1, 2))
    event_end = event_start + timedelta(hours=1 + np.random.random()*2)
    
    # Generate metric contributions (sum to 1)
    contributions = np.random.dirichlet(np.ones(len(metrics)))
    
    events.append({
        'start': event_start,
        'end': event_end,
        'contributions': dict(zip(metrics, contributions)),
        'severity': np.random.choice(['Low', 'Medium', 'High']),
        'group': f"Group {i % 3 + 1}"  # Group events for correlation
    })

# Create figure and axis
fig, (ax_main, ax_contrib) = plt.subplots(2, 1, figsize=(14, 10), 
                                         gridspec_kw={'height_ratios': [3, 1]})

# Plot grouped ribbons (correlated events)
groups = sorted(set(event['group'] for event in events))
group_colors = ['#FF9999', '#99FF99', '#9999FF']

for group_idx, group in enumerate(groups):
    group_events = [e for e in events if e['group'] == group]
    
    for event in group_events:
        # Convert datetime to numeric for plotting
        start_num = mdates.date2num(event['start'])
        end_num = mdates.date2num(event['end'])
        duration = end_num - start_num
        
        # Create ribbon (rounded rectangle)
        ribbon_height = 0.6
        y_center = group_idx + 1
        y_bottom = y_center - ribbon_height/2
        
        # Draw the main ribbon
        rect = patches.Rectangle(
            (start_num, y_bottom), duration, ribbon_height,
            linewidth=2, edgecolor=group_colors[group_idx], 
            facecolor=group_colors[group_idx] + '80',  # 80 = 50% opacity
            alpha=0.7
        )
        ax_main.add_patch(rect)
        
        # Add dashed border for persistent windows
        border = patches.Rectangle(
            (start_num, y_bottom), duration, ribbon_height,
            linewidth=2, edgecolor=group_colors[group_idx], 
            facecolor='none', linestyle='--', alpha=0.9
        )
        ax_main.add_patch(border)
        
        # Add event label
        ax_main.text(start_num + duration/2, y_center, f"Event {events.index(event)+1}",
                    ha='center', va='center', fontweight='bold', fontsize=9)

# Set up main axis
ax_main.set_yticks(range(1, len(groups) + 1))
ax_main.set_yticklabels(groups)
ax_main.set_ylabel('Event Groups')
ax_main.set_title('Correlated Events with Grouped Ribbons and Persistent Windows', fontsize=14, pad=20)

# Format x-axis for dates
ax_main.xaxis_date()
fig.autofmt_xdate()

# Create contribution bars at detection time
detection_times = [event['start'] for event in events]
detection_nums = [mdates.date2num(dt) for dt in detection_times]

# Prepare contribution data for stacked bars
contrib_data = np.array([[event['contributions'][metric] for event in events] 
                        for metric in metrics])

# Plot stacked contribution bars
bottom = np.zeros(len(events))
bar_width = 0.02  # Width in days (converted from datetime)

for i, metric in enumerate(metrics):
    ax_contrib.bar(detection_nums, contrib_data[i], bottom=bottom, 
                  width=bar_width, color=colors[i], label=metric, alpha=0.8)
    bottom += contrib_data[i]

# Set up contribution axis
ax_contrib.set_ylabel('Metric Contribution')
ax_contrib.set_xlabel('Detection Time')
ax_contrib.set_title('Per-Metric Contribution at Detection Time', fontsize=12, pad=15)
ax_contrib.legend(loc='upper right')
ax_contrib.set_ylim(0, 1)
ax_contrib.xaxis_date()
ax_contrib.grid(True, alpha=0.3)

# Add vertical lines from detection points to ribbons
for i, (event, det_num) in enumerate(zip(events, detection_nums)):
    group_idx = groups.index(event['group'])
    y_ribbon = group_idx + 1
    y_contrib = 0  # Start from bottom of contribution bar
    
    # Draw connecting line
    ax_main.plot([det_num, det_num], [y_contrib, y_ribbon - 0.3], 
                color='gray', linestyle=':', alpha=0.7, linewidth=1)
    
    # Add marker at detection point
    ax_main.plot(det_num, y_ribbon - 0.3, 'o', color='red', markersize=6, alpha=0.8)

# Adjust layout and add grid
ax_main.grid(True, alpha=0.3)
plt.tight_layout()

# Add overall description
plt.figtext(0.02, 0.02, 
           "Visualization shows correlated events as grouped ribbons with dashed borders.\n"
           "Persistent windows indicate ongoing events, and connecting lines show detection points\n"
           "with metric contribution breakdown in the lower panel.",
           fontsize=10, alpha=0.7)

# Save the plot instead of showing it
plt.savefig('correlated_events_visualization.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'correlated_events_visualization.png'")

# Print event details for reference
print("\nEvent Details:")
for i, event in enumerate(events):
    contributions_str = ", ".join([f"{k}: {v:.3f}" for k, v in event['contributions'].items()])
    print(f"Event {i+1}: {event['start'].strftime('%H:%M')}-{event['end'].strftime('%H:%M')} "
          f"| Group: {event['group']} | Contributions: {{{contributions_str}}}")

# Also print a summary of the visualization
print(f"\nVisualization Summary:")
print(f"- {len(events)} events across {len(groups)} groups")
print(f"- Events grouped by: {', '.join(groups)}")
print(f"- Metrics analyzed: {', '.join(metrics)}")
print(f"- Time range: {events[0]['start'].strftime('%Y-%m-%d %H:%M')} to {events[-1]['end'].strftime('%Y-%m-%d %H:%M')}")