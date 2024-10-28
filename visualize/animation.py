from utils.file_utils import get_config
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dataset.occ_flow_utils import GridMap


# Load data
config = get_config()
data = np.load('/hdd/HetianGuo/I24/63858a2cfb3ff533c12df166_112.0_36.0.npy', allow_pickle=True).item()
grid_map = GridMap(config)
occupancy_map, flow = grid_map.get_map_flow(data)

# Grid size
grid_size_x = config.dataset.grid_size_x
grid_size_y = config.dataset.grid_size_y
x, y = np.meshgrid(np.arange(grid_size_x), np.arange(grid_size_y))

# Layout for the Plotly figure
layout = go.Layout(
    width=800, height=800,
    xaxis=dict(range=[0, grid_size_x], title='X'),
    yaxis=dict(range=[0, grid_size_y], title='Y', scaleanchor='x'),
    showlegend=False,
    title='Occupancy Map and Flow Visualization',
    plot_bgcolor='rgba(0,0,0,0)',
    updatemenus=[  # Play/Pause controls
        dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 500, 'redraw': False},
                        
                        'transition': {'duration': 300}
                    }]
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                )
            ]
        )
    ],
    sliders=[  # Slider for controlling frames
        dict(
            steps=[
                dict(
                    method='animate',
                    args=[[str(idx)], {
                        'mode': 'immediate',
                        'frame': {'duration': 300},
                        'transition': {'duration': 300}
                    }],
                    label=str(idx)
                ) for idx in range(flow.shape[0])
            ],
            active=0, transition={'duration': 300},
            x=0.1, y=0, len=0.9
        )
    ]
)

# Initialize frames
frames = []

# Generate frames with quiver arrows, heatmap, and road lines
for idx in range(flow.shape[0]):
    traces = []

    # Add road lines
    road_lines_y = [12, 24, 36, 48, 60, -12, -24, -36, -48, -60]
    for road_line_y in road_lines_y:
        y_coord = road_line_y * grid_size_y / 160 + grid_size_y / 2
        traces.append(go.Scatter(
            x=[0, grid_size_x],  # Start and end on x-axis
            y=[y_coord, y_coord],  # Horizontal line at y
            mode='lines',
            line=dict(color='black', width=1)
        ))

    # Extract map and flow data for the current frame
    flow_frame = flow[idx].swapaxes(0, 1)
    map_frame = occupancy_map[idx].swapaxes(0, 1)
    valid_mask = (flow_frame[:, :, 0] != 0) | (flow_frame[:, :, 1] != 0)
    # Add quiver plot (if valid flow data exists)
    if flow_frame[valid_mask].shape[0] > 0:
        
        quiver_fig = ff.create_quiver(
            x[valid_mask], y[valid_mask],
            flow_frame[valid_mask][:, 0], flow_frame[valid_mask][:, 1],
            scale=0.2,
            line=dict(color='black', width=2)
        )
        
        traces.extend(quiver_fig.data)  # Add quiver traces
    else:
        # add blank quiver plot
        quiver_fig = ff.create_quiver(
            x=[0], y=[0],
            u=[0], v=[0],
            scale=0.2,
            line=dict(color='black', width=2)
        )
        traces.extend(quiver_fig.data)
    # Add heatmap (background)
    traces.append(go.Heatmap(
        z=map_frame,
        x=np.arange(grid_size_x),
        y=np.arange(grid_size_y),
        colorscale=[
            [0.0, 'rgba(0,0,0,0)'],
            [0.5, 'rgba(0,0,0,1)'],
            [1.0, 'rgba(0,0,0,1)']],
        opacity=0.2,
        zmin=0,
        zmax=1,
        showscale=False
    ))

    
    # fig = go.Figure(data=traces)
    # fig.show()
    # Create a frame with all traces
    frame = go.Frame(data=traces, name=str(idx))
    frames.append(frame)

# Initialize the figure with the first frame's data
fig = go.Figure(data=frames[0].data, layout=layout, frames=frames)

# Display the figure
fig.show()

import gif
gif.save(frames, 'example.gif', duration=100)