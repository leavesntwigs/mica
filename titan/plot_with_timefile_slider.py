import plotly
import matplotlib.pyplot as plt
import xarray as xr
import xradar as xd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import os

# import colormap_fetch

# color maps ...

#    print("color_scale_name: ", color_scale_name)
#    colorscale_for_go = colormap_fetch.fetch(color_scale_name)
#
#
#    (edges_norm, colors_norm) = colormap_fetch.normalize_colormap(edges, color_scale_hex)
#    cmap = colors.ListedColormap(colors_norm) # (color_scale_hex)
#    # Plot the quadmesh
#    #            (X(column), Y(row), Z(row,column))
#    #psm = ax.pcolormesh(Theta, R, Z, cmap='seismic', rasterized=True, shading='nearest')
#    psm = ax.pcolormesh(Theta, R, Z,
#        cmap=cmap,
#        vmin=edges_norm[0], vmax=edges_norm[-1],
#        # norm=norm,
#        # norm=colors.BoundaryNorm(edges, ncolors=len(edges)),
#        rasterized=True, shading='nearest')
# end color maps

def file_list(directory_path):
    files = []
    try:
        # Get all entries (files and directories) in the path
        entries = os.listdir(directory_path)
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            # Check if the entry is a file
            if os.path.isfile(full_path):
                files.append(entry)
    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
    return files    

# read cartesian radar data

#filename = "/Users/brenda/data/ams2025/radar/cart/qc/KingCity/20220521/ncf_20220521_173550.nc"

# KingCity goes with derecho
path = "/Users/brenda/data/ams2025/radar/cart/qc/KingCity/20220521"
#ds = xr.open_dataset(filename)
#z=np.nan_to_num(ds.VEL.data[0,17], nan=-32)
# fig = go.Figure(data=go.Heatmap(z=z, type='heatmap', colorscale='Viridis'))
# fig.show()

z_step = 10

filenames = sorted(file_list(path))
print(filenames)

# Create figure
fig = go.Figure()
fig = make_subplots(rows=1, cols=1, shared_xaxes='all', shared_yaxes='all')



# Add traces, one for each slider step
#     storm traces are within each tenth (0-9 are for first time step
#        10 - 19 are all traces for the second file / time step   
# 

#for step in np.arange(0, 20, 1):
#    fig.add_trace(
#        go.Heatmap(
#            visible=False,
#            z=np.nan_to_num(ds.VEL.data[0,step], nan=-32), 
#            type='heatmap', 
#            colorscale='Viridis',
#            #line=dict(color="#00CED1", width=6),
#            name="v = " + str(step),
#            #x=np.arange(0, 10, 0.01),
#            #y=np.sin(step * np.arange(0, 10, 0.01))
#            ))

for step in range(0, len(filenames)):
    file = filenames[step]
    file_path = os.path.join(path, file) 
    ds = xr.open_dataset(file_path)
    #print(ds.z0.data[10])
    # time_fig = make_subplots(rows=1, cols=1, 
        # shared_xaxes='all', shared_yaxes='all') # [[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=[0,100,200,0], y=[0,200,0,0]))
    fig.add_trace(
        go.Heatmap(
            visible=False,
            z=np.nan_to_num(ds.VEL.data[0,z_step], nan=-32), 
            type='heatmap', 
            colorscale='Viridis',
            #line=dict(color="#00CED1", width=6),
            name="v = " + str(step),
            zmin=-32, zmax=40,
            ))
    #fig.add_trace(time_fig, row=1, col=1, secondary_y=False)

# Make 10th trace visible
fig.data[10].visible = True

# to make image square 
fig.update_layout(yaxis_scaleanchor="x")

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        label=i # filenames[i][13:19]
        #label=filenames[i][13:19]
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    # step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
    if (i % 2 == 1): 
        step["args"][0]["visible"][i-1] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "time step: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()

