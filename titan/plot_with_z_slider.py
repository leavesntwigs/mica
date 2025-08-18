import plotly
import matplotlib.pyplot as plt
import xarray as xr
import xradar as xd
import plotly.graph_objects as go
import numpy as np


# read cartesian radar data

filename = "/Users/brenda/data/ams2025/radar/cart/qc/KingCity/20220521/ncf_20220521_173550.nc"
ds = xr.open_dataset(filename)
z=np.nan_to_num(ds.VEL.data[0,17], nan=-32)
# fig = go.Figure(data=go.Heatmap(z=z, type='heatmap', colorscale='Viridis'))
# fig.show()


# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for step in np.arange(0, 20, 1):
    fig.add_trace(
        go.Heatmap(
            visible=False,
            z=np.nan_to_num(ds.VEL.data[0,step], nan=-32), 
            type='heatmap', 
            colorscale='Viridis',
            #line=dict(color="#00CED1", width=6),
            name="v = " + str(step),
            #x=np.arange(0, 10, 0.01),
            #y=np.sin(step * np.arange(0, 10, 0.01))
            ))

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
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "z: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()

