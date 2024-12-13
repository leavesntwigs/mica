import plotly.graph_objects as go
import numpy as np

# Generate meshgrid data
r = np.linspace(0, 1, 10)
theta = np.linspace(0, 2*np.pi, 20)
R, Theta = np.meshgrid(r, theta)

# Convert to Cartesian coordinates
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# Create the figure
fig = go.Figure(data=[
    go.Scatterpolar(
        r=R.flatten(),
        theta=Theta.flatten(),
        mode='markers',
        marker=dict(
            size=15,
            color='blue',
            symbol='square-x'
        )
    )
])

# Customize the layout
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1]),
        angularaxis=dict(
            tickmode='array',
            tickvals=np.arange(0, 360, 45), 
            ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
        )
    )
)

fig.show()
