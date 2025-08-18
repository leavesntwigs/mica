import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#import matplotlib.dates as mdates
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#from shapely.geometry import Polygon

file = "/Users/brenda/data/ams2025/titan/ascii/Tracks2Ascii.derecho.txt"

#open file and extract column names
f = open(file)
lines = f.readlines()
f.close()

label_line_index = None  

for i, line in enumerate(lines):
    if 'labels' in line:
        label_line_index = i
        break  
labels = lines[label_line_index].split(":", 1)[1].strip().split(",")

#the data lines are the ones that do not start with #
data_lines = [line.strip() for line in lines if not line.startswith("#")]

rows = []
for line in data_lines:
    parts = line.split()

    try:
        # Try parsing the polygon count value (always right before 72 values)
        poly_count_index = -73  # 72 floats + 1 count (the column starts with the numnber 72, which is not part of the values)

        # Parents and children may be missing
        parent_str = parts[poly_count_index - 2]
        child_str = parts[poly_count_index - 1]

        # Handle missing values marked as "-"
        parents = int(parent_str) if parent_str != '-' else np.nan
        children = int(child_str) if child_str != '-' else np.nan

        # Polygon values: skip the count, get the next 72 values
        polygon_values = list(map(float, parts[poly_count_index + 1:]))

        # Fixed columns
        fixed_cols = parts[:poly_count_index - 2]

        # Combine into one row
        row = fixed_cols + [parents, children, polygon_values]
        rows.append(row)
    except Exception as e:
        continue



# Final columns: fixed + 3 custom ones
final_labels = labels[:len(rows[0]) - 3] + ['parents', 'children', 'nPolySidesPolygonRays']

# Create DataFrame
df = pd.DataFrame(rows, columns=final_labels)

# Convert date and time columns to datetime
df['date_utc'] = pd.to_datetime(
    df['Year'].astype(str) + '-' +
    df['Month'].astype(str).str.zfill(2) + '-' +
    df['Day'].astype(str).str.zfill(2) + ' ' +
    df['Hour'].astype(str).str.zfill(2) + ':' +
    df['Min'].astype(str).str.zfill(2) + ':' +
    df['Sec'].astype(str).str.zfill(2),
    format='%Y-%m-%d %H:%M:%S',
    utc=True
)
# Print df 
print(df)


# ----------


# fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

# Set the map extent (lon_min, lon_max, lat_min, lat_max)
# ax.set_extent([-84, -76, 42, 46], crs=ccrs.PlateCarree())
# ax.coastlines()

#color by time
# timesteps = df0['date_utc'].unique()
# palette = sns.color_palette("gist_ncar", n_colors=len(timesteps))
# gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size': 14}
# gl.ylabel_style = {'size': 14}

# for idx, row in df0.iterrows():
#    lat_centroid = float(row['EnvelopeCentroidLat(deg)'])
#    lon_centroid = float(row['EnvelopeCentroidLon(deg)'])
#    rays = row['nPolySidesPolygonRays']
#    
#    if not rays or len(rays) == 0:
#        continue  
#    
#    angles = np.deg2rad(np.arange(0, 360, 5))  # 72 vertices at every 5 degrees
#    rays = np.array(rays, dtype=float) #from centroid to vertex
#    
#    # Rays in km to degrees
#    ray_x = rays * np.cos(angles)
#    ray_y = rays * np.sin(angles)
#
#    # Approximate conversion from km to degrees lat/lon
#    lat_vertices = lat_centroid + ray_y / 111
#    lon_vertices = lon_centroid + ray_x / (111 * np.cos(np.deg2rad(lat_centroid)))
#
#    polygon_points = list(zip(lon_vertices, lat_vertices))
#    
#    poly = Polygon(polygon_points)
#    time_idx = np.where(timesteps == row['date_utc'])[0][0]
#
#    ax.add_geometries([poly], crs=ccrs.PlateCarree(),
#                      edgecolor=palette[time_idx], facecolor='none', linewidth=1)
#    
#    ax.plot(lon_centroid, lat_centroid, marker='o',color='grey', markersize=1.5, transform=ccrs.PlateCarree())



#plt.title("Track Polygons for ComplexNum=0", fontsize=16)
#plt.show()


# plot_with_timefile_slider.py
import plotly
import matplotlib.pyplot as plt
import xarray as xr
import xradar as xd
import plotly.graph_objects as go
import numpy as np
import os

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

# Add traces, one for each slider step
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
    fig.add_trace(
        go.Heatmap(
            visible=False,
            z=np.nan_to_num(ds.VEL.data[0,z_step], nan=-32), 
            type='heatmap', 
            colorscale='Viridis',
            #line=dict(color="#00CED1", width=6),
            name="v = " + str(step),
            zmin=-32, zmax=40,
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
        label=filenames[i][13:19]
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
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

