import os
import plotly
import matplotlib.pyplot as plt
import xarray as xr
import xradar as xd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import datetime
import bhs
import build_lineage
import colormap_fetch

from datetime import datetime, timezone

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


def get_file_date_time(file_name):
    return file_name[4:19]

# df is a dataframe from TITAN ascii file that contains storm polygons
# df['utc', 'polygon_x', 'polygon_y']
def plot_with_timefile_slider(z_step, path, df, df_complete):


    print(df.keys())

    filenames = sorted(file_list(path))
    print(filenames)
   
 
    # associate polygons and radar data files by date & time 
    # 'ncf_20220521_145949.nc'   compare to 2022-05-21 14:59:49+00:00
    # create a parallel list of filenames with empty lists to keep the 
    #     row index of associated polygons
    # use the filename times as the keys, and then add the polygon row idx to the list
    associated = {get_file_date_time(f): []  for f in filenames}
    print(associated)

    df_sorted = df.sort_values(by=['utc'])

    #file_date_time = get_file_date_time(filenames[file_idx])
    #print(file_date_time)

    for idx, row in df_sorted.iterrows():
        poly_date_time = row['utc'].strftime('%Y%m%d_%H%M%S')
        associated[poly_date_time].append(idx)
                
    print(associated)        

    # get the storms for each time step
    sbts = bhs.build_helper_structures(df_complete)
    
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
  
    colorscale_for_go = colormap_fetch.fetch("dbz_color")
 
    trace_count = 0
    map_trace_indexes = []

    for step in range(0, len(filenames)):
        file = filenames[step]
        file_path = os.path.join(path, file) 
        ds = xr.open_dataset(file_path)
        x = ds.x0.data
        y = ds.y0.data
        #print(ds.z0.data[10])
        # time_fig = make_subplots(rows=1, cols=1, 
            # shared_xaxes='all', shared_yaxes='all') # [[{"secondary_y": True}]])
        #fig.add_trace(go.Scatter(x=[0,100,200,0], y=[0,200,0,0]))
        file_date_time = get_file_date_time(file)
        include_polygons = False
        if include_polygons:
            assoc_polys = associated[file_date_time]
            for p_idx in assoc_polys:
                # p_idx = assoc_polys[0]
                # make a list of concatenated polygons, separated by blank/plug
                fig.add_trace(go.Scatter(x=df['polygon_x'][p_idx], y=df['polygon_y'][p_idx]))
                trace_count += 1

        fig.add_trace(
            go.Heatmap(
                visible=False,
                x=x, y=y,
                z=np.nan_to_num(ds.DBZ.data[0,z_step], nan=-32), 
                type='heatmap', 
                # colorscale='Viridis',
                colorscale=colorscale_for_go,
                #colorscale=[
                #    [0,   'rgb(0,0,255)'],
                #    [0.5, 'rgb(255,0,255)'],
                #    [1,   'rgb(0,255,0)'],
                #    ],
                #line=dict(color="#00CED1", width=6),
                name="v = " + str(step),
                #zmin=-32, zmax=40,
                zmin=-20, zmax=80,
                ))
        trace_count += 1

        time_step = datetime(int(file[4:8]),int(file[8:10]),int(file[10:12]),
            int(file[13:15]),int(file[15:17]),int(file[17:19]),tzinfo=timezone.utc)
        # >>> fns2.isoformat()
        # '2022-05-21T15:05:50+00:00'

        sbts_key = time_step.isoformat()
        # '2011-11-04T00:05:23'   storms by time step (sbts) keys are this format
        storm_simple_nums_str = sbts[sbts_key]
        # storm_simple_nums = list(map(int, storm_simple_nums_str))
        storm_simple_nums = storm_simple_nums_str
        print("sbts_key: ", sbts_key, " storm_simple_nums: ", storm_simple_nums)
        
        all_storms_t1 = [build_lineage.vol_centroid(s,df_complete) for s in storm_simple_nums]

        # add the storm tracks as arrows for this time step
        # fig.add_trace(go.Scatter(x=df['VolCentroidX(km)'], y=df['VolCentroidY(km)'],mode='lines+markers'))
        # df[(df["SimpleNum"] == 3)] 
        # use keyword None to break parents and children into segments
        # format of coordinates: x|y=[current, child, None, current, child, None ...]
        #child_x, child_y = prepare_child_connections(df_complete, time_step_key, 'child')
        #parent_x, parent_y = prepare_parent_connections(df_complete, time_step_key, 'parent')
        for (simple_num_s, xs, ys) in all_storms_t1:
            # fig.add_trace(go.Scatter(x=[-100,100,None,-100,100], y=[-100,100,None,-100,-100],
            fig.add_trace(go.Scatter(x=xs, y=ys, text=str(simple_num_s), mode="text+lines+markers",
                marker= dict(size=10,symbol= "arrow-bar-up", angleref="previous",
                color="white")))
            trace_count += 1
        map_trace_indexes.append(trace_count)

        #fig.add_trace(time_fig, row=1, col=1, secondary_y=False)
    
    # Make 10th trace visible
    fig.data[10].visible = True
    
    # to make image square 
    fig.update_layout(yaxis_scaleanchor="x")
   
    print("map indexes: ", map_trace_indexes)
 
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
        # if (i % 2 == 1): 
        #    step["args"][0]["visible"][i-1] = True  # Toggle i'th trace to "visible"
        if i in map_trace_indexes:
            file_num = map_trace_indexes.index(i)
            step['label'] = filenames[file_num][13:19]
            end_trace = i
            # TODO: fix up the traces, there are more now, with the storm lineage
            if file_num > 0:
                start_polygon_trace = map_trace_indexes[file_num-1] + 1
            else:
                start_polygon_trace = 1
            for j in range(start_polygon_trace,end_trace+1):
                step["args"][0]["visible"][j] = True  # Toggle associated polygon traces to "visible"
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
    
