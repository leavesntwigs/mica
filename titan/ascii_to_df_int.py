import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#import matplotlib.dates as mdates
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#from shapely.geometry import Polygon

# file = "/Users/brenda/data/ams2025/titan/ascii/Tracks2Ascii.derecho.txt"

def ascii_to_df(file):
    
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
            if parent_str == '-':
                parents = []
            else:
                # convert parent_str to list of ints
                parents = [int(str) for str in parent_str.split(",")]  
           
            child_str = parts[poly_count_index - 1]
            if child_str == '-':
                children = []
            else:
                # convert parent_str to list of ints
                children = [int(str) for str in child_str.split(",")]  
           
    
            # Handle missing values marked as "-"
            #parents = int(parent_str) if parent_str != '-' else np.nan
            #children = int(child_str) if child_str != '-' else np.nan
    
            # Polygon values: skip the count, get the next 72 values
            polygon_values = list(map(float, parts[poly_count_index + 1:]))
    
            # Fixed columns
            fixed_cols = parts[:poly_count_index - 2]
    
            # Combine into one row
            row = fixed_cols + [parents, children, polygon_values]
            rows.append(row)
        except Exception as e:
            print("ERROR: skipping line: ", line)
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
    # return df 
    return df
