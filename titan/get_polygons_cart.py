import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

# extract the polygons from the data frame and convert to cartesian coords

# Convert date and time columns to datetime
#df['date_utc'] = pd.to_datetime(
#    df['Year'].astype(str) + '-' +
#    df['Month'].astype(str).str.zfill(2) + '-' +
#    df['Day'].astype(str).str.zfill(2) + ' ' +
#    df['Hour'].astype(str).str.zfill(2) + ':' +
#    df['Min'].astype(str).str.zfill(2) + ':' +
#    df['Sec'].astype(str).str.zfill(2),
#    format='%Y-%m-%d %H:%M:%S',
#    utc=True
#)

def get_polygons_cart(df):

    utc_list = []
    polygons_x_list = []
    polygons_y_list = []

    for idx, row in df.iterrows():
        centroid_x = float(row['VolCentroidX(km)'])
        centroid_y = float(row['VolCentroidY(km)'])
        centroid_z = float(row['VolCentroidZ(km)'])
        rays = row['nPolySidesPolygonRays']
        
        if not rays or len(rays) == 0:
            continue  
        
        angles = np.deg2rad(np.arange(0, 360, 5))  # 72 vertices at every 5 degrees
        rays = np.array(rays, dtype=float) #from centroid to vertex
        
        # Rays in km to degrees
        ray_x = rays * np.cos(angles)
        ray_y = rays * np.sin(angles)
    
        # Approximate conversion from km to degrees lat/lon
        #lat_vertices = lat_centroid + ray_y / 111
        #lon_vertices = lon_centroid + ray_x / (111 * np.cos(np.deg2rad(lat_centroid)))
 
        y_vertices = centroid_y + ray_y
        x_vertices = centroid_x + ray_x

        # add the first point to the end of the list to complete the polygon
        y_vertices = np.append(y_vertices, y_vertices[0])
        x_vertices = np.append(x_vertices, x_vertices[0])
    
        ##polygon_points = list(zip(lon_vertices, lat_vertices))
        #polygon_points = list(zip(x_vertices, y_vertices))
        
        #poly = Polygon(polygon_points)
        #time_idx = np.where(timesteps == row['date_utc'])[0][0]
        time_utc = row['date_utc']
  
        utc_list.append(time_utc) 
        polygons_x_list.append(x_vertices) 
        polygons_y_list.append(y_vertices) 

        #ax.add_geometries([poly], crs=ccrs.PlateCarree(),
        #                  edgecolor=palette[time_idx], facecolor='none', linewidth=1)
        
        #ax.plot(lon_centroid, lat_centroid, marker='o',color='grey', markersize=1.5, transform=ccrs.PlateCarree())
    
      
   
    d = {'utc': utc_list, 'polygon_x': polygons_x_list, 'polygon_y': polygons_y_list}
    df = pd.DataFrame(data=d)  
    
    return df  

    #plt.title("Track Polygons for ComplexNum=0", fontsize=16)
    #plt.show()
    
    
