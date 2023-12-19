
from ipyleaflet import Map, Marker, Polygon, basemaps

center = (52.204793, 360.121558)

m = Map(basemap=basemaps.CartoDB.Positron, center=center, zoom=10)

marker = Marker(location=center, draggable=True)
m.add(marker);

display(m)

from netCDF4 import Dataset, num2date

import numpy as np
import xarray as xr

f1 = Dataset("~/data/mdv/mrms_3d/20150626/20150626_230012.mdv.nc")

lats = f1.variables["y0"][:]
lons = f1.variables["x0"][:]
time = f1.variables["time"]

dates = num2date(time[:], time.units)
time_of_day = dates[0].strftime("%Y-%m-%d %H:%M:%S")
print(time_of_day)

dbz = f1.variables["DBZ"]
dbz_dimensions = f1.variables["DBZ"].dimensions
print(dbz_dimensions)
print(dbz[0,18,10,10])
type(dbz)
print(f1.variables.keys())
print(dbz)
print(len(lats), len(lons))

print(lats[0], lons[0])

center = (lats[0], lons[0])



m = Map(basemap=basemaps.CartoDB.Positron, center=center, zoom=10)

marker = Marker(location=center, draggable=True)
m.add(marker);

polysize_x = abs(lats[15] - lats[14]) / 0.5
polysize_y = abs(lons[15] - lons[14]) / 0.5
for ix in range(10, 30, 3):
    for iy in range(10,30, 3):
        x = lats[ix]
        y = lons[iy]
        polygon = Polygon(
            locations=[(x, y), (x+polysize_x, y), (x, y-polysize_y)],
            color="green",
            fill_color="green"
        )
        m.add(polygon);

# color scale using a choropleth layer?
layer = ipyleaflet.Choropleth(
    geo_data=geo_json_data,
    choro_data=unemployment,
    colormap=linear.YlOrRd_04,
    border_color='black',
    style={'fillOpacity': 0.8, 'dashArray': '5, 5'})

m = ipyleaflet.Map(center = (43,-100), zoom = 4)
m.add(layer)


display(m)

