import plotly
import matplotlib.pyplot as plt
import xarray as xr
import xradar as xd

def get_geocoords(ds):
    """
    Converts Cartesian coordinates (x, y, z) in a radar dataset to geographic
    coordinates (longitude, latitude, altitude) using CRS transformation.

    Parameters
    ----------
    ds : xarray.Dataset
        Radar dataset with Cartesian coordinates.

    Returns
    -------
    xarray.Dataset
        Dataset with added 'lon', 'lat', and 'alt' coordinates and their attributes.
    """
    from pyproj import CRS, Transformer

    # Convert the dataset to georeferenced coordinates
    ds = ds.xradar.georeference()
    # Define source and target coordinate reference systems (CRS)
    src_crs = ds.xradar.get_crs()
    trg_crs = CRS.from_user_input(4326)  # EPSG:4326 (WGS 84)
    # Create a transformer for coordinate conversion
    transformer = Transformer.from_crs(src_crs, trg_crs)
    # Transform x, y, z coordinates to latitude, longitude, and altitude
    trg_y, trg_x, trg_z = transformer.transform(ds.x, ds.y, ds.z)
    # Assign new coordinates with appropriate attributes
    ds = ds.assign_coords(
        {
            "lon": (ds.x.dims, trg_x, xd.model.get_longitude_attrs()),
            "lat": (ds.y.dims, trg_y, xd.model.get_latitude_attrs()),
            "alt": (ds.z.dims, trg_z, xd.model.get_altitude_attrs()),
        }
    )
    return ds


def fix_sitecoords(ds):
    coords = ["longitude", "latitude", "altitude", "altitude_agl"]
    for coord in coords:
        # Compute median excluding NaN
        data = ds[coord].median(skipna=True).item()
        attrs = ds[coord].attrs if coord in ds else {}
        ds = ds.assign_coords({coord: xr.DataArray(data=data, attrs=attrs)})
    return ds

#filename = "hail/2024080600_00_ODIMH5_PVOL6S_VOL_CASSM.h5"
filename = "hail/2024080602_30_ODIMH5_PVOL6S_VOL_CASSM.h5"


# read only DBZH field, one sweep,  for all data files
#
ds = xr.open_dataset(filename, group="sweep_0", engine="odim")
print(ds)

dtree1 = xd.io.open_odim_datatree(filename)
dtree1 = dtree1.xradar.georeference()
print(dtree1["sweep_0"].ds)
try:
    dtree2 = dtree1.xradar.georeference()
except Exception:
    print("Georeferencing failed!")
print(dtree2["sweep_0"].ds)

dtree1 = dtree1.xradar.map_over_sweeps(get_geocoords)
ds = dtree1["sweep_0"].to_dataset()

#
# I think we can do a scatterplot with square markers with cartesian data
#
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ds["DBZH"].plot(x="x", y="y", vmin=-10, vmax=75, cmap="plasma", ax=ax[0])
ds["DBZH"].plot(x="lon", y="lat", vmin=-10, vmax=75, cmap="plasma", ax=ax[1])
plt.show()


