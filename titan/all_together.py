

import ascii_to_df
import plot_with_timefile_slider_f
import get_polygons_cart


file = "/Users/brenda/data/ams2025/titan/ascii/Tracks2Ascii.derecho.txt"

df = ascii_to_df.ascii_to_df(file)
df.sort_values('date_utc')

df_polys = get_polygons_cart.get_polygons_cart(df)

path = "/Users/brenda/data/ams2025/radar/cart/qc/KingCity/20220521"
z_step = 10

plot_with_timefile_slider_f.plot_with_timefile_slider(z_step, path, df_polys)


#  plot the nodes (vol_centroids)  and arrows (connecting the nodes to show the tracking) on the radar data. Walking the parent child tree. 
