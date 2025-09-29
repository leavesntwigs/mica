

def get_xy(simple_num, df):
    # l = [(1,10),(2,20),(3,30)]
    storm_df = df[df['SimpleNum'] == str(simple_num)][['VolCentroidX(km)','VolCentroidY(km)']]
    if (storm_df.empty):
        print("ERROR, storm not found: ", simple_num)
        return
    v_x = storm_df['VolCentroidX(km)'].to_list()[0]
    v_y = storm_df['VolCentroidY(km)'].to_list()[0]
    return (float(v_x), float(v_y))
    
# >>> get_xy('0',df)
# (-200.502, -83.8344)


# child_x, child_y = prepare_child_connections(df_complete, time_step_key, 'child')
# storms_by_time is a dictionary of storm SimpleNums indexed by time
# e.g.  {Timestamp('2022-05-21 14:59:49+0000', tz='UTC'): ['0', '1', '2', '3'], ...}
def build_lineage(df_complete, storms_by_time, time_step_key, linkage='child'):


    linkage_x = []
    linkage_y = []

    # get the list of storms for the time step
    if time_step_key in storms_by_time:
        storm_simpleNums = storms_by_time[time_step_key]
        # [x] [y] double list comprehension? or just stack and then split?
        #storm_simpleNum 
        #r = all_together.df[all_together.df["SimpleNum"] == '3']
        #vol_centroid_x = 
        [vol_centroid(s,df_complete) for s in storm_simpleNums]
        # linkage_x, linkage_y = vol_centroid('5',df_complete)   # this works
       

# example usage ...
#>>> storms_by_time = d
#>>> time_step_key = '2022-05-21T15:05:50+00:00'
#>>> storm_simpleNums = storms_by_time[time_step_key]
#>>> storm_simpleNums
#['1', '4', '5', '6', '7']
#>>> all_storms_t1 = [build_lineage.vol_centroid(s,df) for s in storm_simpleNums]
#>>> all_storms_t1
#[
#         x's    			y's
#	(('-191.613', -178.721, None), ('-138.109', -129.495, None)), 
#	(('-161.438', -155.47, None),  ('-113.328', -82.7222, None)), 
#	(('-191.546', -183.658, None, '-191.546', -200.502, None, '-191.546', -190.467, None, '-191.546', -182.469, None), ('-82.5627', -85.5091, None, '-82.5627', -83.8344, None, '-82.5627', -124.533, None, '-82.5627', -98.2481, None)), 
#	(('-176.74', -178.721, None), ('-131.12', -129.495, None)), 
#	(('-179.62', -183.658, None), ('-112.254', -85.5091, None))
#]
#
#TODO: what should this structure be? a dictionary{key: time, list of storms, [???
#
#dictionary { key: time, ((s#, c#), (parents), (children)) }  # parents & children are formatted x,y points ready for go.addtrace
#parents/children = trace = (x's, y's)
#x's/y's = (storm centroid_x, parent_centroid_x, None, storm_centroid_x, parent_centroid_x, None, ...)) 
#TODO: associate the trace #'s with the time?  Do in plot function
#
#So, really, it is like a database with time and storm #'s as keys
#The database tables could be data frames?
#time_trace_table (time*, [trace #'s])
#
#time_lineage_segments_table (time*, 



#exploring ...
#
#>>> sbts = bhs.build_helper_structures(df)
#>>> utc1 = df.head(1)['date_utc']
#
#                                 TODO: change bhs to insert these as keys
# 					convert storms to ints
#k1 = utc1.iloc[0].isoformat()    *** need this as a key into the storms_by_time_step dictionary
#
#>>> storms = sbts[k1]
#
#---
#
#>>> k1 = utc1.iloc[0]
#>>> storms = sbts[k1]
#>>> storms
#['0', '1', '2', '3']
#>>> storms.map(int)
#>>> storm_parents = df[df['SimpleNum'] == storms[0]]['parents']
#>>> storm_parents
#0    []
#Name: parents, dtype: object
#>>> storm_children = df[df['SimpleNum'] == storms[0]]['children']
#>>> storm_children
#0    [5]
#
#vol_centroids = [vol_centroid(s,df) for s in storms]

# *** start here ***
def vol_centroid(storm_simple_num, df):


    #storm_df = df[df['SimpleNum'] == storms[0]][['children','parents','VolCentroidX(km)','VolCentroidY(km)']]
    storm_df = df[df['SimpleNum'] == storm_simple_num][['children','parents','VolCentroidX(km)','VolCentroidY(km)']]
    children = storm_df['children'].to_list()[0]
    parents = storm_df['parents'].to_list()[0]
    v_x = storm_df['VolCentroidX(km)'].to_list()[0]
    v_y = storm_df['VolCentroidY(km)'].to_list()[0]

    # ok, this becomes recursive ... we are performing the same operation on all children, all parents, and then appending to lists


    # storm_parents = df[df['SimpleNum'] == storms[0]]['parents']

    # for each storm simple_num, get the centroid of the children
    children_centroid_xy = [get_xy(s,df) for s in children]  # looks like [(vx,vy), (vx2,vy2) ...] 

    # for each storm simple_num, get the centroid of the parents
    parents_centroid_xy = [get_xy(s,df) for s in parents]  # looks like [(vx,vy), (vx2,vy2) ...] 

    # format (storm_x, child1_x, None, storm_x, child2_x, None, ...)  same for parents and same for y
    # where storm_x = (v_x, v_y), child1_x = (x,y), and None = (None, None)

    lc = []
    for p in children_centroid_xy:
        #print("p: ", p)
        lc.append((v_x, v_y))
        lc.append(p)
        lc.append((None,None))

    lp = []
    for p in parents_centroid_xy:
        #print("p: ", p)
        lp.append((v_x,v_y))
        lp.append(p)
        lp.append((None,None))

    l = lc + lp 
    #linkage_x + get_xy()   # [v_x, p_x, None]
    linkage_x, linkage_y = zip(*l)

    # insert parent (x & y), linkage(x & y), None, None, to make segments
    return linkage_x, linkage_y


