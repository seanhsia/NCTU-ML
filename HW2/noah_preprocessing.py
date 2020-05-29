import numpy as np
import pandas as pd

data = pd.read_csv("data_noah.csv", encoding='utf-8')
feature_names = list(data)
data = data.drop(['dateStamp','park_sv_id','play_guid','ab_total','ab_count','pitcher_id','batter_id','ab_id','des','type','id','sz_top','sz_bot','zone_location','pitch_con','stand','strikes','balls','p_throws','gid','pdes','spin','norm_ht','inning','pitcher_team','tstart','vystart','ftime','pfx_x','pfx_z','uncorrected_pfx_x','uncorrected_pfx_z','x0','y0','z0','vx0','vy0','vz0','ax','ay','px','pz','pxold','pzold','tm_spin','sb'], axis=1)
data.to_csv("data_Noah_Preprocessing.csv",index_label=False,index=False)