import pandas as pd
from datetime import datetime, timedelta
from gamma import BayesianGaussianMixture, GaussianMixture
from gamma.utils import convert_picks_csv, association, from_seconds, estimate_station_spacing
import numpy as np
from sklearn.cluster import DBSCAN 
from datetime import datetime, timedelta
import os
import json
import pickle
from pyproj import Proj
from tqdm import tqdm
from pyproj import Transformer 
from obspy.signal.util import util_geo_km

############format to match for gamma 
data_dir = lambda x: os.path.join("SS_inventory","t2", x)
stations =  pd.read_csv(data_dir("stations.txt"),delimiter = '|')#read inv output from SSarray
stations.rename(columns = str.lower)
stations.drop(["azimuth",'dip'],axis=1)#channel attributes, not stations 
components = stations.groupby("station")["channel"].apply(lambda x: ','.join([(z[-1]) for z in list(x)]))
response = stations.groupby("station")["Scale"].apply(list)
stations["channel"]=stations["channel"].apply(lambda x: x[:-1])
stations = stations.groupby("station").agg(min).reset_index()
stations["components"] = components
stations["response"] = response
latlon = stations[["longitude","latitude"]]
center = latlon.mean()
stations[["x(km)", "y(km)"]]= pd.Series(zip(*stations.apply(lambda x: util_geo_km(center[0],center[1],x.longitude,x.latitude),axis = 1)))
stations["z(km)"] = stations["levation"]/1000
stations["station"] = stations.station.apply(str)
stations["id"] = stations.fillna('')[["#Network","station","Location","channel"]].apply(".".join, axis=1)
#stations.apply(lambda x: '.'.join(([x["#Network"],str(x.station),x.Location,x.channel[:-1]])),axis=1) 
picks_csv = pd.read_csv(data_dir("picks.csv"))#read picks from phasenet
minmax = latlon.agg([min,max])
minmax.iloc[0] = np.floor(minmax.iloc[0]*10)/10
minmax.iloc[1] = np.ciel(minmax.iloc[0]*10)/10

config = {  'center': list(center),
            'xlim_degree':  list(minmax.longitude),
            'ylim_degree': list(minmax.latitude),
            'degree2km': 111.19492474777779,
            ### setting GMMA configs
            "use_dbscan":True,
            "use_amplitude": True,
            "method":"BGMM",
            "oversample_factor":8,
            "dims": ['x(km)', 'y(km)', 'z(km)'],
            "vel": {"p": 6.0, "s": 6.0 / 1.75}
            }
#if config["method"] == "GMM": ## GaussianMixture
    #config["oversample_factor"] = 1
#earthquake params from gamma example 
config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
config["dims"] = ['x(km)', 'y(km)', 'z(km)']
config["x(km)"] = (np.array(config["xlim_degree"])-np.array(config["center"][0]))*config["degree2km"]*np.cos(np.deg2rad(config["center"][1]))
config["y(km)"] = (np.array(config["ylim_degree"])-np.array(config["center"][1]))*config["degree2km"]
config["z(km)"] = (0, 20)
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # z
    (None, None),  # t
)
# DBSCAN
config["dbscan_eps"] = estimate_station_spacing(stations) / config["vel"]["p"] * 5.0 #s
config["dbscan_min_samples"] = 3
