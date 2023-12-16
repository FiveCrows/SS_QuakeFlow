from SSarray import *
import pandas as pd
from tqdm import tqdm 
from obspy.signal import cross_correlation as cc
from obspy.signal.array_analysis import get_geometry
from obspy.signal.array_analysis import array_processing
from anytree import Node, RenderTree
vp_max = 8000
vs_max = 4500

#class(geotree):
#    def __init__(geometry):

sampleRate = 1000
inv = SS_inventory("SS_inventory")
net = inv.networks[0]
picks = pd.read_csv("results/picks.csv")
#picks = picks.sort_values('phase_score')
picks = picks[picks["phase_score"]>0.5]
pickStream = obspy.Stream()
#pick_index = picks[-3]
#net.stations = [s for s in net if int(s.sample_rate) >=1000]#only process ones at 1000 hz or larger correlate with later
for p in picks.iterrows():
    for s in tqdm(net):
        path = os.path.join(inv.dir,net.code,s.code+'_')
        fname = str(p[1]["phase_index"])+'.MiniSeed'
        stream = s.loadStream_files(fname,path)
        trace = stream[0]
        time = trace.stats.starttime
        stream.trim(time+9,time+11)
        if s.sample_rate != sampleRate:
            if (s.sample_rate >1000):
                d = int((s.sample_rate//1000)%16)
                stream.decimate(d)        
            stream.resample(1000)
        #for t in stream:
        #    t.id = s.script_name + "_"+t.id
        #trace.copy().decimate(10).decimate(10).spectrogram()
        #pickStream.trim()
        pickStream.extend(stream)
    

#pickStream.select(station = (pickStream[-2]).stats.station)
pickStream2 = pickStream.copy()
pickStream2.filter('lowpass',freq = 50)
s_ref = pickStream2.select(station = pick.station_id.split('.')[1])#pick the station phasenet was picked on 
dt = [cc.xcorr_3c(s_ref,pickStream2.select(station = s.code), 200,full_xcorr=True) for s in net.stations]#find the arrival time differences for those stations 
fig,ax = plt.subplots(len(dt))
for i in range(len(ax)):
    ax[i].plot(range(len(dt[i][2])),dt[i][2])
plt.show()
[len(pickStream.select(station = s.code)) for s in net.stations[1:]]
#plt.plot(range(sDist*2+1),(cross_correlation.correlate(pickStream[-2],pickStream[4],sDist)))
#plt.show()
#cross_correlation.correlate_stream_template(pickStream,pickStream).plot()

geometry = obspy.signal.array_analysis.get_geometry(pickStream.select(component='Z'))#get relative positions in km

sll_x = 1/vp_max
sll_y = 1/vp_max
#obspy.signal.array_analysis.get_timeshift(geometry,sll_x,sll_y,0.0001,100,100)


fig,ax = plt.subplots(len(dt))
for i in range(len(ax)):
    ax[i].plot(range(len(dt[i][2])),dt[i][2])
    
plt.plot(len())
obspy.signal.array_analysis.get_timeshift()