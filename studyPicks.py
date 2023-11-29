from SSarray import *
import pandas as pd

inv = SS_inventory("SS_inventory")
net = inv.networks[0]
picks = pd.read_csv("results/picks.csv")
picks = picks.sort_values('phase_amplitude')
picks = picks[picks.phase_score>0.5]
#have an inventory that can find streams with ZNE names
#but keep XYZ inventory to rotate incoming streams
inv_ZNE = inv.copy()
inv_ZNE[0].axisTo_ZNE()
netDir = os.path.join("SS_inventory",net.code)
if not os.path.exists(netDir):
    os.makedirs(netDir)
fname_xml = os.path.join(netDir,"stations.xml")
inv_ZNE.write(fname_xml, format="stationxml", validate=True)
pickWindow = 20
for station in tqdm(net.stations):
    pickStreams = station.loadStream_picks(picks)    
    for s in pickStreams:
        s._rotate_to_zne(inv,"ZXY")
        s.filter('lowpass', freq = 25, zerophase = True)
    #net = pickStreams[0][0].stats['network']    
    df_dir = os.path.join(netDir,"mseed.csv")
    if os.path.isdir(df_dir):
        df = pd.read_csv(df_dir)
    else:
        print("newDir")
        df = pd.DataFrame(columns = ["fname", "phase_index","start_time"])    
    for stream in pickStreams:         
        id = stream[0].id
        net_id,num,loc = id.split('.')[:-1]
        outDir = os.path.join(netDir, num+"_"+loc)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        fname = os.path.join(netDir, num+"_"+loc,"{}.MiniSeed".format(stream.phase_index))
        stream.write(fname,format = 'mseed')
        df = df._append({'fname':fname,
                         "phase_index": stream.phase_index,
                         "pickWindow": pickWindow,
                         "start_time":stream[0].stats["starttime"]},ignore_index=True)
    df.to_csv(df_dir,index = False)
    
#pickStreams[75].copy().decimate(10).spectrogram()
#inv = SS_inventory()
#net = inv.networks[0]

