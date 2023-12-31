import re
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import json
import obspy
import requests
import os
#import simplekml (moved to gen kml function)
import laspy
import pandas as pd
from pyproj import Transformer 
from collections import defaultdict
from obspy.core.inventory import Inventory, Network, Station, Channel, Response
from obspy.core import UTCDateTime, AttribDict
from obspy.clients.nrl import NRL
from bisect import bisect
from tqdm import tqdm
import utm
api_openTopography = "143a4c09555886ceb9760b6cf432902c"

class SS_inventory(Inventory):
    def __init__(self, dir = "SS_inventory", fr_subDir = "SS_frTests", net_subDir = "UofUTest2", results_addendum = '_'):
        """Creates an inventory object for smartsolo geophones
        Args:
            dir (str, the directory of the inventory object): _description_. Defaults to "SS_inventory".
            fr_subDir (str, the dir subfolder for files from the smartsolo test rack): . Defaults to "SS_frTests".
            net_subDir (str, optional): the dir  subfolder for   data folders pulled from smartsolos. Defaults to "UofUTest1".
        """

        self.networks=[]
        self.dir = "SS_inventory"
        self.fr_dir=dir+"/"+fr_subDir
        self.net_subDir = net_subDir
        self.loadNet(net_subDir)
        super().__init__(self.networks)

    def loadNet(self,net_subDir):
        """generate a network object to attach on inventory from a netDir using data from folder within inventory dir

        Args:
            netDir (string): Should contain datafiles loaded from any number of smartsolo devices deployed together
        """
        netDir = self.dir+"/"+net_subDir
        self.networks.append(SS_net(netDir,self))


    def pickWithPhasenet():    
        pass
    
    def predWithGamma():
        pass

    def exportStationsXML():
        pass
    
    def rePredictHypoDD():
        pass

    def compileTests(self):
        testFiles = list(filter(lambda x: x.endswith(".xlsx"), os.listdir(self.frDir)))
        initial = True
        df = pd.DataFrame()
        for file in testFiles:
            test = pd.read_excel(self.frDir +"/"+file)
            settings= test.iloc[0,0].split(',')
            sample_rate = float(re.findall(r"[-+]?(?:\d*\.*\d+)", settings[0])[0])#'ms'
            gain = int(re.findall(r"[-+]?(?:\d*\.*\d+)", settings[1])[0])#'db'
            columns = list(test.iloc[2,:])
            if 'Linear' in settings[3]:
                aaFilter = "Linear"
            else:
                aaFilter = "Minimum"
            test = test[3:]
            test.columns = columns
            test["sample_rate"]= sample_rate
            test["gain"]= gain
            test["aaFilter"]= aaFilter
            df = pd.concat([df,test])    
        df.to_csv(self.frDir+"/tests.csv" )
        #df[(df[d.keys()] == d.values()).all(axis=1)] line for getting 

    def extendZNE (self):
        new_nets = self.nets.copy()
        self.network.extendTo_ZNE()

class SS_net(Network):
    """Container for SmartSolo Station objects, extends the obspy Network class 
    Also handles lidar altitude and geospatial transforms"""    
    reg_system = "epsg:4326"#latitude, longitude
    las_system = "epsg:26912"#what opentopology uses for a metric system in SLC
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26912")
    #laspy.read("topography/points.laz")

    def __init__(self,dir, inventory, lidar_pc = "pc.las", location = "NA"):
            self.dir = dir
            self.pc_dir = dir+"/"+lidar_pc
            self.netName = dir.split('/')[-1]
            self.stations = []
            self.inventory = inventory
            for sub in tqdm(os.listdir(dir)):
                if sub.isnumeric():
                    self.includeStation(dir+"/"+sub,inventory.fr_dir)
            #self.pc = laspy.read(self.pc_dir)
            super().__init__(self.netName[-2:],self.stations, alternate_code = self.netName)
            self.miny = min([s.latitude] for s in self.stations) [0]
            self.maxy = max([s.latitude] for s in self.stations) [0]
            self.maxx = max([s.longitude] for s in self.stations)[0]
            self.minx = min([s.longitude] for s in self.stations)[0]
    
    def lazTolas():
        """
        just to decompress laz files and saves into las files that load a little faster if wanted
        """
        las = laspy.read(self.pc_dir)
        las = laspy.convert(las)
        las.write(self.dir+"/pc.las")

    def includeStation(self,dir, fr_dir):
        station = SS_Station(dir,fr_dir, self)
        self.stations.append(station)
    
    def findTopographies(self,format = "PointCloud"):
        """checks opentopography for available topographical datasets in the deployment region

        Args:
            format (str, optional): _description_. Defaults to "PointCloud".

        Returns:
            _type_: _description_
        """
        url_findTops = "https://portal.opentopography.org/API/otCatalog?productFormat={}&minx={}&miny={}&maxx={}&maxy={}&detail=true&outputFormat=json&include_federated=true".format(format,self.minx,self.miny,self.maxx,self.maxy)        
        response = requests.get(url_findTops).json()
        return response
    
    def checkAltSets():
        angle_floats = ["eCompass North", "Tilted Angle", "Roll Angle", "Pitch Angle"]

    def genKML(self):
        import simplekml
        kml = simplekml.Kml()
        for s in net.stations:
            kml.newpoint(name = "SS_{}".format(s.serial_number), coords = [(s.longitude,s.latitude)])
        kml.save(self.dir+"/GPS.kml")

    def compileTests():
        pass  

    def getDistance(self,s1,s2):
        c1 = self.transformer.transform(s1.longitude,s1.latitude)
        c2 = self.transformer.transform(s2.longitude,s2.latitude)
        return c2-c1
    
    def loadStreams_time(self,startTime,endTime):
        """calls loadStream_time for every station in the net
            see net.loadStream_time

        Args:
            startTime (UTC_DateTime): stream start
            endTime (UTC_DateTime): stream end
        """
        
        for station in tqdm(self.stations):
            station.loadStream_time(startTime, endTime)

    def loadStreams_number(self,number):
        """calls loadStream_number for every station in the net

        Args:
            number (int): the stream filenumber
        """
        
        for station in tqdm(self.stations):
            station.loadStream_number(number)

    def write(self,addendum = '_'):
        """writes network to files processable by phasenet and gamma

        Args:
            addendum (str, optional): something distinguish the new folder name with . Defaults to '_'.
        """
        outDir = self.dir+addendum
        fnames = []
        if not os.path.exists(outDir):
            os.mkdir(outDir)                
        for s in self.stations:
            fnames.append(s.writeStream(outDir))
        xmlDir = outDir+"/stations.xml"
        txtDir = outDir+"/stations.txt"
        #if not os.path.exists(xmlDir):
        self.inventory.write(xmlDir,format = "stationxml", validate = True)
        self.inventory.write(txtDir,format = "stationtxt", validate = True)

        fnameCSV = pd.Series([f for f in fnames if f is not None], name = 'fname')        
        fnameCSV.to_csv(self.dir+'_'+'/mseed.csv', index = False)
        
    def plotLogNumericals(self, includeMemory = False):
        refStation = self.stations[0]
        numericSeries = refStation.numericSeries
        if not includeMemory:
            numericSeries = [key for key in numericSeries if refStation.typeKeys[key][0] != "Memory"]
        n = len(numericSeries)        
        factors = [i for i in range(2,n) if n%i == 0]
        if len(factors) == 0:
            n = n+1
            factors = [i for i in range(2,n) if n%i == 0]
        d2 = factors[np.floor(len(factors)/2+0.5).astype(int)]        
        fig,ax = plt.subplots(n//d2,d2)
        for station in self.stations:
            station.plotNumericals((fig,ax),includeMemory = includeMemory)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),loc='outside upper right', fontsize = 16)        
        fig.suptitle(self.stations[0].start_date + " Numerical Data for multiple stations", fontsize = 30)
        plt.tight_layout()
        plt.savefig(self.stations[0].start_date.replace('/','_')+"_{}_logNumericals.png".format(self.netName))    
        plt.show()

    def plotAgainstTemp(self):
        refStation = self.stations[6]
        numericSeries = refStation.numericSeries        
        numericSeries = [key for key in numericSeries if refStation.typeKeys[key][0] != "Memory"]
        key = numericSeries.pop()        
        numericSeries.pop()
        numericSeries.pop(2)
        idx, values = zip(*[p for p in refStation.data[key] if p[0] is not None])
        df = pd.DataFrame(values,idx)
        df.columns = ([key])
        for key in numericSeries:
            idx, values = zip(*[p for p in refStation.data[key] if p[0] is not None])
            df2 = pd.DataFrame(values,idx)
            df2.columns = ([key])    
            df = pd.merge_asof(df,df2, right_index = True, left_index = True)
            #df = df.merge(df2, right_index = True, left_index = True)
            if len((df))>0:
                print(df)
        df.set_index('temperature')
        #df.plot()
        return df

    def axisTo_ZNE(self):
        for s in self.stations:
            sZNE = s.axisTo_ZNE()
            
    def export_gammaCSV(self):
        """export station data to a csv format processable with gamma
        """
        keys = {id, longitude, latitude, elevation, components, response, x, y, z}
        #getVal{}
        pass

        

class SS_Station(Station):
    """
    Extension of obspy Station that reads SmartSolo Files during construction,
    and augments itself with XYZ SS_Channels 
    """

    def __init__(self,dir, fr_dir, network = None):
        """init function

        Args:
            dir (string): A folder containing files from a smartsolo device, 
              should contain a file DigiSolo.LOG, and files seis00**.MiniSeed
            fr_dir (string): A folder containing frequency response files, containing SS test results
        """        

        self.cycleTimes = []        
        self.dir = dir
        self.network = network
        self.fr_dir = fr_dir
        self.readLog(dir+"/DigiSolo.LOG")
        #pass device info from log onto object
        super().__init__(str(self.serial_number)[-5:],self.latitude, self.longitude, self.altitude, alternate_code = str(self.serial_number))
        for key in self.typeKeys:
            if self.typeKeys[key][0] == 'DeviceInfo':
                setattr(self,key,self.data[key][-1][1])
        self.start_date = UTCDateTime(self.data["start_acquisition_filename"][0][0])
        self.n_files  = len(self.data['start_acquisition_filename'])+len(self.data['changed_acquisition_filename'])
        print(self.start_date)
        
        r1 = R.from_euler('XYZ', [self.datStats["roll_angle"]["mean"], self.datStats["pitch_angle"]["mean"], self.datStats["ecompass_north"]["mean"]], degrees=True)
        xyz = {'Y':[1,0,0],'X':[0,1,0],'Z':[0,0,1]}
        dips = {key:-90+np.arccos(np.dot(r1.apply(xyz[key]),[0,0,1]))*180/np.pi for key in xyz}
        azimuths = {key:np.arctan2(r1.apply(xyz[key])[0],r1.apply(xyz[key])[1])*180/np.pi for key in xyz}
        gains = {'X':self.channel_1_gain,'Y':self.channel_2_gain,'Z':self.channel_3_gain}
        self.sample_rate = 100000/self.sample_rate        
        fileTimes = self.data["changed_acquisition_filename"]+self.data["start_acquisition_filename"]
        fileTimes.sort()
        self.fileTimes = fileTimes

        for key in xyz.keys():
            channel = SS_Channel(self, fr_dir, key, dips[key],azimuths[key], gains[key])
            self.channels.append(channel)
        try:
            pass
            #self.set_FRresponse()
        except:
            print("no response")
        print('hello world')

    def readTime(self,line):
       ###to read lines with time in solo logs
       line = re.findall('"([^"]*)"', line)[0]
       return datetime.strptime(line, "%Y/%m/%d,%H:%M:%S")
    
        
    def readFloat(self,line):
        """ read lines with floats in solo logs
        Args:
            line (string): float string
        Returns:
            float: the float from the string
        """       
        return (re.findall("[-+]?(?:\d*\.*\d+)",line))    
    #read smartsolo log file 

    def readLog(self,logDir):
        """reads smartsolo log files from directory
           and stores the information as a dict that detects
           data blocks and keys,
           parses each key = value pairs a structure like {key:[(block time, value)]           
           typeKeys to store each keys block and data type like {key:(key block, data type)}
           and blockTimes as {blockKey:[block times]}

        Args:
            logDir (string): directory must contain a file DigiSolo.LOG
        """

        with open(logDir) as f:
            f = f.readlines()
        blockHeads = [n for n in range(len(f)) if f[n].startswith("[")]
        data = defaultdict(list)
        typeKeys = {}
        readKeys = {"int":       lambda x: int(x),
                    "string":    lambda x: x.replace("\"",""),#remove quotes
                    "doubleList":lambda x: [float(x) for x in s],
                    "double":    lambda x: float(x),
                    "unknown":   lambda x: x
                    }#how to read values for different sorts of keys
        ######build data into organized dicts
        blockTimes = defaultdict(list)
        for i in range(len(blockHeads)):    
            header = re.match(r"([A-za-z]+)", f[blockHeads[i]][1:]  , re.I).group()
            top = blockHeads[i]+1
            blockHeads.append(len(f))
            bot = blockHeads[i+1]    
            splits = [line.split("=")   for line in f[top:bot] if "=" in line]
            splits = {split[0].strip().lower().replace(" ","_").replace("-","_"):split[1].strip() for split in splits}    
            #so time can be seperated as the index 
            if "utc_time" in splits.keys():        
                time = self.readTime(splits.pop("utc_time"))                                    
                blockTimes[header].append(time)
            else:
                time = None
            for (key,value) in splits.items():        
                #identify what sort of value the key pairs to
                if key not in data:        
                    if value.isnumeric():                
                        typeKeys[key] = (header,"int")
                    elif value[0] == "\"":                
                        typeKeys[key] = (header,"string")
                    elif re.match("[-+]?(?:\d*\..*\d+)",value):
                        s = value.split(',')
                        if len(s)>=2:                    
                            typeKeys[key] = (header,"doubleList")
                        else:                    
                            typeKeys[key] = (header,"double")
                    else:
                            typeKeys[key] = (header,"unknown")
                #finally, read and store the value where it belongs
                print()
                parsedValue = readKeys[typeKeys[key][1]](value)
                data[key].append((time, parsedValue))
                
        self.data = data
        self.typeKeys = typeKeys
        self.blockTimes = blockTimes
        #angle_floats = ["eCompass North", "Tilted Angle", "Roll Angle", "Pitch Angle"]
        #pos_floats = ["Longitude", "Latitude", "Altitude"]                                
        #check for possible unwanted gap in time 
        for timeSeries in blockTimes.items():
            diff = np.diff(timeSeries[1])
            mode = stats.mode(diff)
            dtMax = np.argmax(diff)            
        if diff[dtMax]>mode*3:
            
            print("Time gap of {} from".format(diff[dtMax]) +  " detected in {}".format(timeSeries[0]))    
               
        self.numericSeries = [key for key in typeKeys if len(data[key])>4 and (typeKeys[key][1] == 'int' or typeKeys[key][1] == 'double')]
        stdevs = {}
        datStats = {}
        ##### further process numerical data  #remove outliers, get some stats
        for key in self.numericSeries:
            vals = [x[1] for x in data[key]]
            q1 = np.nanpercentile(vals, 25)
            q3 = np.nanpercentile(vals, 75)
            IQR = q3-q1
            lwr_bound = q1-(3*IQR)
            upr_bound = q3+(3*IQR)
            for i in range(len(vals)):            
                if vals[i] < lwr_bound or vals[i] > upr_bound:
                    data[key][i] = (data[key][i][0],None)
            stdevs[key] = np.nanstd(vals)
            datStats[key] = {'mean': np.nanmean(vals),
                             'std': stdevs[key],
                             'mean_err': stdevs[key]/np.sqrt(len(np.array(vals)[~np.isnan(vals)]))
                            }
        #with outliers gone save the data to the object
        self.data = data
        ####Get statistics on calculated data                        
        self.datStats = datStats
        self.latitude = self.datStats['latitude']['mean']
        self.altitude = self.datStats['altitude']['mean']
        self.longitude = self.datStats['longitude']['mean']        
        self.serial_number = data["serial_number"][-1][1]
        #convert to meters
        for key in ["latitude","longitude"]:
                datStats[key]['m_err'] = datStats[key]['mean_err']*111139
                datStats['altitude']['m_err'] = datStats['altitude']['mean_err']    

    #def downSample(self,sample_rate):
    #    """compresses station stream with decimation for bulk of downsampling, 
    #    and then perform an exact resampling with a fourier resample,
    #    down to sample_rate, and updates object parameters accordingly 
#
    #    Args:
    #        sample_rate (int): the desired sample_rate, hz
    #    """
    #    st_sr = self.stream[0].stats.sampling_rate
    #    scale = int(np.floor(st_sr/sample_rate))
    #    if scale >16:
    #        scale = 16
    #    remain = st_sr/sample_rate-scale
    #    print("scale{}".format(scale))
    #    self.stream.decimate(scale)
    #    if remain != 0:
    #        self.stream.resample(sample_rate)
    #    self.sample_rate = sample_rate

    def plotNumericals(self, subplots = None,includeMemory = False, series = None):
        """Plotting tool for numerical time series from ss logs

        Args:
            subplots (_type_, list): pyplot axes list, used to plot whole nets
            includeMemory (bool, optional): whether to include numerical memory 
        """
        ########plot for numerical data        
        #find squarest factors for numberSeries
        # 
        if includeMemory:   
            numericSeries = self.numericSeries
        elif series == None:        
            numericSeries = [key for key in self.numericSeries if self.typeKeys[key][0] != "Memory"]
        else: 
            numericSeries = series

        n = len(numericSeries)
        if subplots == None:
            if n<6:
                factors = [n,1]                        
            elif len(factors) == 0:#in case number is prime
                n == n+1
                factors = [i for i in range(2,n) if n%i == 0]
            else:
                factors = [i for i in range(2,n) if n%i == 0]            
            d2 = factors[np.floor(len(factors)/2+0.5).astype(int)]
            fig,ax = plt.subplots(d2,n//d2)
        else:
            fig,ax = subplots
            d2=ax.shape[1]

        #plot out with that grid
        for i in range(n):
            series = numericSeries[i]
            (x,y) = list(zip(*self.data[series]))    
            axPick = ax[i%d2, i//d2]
            axPick.plot(x,y,label = self.script_name)
            axPick.set_ylabel(series, fontsize = 16)
            axPick.set_xticklabels(axPick.get_xticklabels(),rotation = 30)
                    
        if subplots == None:
            fig.suptitle(self.start_date + " Numerical Data for Station {}".format(self.serial_number))
            plt.savefig(self.start_date.replace('/','_')+"_{}_logPlot.png".format(self.serial_number))    
            plt.tight_layout()                   
            plt.show()
        ########plot for string data
        #textSeries = [key for key in typeKeys if len(dataDict[key])>2 and (typeKeys[key][1] == 'string')]
    
    def loadStream_files(self, fname, dir = None):
        """load Stream by file name, and edit stream metadata to match with station        
        file string may use * to pick up multiple files
        Args:
            fname_start (str) select the files to load the stream for by the starting text.

        Returns: 
            Stream
        """
        if dir == None:
            dir = self.dir
        fdir = dir+"/"+fname     
        print(fdir)  
        stream = obspy.read(fdir)
        for trace in stream:
            id = trace.id.split(".")
            id[0] = self.network.code
            id[1] = self.code
            trace.id = ".".join(id) 
            c = AttribDict({'latitude':self.latitude,
                            'longitude':self.longitude,
                            'elevation':self.altitude})                       
            trace.stats.coordinates = c
            
            
        return stream

    def loadStream_number(self, number):
        """Load SS output Stream by file number

        Args:
            number (int): file number
        Returns:
            None
        """
        nZeros = 3-len(str(number))
        fname = self.dir+"/seis"+"0"*nZeros+"{}*.MiniSeed".format(number)             
        stream = self.loadStream_file(fname)           
        stream.code = number                
        return stream
        
    def findFile_time(self, dateTime):
        times,values = zip(*self.fileTimes)
        times = list(times)        
        file = values[bisect(times,dateTime)-1]        
        return file 
    
    def loadStream_time(self, startTime, endTime):
        """finds and loads stream data at the specified start and and finish time 
        Args:
            startTime (UTC_DateTime): the desired start time of the stream
            endTime (UTC_DateTime): the desired end time of the stream

        Returns:
            _type_: _description_
        """                
        startFile = self.findFile_time(startTime)
        endFile = self.findFile_time(endTime)                                
        stream = self.loadStream_files(startFile)
        if startFile != endFile:
            stream.extend(self.loadStream_files(endFile))
        stream = stream.trim(startTime,endTime)
        return stream
    
    def adjAtt(self,inv):
        self.stream._rotate_to_zne(inv,"ZXY")            

    def axisTo_ZNE(self, ):
        """
        replace XYZ channels with  Z, N, E channels
        Z gets replaced
        would just add new ones but its too slow
        
        """

        replacements = {'Z':'Z','X':'N', 'Y':'E'}
        dips = {'Z':-90,'N':'0','E':0}
        azimuths = {'Z':0,'N':0,'E':90}
        for c in self:                        
            newAx = replacements[c.code[-1]]
            c.code = c.code[:-1] + newAx
            c.dip = dips[newAx]
            c.azimuth = azimuths[newAx]
               
    def loadStream_pick(self,pick, window = 20):
        """A function to load stream time for a given phasenet pick

        Args:
        pick
        window (int): seconds

        """
        startTime = UTCDateTime(pick.phase_time) - timedelta(seconds = window/2)    
        endTime = UTCDateTime(pick.phase_time) +timedelta(seconds = window/2)
        pickStream = self.loadStream_time(startTime,endTime)
        pickStream.phase_index = pick.phase_index
        return pickStream
     
    def loadStream_picks(self, picks, window = 20):
        """efficiently loads and lists every stream for every pick in a dataframe 
           also adds phase_index to stream for reference

        Args:
            picks (DataFrame): a pandas dataframe describing picks, must have phase_time for each pick  
            window (int, optional): _description_. specify region of time around pick to load
        """
        
        picks['file'] = picks['phase_time'].apply(lambda x: self.findFile_time(UTCDateTime(x)))
        pGroups = picks.groupby('file')['phase_time'].apply(list)
        picks_ti = picks.set_index("phase_time")#to find other values with 
        streams = []
        for file,times in tqdm(pGroups.items()):
            stream = self.loadStream_files(file)
            print(stream)            
            for time in times:                
                cut = stream.slice(UTCDateTime(time)-timedelta(seconds = window/2), UTCDateTime(time) + timedelta(seconds = window/2))
                cut.phase_index = picks_ti.loc[time].phase_index
                if len(cut) ==0:
                    print(stream)
                    print(time)
                    print(file)
                streams.append(cut)
        return streams        

    def setFRresponse(self):
        for c in self.channels:
            c.setFR_SStestCSV(self.fr_dir)
            
    def predictPrecision(n_days =35):
        nd_precKey = '{}day_mPrecision'.format(n_days)
        nd_precKey_d = '{}day_degPrecision'.format(n_days)
        d_meas = (max(times[start:])-min(times[start:])).total_seconds()/86400
        for key in pos_floats:
            datStats[key][nd_precKey] = datStats[key]['m_err']*np.sqrt(d_meas/n_days) #35 day error
        for key in angle_floats:
            datStats[key][nd_precKey_d] = datStats[key]['mean_err']*np.sqrt(d_meas/n_days) #35 day error    
        ####params for both plots
        fig,ax = plt.subplots(2)
        barWidth = 0.3
        #####plot position precision
        br1 = np.arange(len(pos_floats))
        heights0 = [datStats[key]['std']/5 for key in pos_floats]
        heights1 = [datStats[key]['m_err'] for key in pos_floats]
        heights2 = [datStats[key][nd_precKey] for key in pos_floats]
        ax[0].bar(br1, heights0, color ='r', width = barWidth, edgecolor ='grey', label ='standard deviation/5')
        ax[0].bar(br1+barWidth, heights1, color ='b', width = barWidth, edgecolor ='grey', label ='dataset precision')
        ax[0].bar(br1+2*barWidth, heights2, color ='g', width = barWidth, edgecolor ='grey', label =("{} day precision").format(n_days))
        ax[0].set_ylabel("meters")
        ax[0].set_xticks(br1 + barWidth, pos_floats)
        ###########for degrees
        br1 = np.arange(len(angle_floats))
        heights0 = [datStats[key]['std']/5 for key in angle_floats]
        heights1 = [datStats[key]['mean_err'] for key in angle_floats]
        heights2 = [datStats[key][nd_precKey_d] for key in angle_floats]
        ax[1].bar(br1, heights0, color ='r', width = barWidth, edgecolor ='grey', label ='standard deviation/5')
        ax[1].bar(br1+barWidth, heights1, color ='b', width = barWidth, edgecolor ='grey', label ='dataset precision')
        ax[1].bar(br1+2*barWidth, heights2, color ='g', width = barWidth, edgecolor ='grey', label =("{} day precision").format(n_days))
        ax[1].set_ylabel("degrees")
        ax[1].set_xticks(br1 + barWidth, angle_floats)
        ax[1].bar
        plt.legend()
        plt.show()

    def writeStream(self,net_dir, stream):
        outDir = os.path.join(net_dir,str(self.serial_number))        
        if not os.path.exists(outDir):
            os.mkdir(outDir)            
        fname = outDir+"/stream{}.MiniSeed".format(stream.code)
        self.write(fname,format = "MSEED")
        return fname

class SS_Channel(Channel):
    """This represents an x,y, or z component of a 3c smartsolo device

    Args:
        Channel (_type_): _description_
    """

    def __init__(self, station, frTest_dir,  axes, dip, azimuth, gain):
        self.TD = 3.31009*10**(-6) #default ,for transfer function
        self.KH=-8.66463*10**-6 # default, for transfer function
        #self.Dd = 0.70722 #default damping from manual
       # self.Gd = 78.54688#default sensitivity from manual
        self.gain = gain        
        self.axes = axes
        self.station = station
        self.aaFilter = self.station.anti_alias_filter_type
        super().__init__(                         
        "LH"+axes,
        location_code = "",
        longitude = station.longitude,
        latitude = station.latitude,
        dip = dip,
        azimuth = azimuth%360,
        elevation = station.elevation,
        sample_rate = station.sample_rate,
        #location = "({},{})".format(station.latitude, station.longitude),
        depth = 0)
        self.setFR_SStestCSV(frTest_dir)

    def setFR_nrl():
            manufacturer = 'DTCC (manufacturers of SmartSolo'
            device = 'SmartSolo IGU-16HR3C'
            preampGain = list(filter(lambda key: ("{} dB".format(gain) in key),NRL().dataloggers[manufacturer][device]))## match preamp gain from NRL
            filterType = 'Linear Phase'
            IIR_Low_Cut = 'Off'
            low_freq = '5 Hz'
            sensorKeys = ['DTCC (manuafacturers of SmartSolo)','5 Hz','Rc=1850', 'Rs=430000']
            dataloggerKeys = [manufacturer, device, preampGain, sampleRate, filterType, IIR_Low_Cut]
            nrl = NRL()
            response = nrl.get_response(
                sensor_keys=sensorKeys,
                datalogger_keys=dataloggerKeys)
            response.plot(min_freq=1E-4, output='DISP')

    def setFR_SStestExcel(self,dir):
        """Reads frequency response data from excel document

        Args:
            dir (_type_): folder containing excel document
        """
        #incomplete
        #testFiles = list(filter(lambda x: x.endswith(".xlsx"), os.listdir('SS_frTests')))
        testExcel = pd.read_excel(dir)
        settings= testExcel.iloc[0,0].split(',')
        #settings = testE
        #make sure file settings match 
        sr = re.findall(r"[-+]?(?:\d*\.*\d+)", settings[0])#'ms'
        gain = re.findall(r"[-+]?(?:\d*\.*\d+)", settings[1])#'db'
        if sr != self.sample_rate:
            print("mismatch")
        
        #find sample rate 

    def setFR_SStestCSV(self,dir):
        tests = pd.read_csv(dir+'/tests.csv')
        d = {"SN": int(self.station.alternate_code), "sample_rate": (1000/self.sample_rate),"gain": self.gain}# "aaFilter": self.aaFilter, 
        matches = tests[(tests[d.keys()] == d.values()).all(axis=1)]#find test with matching parameters
        if len(matches) == 0:
            print("no match!")
            return None
            raise Exception("Error, no matching test!")     
        match = matches.sort_values("Test Time").iloc[-1]#get latest test
        self.sensitivity = match[self.axes + " S.Sensitivity.(V/m/s)"]
        self.damping = match[self.axes + " S.Damping"]
        self.w0 = match[self.axes + " N.Freq (Hz)"]*2*np.pi
        sensitivity = 1000*self.sensitivity*10**(0.05*self.gain)#adjust with gain, also mV to V  
        self.response = Response.from_paz([0,0],self.getPoles()[1:], self.sensitivity)        
        
    def getPoles(self):
        """Assumes the transfer function from SmartSolo, 
        H(s) = s^2/((1+T_D*s)*(s^2+2*T_D*D_d*w_0*s+w_0^2)+KH*s^2   
        default values from manual
        and returns the poles of the transfer function
        Args:
            TD (_type_, optional): _description_. Defaults to 3.31009*10**(-6)
            KH (_type_, optional): _description_. Defaults to -8.66463*10**-6.
            w0 (float, optional): the natural frequency, Defaults to ten pi, 5hz
            Dd (float, optional): Damping. Defaults to 0.70722.
        """
        TD = self.TD        
        Dd = self.damping
        w0 = self.w0
        p3 = TD
        p2 = 1+self.KH+2*TD*Dd*w0
        p1 = TD*(w0**2)+2*Dd*w0
        p0 = w0**2
        return np.roots([p3,p2,p1,p0])

def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])
    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),colors)):                                      
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)
        axis.label.set_color(c)
        axis.line.set_color(c)
        axis.set_tick_params(colors=c)
        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]
        ax.text(*text_plot, axlabel.upper(), color=c,
                va="center", ha="center")
        ax.text(*offset, name, color="k", va="center", ha="center",
            bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})
        
def write_PickStreams(streams):
    """places pickstreams into file, but also logs the fnames in mseed.csv
    Args:
        streams (_type_): streams loaded with loadStream_pick (has a phase index)        
    """








#startTime = UTCDateTime(2023,10,13,16,8,47,1)-timedelta(seconds = 0)
#endTime = UTCDateTime(2023,10,13,17,47,1)+timedelta(seconds = 000)
#if __name__ == "__main__":
if False:
    inv = SS_inventory("SS_inventory")
    net = inv[0]
    net.SSarray.py()
    for s in net.stations:
        for x in tqdm(range(s.n_files)):
            s.loadStream_number(x)     
            s.adjAtt(inv)
            net.write()
            replacements = {'Z':'Z','N':'X', 'E':'Y'}
            for c in s:
                c.code = c.code[:2] +replacements[c.code[2]]
        print(s.start_date)
    df = net.plotAgainstTemp()
    tilt_temp = (df.set_index('temperature')['ecompass_north'])
    
    
    net.plot_response(0.1)
    net.plotLogNumericals()
    net.stations = net.stations[:1]
    net.plotLogNumericals()
    s1 = net.stations[2]
    st = s.loadStream(8)
    st1 = s1.loadStream(4)
    from datetime import timedelta
    t_start = st[0].stats.endtime-timedelta(0,24000)
    t_start2 = st[0].stats.endtime-timedelta(0,10000)
    t_stop = st[0].stats.endtime-timedelta(0,5000)
    st.trim(t_start,t_stop)



inv = SS_inventory(net_subDir = "desert_deployment")
