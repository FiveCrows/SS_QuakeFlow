import re
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import json
import obspy
import requests
import sys
import os
import laspy
import pandas as pd
from pyproj import Transformer 
from obspy.core.inventory import Inventory, Network, Station, Channel, Response
from obspy.clients.nrl import NRL

api_openTopography = "143a4c09555886ceb9760b6cf432902c"

class SS_inventory(Inventory):
    def __init__(self, dir = "SS_inventory", fr_subDir = "SS_frTests", net_subDir = "UofUmockTest1"):
        """Creates an inventory object for smartsolo geophones

        Args:
            dir (str, the directory of the inventory object): _description_. Defaults to "SS_inventory".
            fr_subDir (str, the dir subfolder for files from the smartsolo test rack): . Defaults to "SS_frTests".
            net_subDir (str, optional): the dir  subfolder for   data folders pulled from smartsolos. Defaults to "UofUmockTest1".
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
        self.networks.append(SS_net(netDir,self.fr_dir))

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
        df.to_csv(self.frDir+"/tests.csv")
        #df[(df[d.keys()] == d.values()).all(axis=1)] line for getting 

class SS_net(Network):
    """Container for SmartSolo Station objects extending the obspy Network
    Also handles lidar altitude, and geospatial transforms

    Args:
        Network (_type_): _description_
    """
    reg_system = "epsg:4326"#latitude, longitude
    las_system = "epsg:26912"#what opentopology uses for a metric system in SLC
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26912")
    #laspy.read("topography/points.laz")
    def __init__(self,dir, fr_dir, lidar_pc = "pc.las"):
            self.dir = dir
            self.pc_dir = dir+"/"+lidar_pc
            self.pc = laspy.read(self.pc_dir)
            self.stations = []
            for sub in os.listdir(dir):
                if sub.isnumeric():
                    self.includeStation(dir+"/"+sub,fr_dir)
                
            super().__init__(dir.split('/')[-1],self.stations)
            self.miny = min([s.latitude] for s in self.stations) [0]
            self.maxy = max([s.latitude] for s in self.stations) [0]
            self.maxx = max([s.longitude] for s in self.stations)[0]
            self.minx = min([s.longitude] for s in self.stations)[0]
    
    def lazTolas():
        """
        just to decompress laz files into las files to load a little faster if wanted
        """
        las = laspy.read(self.pc_dir)
        las = laspy.convert(las)
        las.write(self.dir+"/pc.las")


    def includeStation(self,dir, fr_dir):
        station = SS_Station(dir,fr_dir)
        self.stations.append(station)

    def findTopographies(self,format = "PointCloud"):
        url_findTops = "https://portal.opentopography.org/API/otCatalog?productFormat={}&minx={}&miny={}&maxx={}&maxy={}&detail=true&outputFormat=json&include_federated=true".format(format,self.minx,self.miny,self.maxx,self.maxy)        
        response = requests.get(url_findTops).json()
        return response
    def checkAltSets():
        angle_floats = ["eCompass North", "Tilted Angle", "Roll Angle", "Pitch Angle"]

    def plotStations():
        pass

    def compileTests():
        pass  

class SS_Station(Station):
    """
    Extension of obspy Station that reads SmartSolo Files during construction
    """
    def __init__(self,dir, fr_dir):
        """init function

        Args:
            dir (_type_): A folder containing files from a smartsolo device, 
              should contain a file DigiSolo.LOG, and files seis00**.MiniSeed
            fr_dir (_type_): A folder containing frequency responses to lookup data in
        """
        self.cycleTimes = []
        self.readLog(dir+"/DigiSolo.LOG")
        super().__init__(self.serial_number, self.latitude, self.longitude, self.altitude)
        r1 = R.from_euler('XYZ', [self.datStats["Roll Angle"]["mean"], self.datStats["Pitch Angle"]["mean"], self.datStats["eCompass North"]["mean"]], degrees=True)
        xyz = {'X':[1,0,0],'Y':[0,1,0],'Z':[0,0,1]}
        dips = {key:-90+np.arccos(np.dot(r1.apply(xyz[key]),[0,0,1]))*180/np.pi for key in xyz}
        azimuths = {key:np.arctan2(r1.apply(xyz[key])[0],r1.apply(xyz[key])[1])*180/np.pi for key in xyz}
        gains = {'X':self.channel_1_gain,'Y':self.channel_2_gain,'Z':self.channel_3_gain}
        for key in xyz.keys():
            channel = SS_Channel(self, fr_dir, key, dips[key],azimuths[key], gains[key])
            self.channels.append(channel)
    def readTime(self,line):
       ###to read lines with time in solo logs
       line = re.findall('"([^"]*)"', line)[0]
       return datetime.strptime(line, "%Y/%m/%d,%H:%M:%S")
        ### read lines with floats in solo logs
    def readFloat(self,line):
       return (re.findall("[-+]?(?:\d*\.*\d+)",line))    
    #read smartsolo log file 
    def readLog(self,logDir):
        with open(logDir) as f:
            f = f.readlines()
        #read code, channel gains, serial number, start time, when gps syncs
        gpsSyncs = [i for i in range(len(f)) if f[i] == "GPS Status = GPS Synchronization\n"]  #locate gps sync record areas\        
        intKeys = ["Serial Number", "Sample Rate", "Channel 1 Gain", "Channel 2 Gain", "Channel 3 Gain"]            
        self.headInts= {}
        ####read int params and start date from log file
        for i in range(gpsSyncs[0]):
            line = f[i]
            split = line.split(" = ")   
            ## only for lines with one =  
            if len(split) == 2:
                key, value = split[0].strip(), split[1].strip()
            else: continue
            ##load int values and start date to variable
            for intKey in intKeys:
                if intKey in key: 
                    vars(self)[intKey.replace(' ','_').lower()] = int(value)
            if "Start Date" in key:
                self.start_date = value[value.find("(")+1:value.find(")")][1:-1]
            if "Anti-alias Filter" in key:
                self.aaFilter = value
        self.sample_rate = self.sample_rate/100
            ######read gps logs
        #load head ints to object variables

        angle_floats = ["eCompass North", "Tilted Angle", "Roll Angle", "Pitch Angle"]
        pos_floats = ["Longitude", "Latitude", "Altitude"]
        other_floats = ["Sattelite Number"]
        GPS_floats = angle_floats+pos_floats + ["Satellite Number"]
        data = {key:[] for key in GPS_floats}
        for row in gpsSyncs:
            i = 1
            key, value = "UTC Time", self.readTime(f[row+i])
            self.cycleTimes.append(value)
            while f[row+i+1] != "\n":
                i+=1
                key, value = f[row+i].split(" = ")
                key, value = key.strip(), value.strip()
                if key in GPS_floats:
                    if value =="Unknown":
                        data[key].append(np.NaN)
                    else:
                        data[key].append(float(value))  
        #check for possible unwanted gap in time 
        diff = np.diff(self.cycleTimes)
        mode = stats.mode(diff)
        dtMax = np.argmax(diff)
        has_timeGap = diff[dtMax]>mode*3
        if has_timeGap:
            print("Time gap of {} from".format(diff[dtMax]) + self.start_date + " detected in {}".format(ser))
            print()
            start = dtMax+1
        else:
            start = 0
        #remove outliers
        for key in GPS_floats:
            vals = data[key]
            q1 = np.nanpercentile(data[key], 25)
            q3 = np.nanpercentile(data[key], 75)
            IQR = q3-q1
            lwr_bound = q1-(3*IQR)
            upr_bound = q3+(3*IQR)
            for i in range(len(vals)):            
                if vals[i] < lwr_bound or vals[i] > upr_bound:
                    vals[i] = np.nan
        #with outliers gone save the datat to object
        self.data = data
        ####Get statistics on calculated data
        stdevs = {key:np.nanstd(data[key]) for key in GPS_floats}
        datStats = {key:{'mean': np.nanmean(data[key]),
              'std': stdevs[key],
              'mean_err': stdevs[key]/np.sqrt(len(np.array(data[key])[~np.isnan(data[key])]))
              }
        for key in GPS_floats}
        self.GPS_floats = GPS_floats
        self.datStats = datStats
        self.latitude = self.datStats['Latitude']['mean']
        self.altitude = self.datStats['Altitude']['mean']
        self.longitude = self.datStats['Longitude']['mean']
        #convert to meters
        for key in ["Latitude","Longitude"]:
                datStats[key]['m_err'] = datStats[key]['mean_err']*111139
                datStats['Altitude']['m_err'] = datStats['Altitude']['mean_err']

    def plotLog(self):
    ####plot data    
    ###prime factor len(data)
        fig, ax = plt.subplots(4,2, figsize=(10,10))
        for i in range(4):
            for j in range(2):
                ax[i,j].plot(times[start:], data[self.GPS_floats[2*i+j]][start:])
                ax[i,j].set_title(self.GPS_floats[2*i+j])
                ax[i,j].set_xlabel("Time")
                ax[i,j].set_ylabel(self.GPS_floats[2*i+j])
        fig.suptitle(self.startDate + " GPS Data for Station {}".format(self.ser))
        plt.tight_layout()    
        plt.savefig(self.startDate.replace('/','_')+"_{}_logPlot.png".format(self.ser))    

    def adjAtt(self):
        r1 = R.from_euler('XYZ', [datStats["Roll Angle"]["mean"], datStats["Pitch Angle"]["mean"], datStats["eCompass North"]["mean"]], degrees=True)
        xyz = {'X':[1,0,0],'Y':[0,1,0],'Z':[0,0,1]}
        dips = {key:-90+np.arccos(np.dot(r1.apply(xyz[key]),[0,0,1]))*180/np.pi for key in xyz}
        azimuths = {key:np.arctan2(r1.apply(xyz[key])[0],r1.apply(xyz[key])[1])*180/np.pi for key in xyz}
    #####get dips and azimuths of basis vectors under rotation r1
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

class SS_Channel(Channel):
    """This represents an x,y, or z component of a 3c smartsolo device

    Args:
        Channel (_type_): _description_
    """

    def __init__(self, station, frTest_dir,  axes, dip, azimuth, gain):
        self.TD = 3.31009*10**(-6) #default ,for transfer function
        self.KH=-8.66463*10**-6 # default, for transfer function
        self.Dd = 0.70722 #default damping from manual
        self.Gd = 78.54688#default sensitivity from manual
        self.gain = gain
        self.aaFilter = station.aaFilter
        self.axes = axes
        super().__init__(                         
        "LH"+axes,
        location_code = station.code,
        longitude = station.longitude,
        latitude = station.latitude,
        dip = dip,
        azimuth = azimuth%360,
        elevation = station.elevation,
        sample_rate = station.sample_rate,
        depth = 0)
        self.setFR_SStestCSV(frTest_dir)
    def setFR_nrl():
            manufacturer = 'DTCC (manufacturers of SmartSolo'
            device = 'SmartSolo IGU-16HR3C'
            preampGain = list(filter(lambda key: ("{} dB".format(gain) in key),NRL().dataloggers[manufacturer][device]))## match preamp gain from NRL
            filterType = 'Linear Phase'
            IIR_Low_Cut = 'Off'
            low_freq = '5 Hz'

            sensorKeys = ['DTCC (manuafacturers of SmartSolo)','5 Hz','Rc=1850, Rs=430000']
            dataloggerKeys = [manufacturer, device, preampGain, sampleRate, filterType, IIR_Low_Cut]
            nrl = NRL()
            response = nrl.get_response( # doctest: +SKIP
                sensor_keys=['Streckeisen', 'STS-1', '360 seconds'],
                datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])
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
        d = {"SN": int(self.location_code), "sample_rate": self.sample_rate, "aaFilter": self.aaFilter, "gain": self.gain}
        matches = tests[(tests[d.keys()] == d.values()).all(axis=1)]#find test with matching parameters
        if len(matches) == 0:
            raise Exception("Error, no matching test!")        
        match = matches.sort_values("Test Time").iloc[-1]#get latest test
        self.sensitivity = match[self.axes + " S.Sensitivity.(V/m/s)"]
        self.damping = match[self.axes + " S.Damping"]
        self.w0 = match[self.axes + " N.Freq (Hz)"]
        self.response = Response.from_paz([0],self.getPoles(), self.sensitivity)        
        
    def getPoles(self,w0 = 31.4159):
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
        


inv = SS_inventory()
net = inv.networks[0]
