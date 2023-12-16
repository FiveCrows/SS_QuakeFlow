import obspy
from obspy.core.inventory import Inventory, Network, Station, Channel, Site 
from obspy.clients.nrl import NRL


inv = Inventory(
    networks =[],
    source = "FiveCrows"
) 
net = Network(
    code = "XX",
    stations = [],
    description = "A test station"
)

sta = Station(
    code = 'TEST',
    latitude = 40.7666,
    longitude = -111.8460824,
    elevation = 1700.0,
    creation_date = "2023-09-14 20:07:45",
    site = Site(name = "paradox")
)

cha_x = Channel(
    code = "LHE",
    location_code = "",
    latitude = 40.7666,
    longitude = -111.8460824,
    elevation = 1700.0,
    azimuth = 0,
    dip = -90,
    sample_rate = 1000,
    depth = 0)

cha_y = Channel(
    code = "LHN",
    location_code = "",
    latitude = 40.7666,
    longitude = -111.8460824,
    elevation = 1700.0,
    azimuth = 0,
    dip = -90,
    sample_rate = 1000,
    depth = 0)

cha_z = Channel(
    code = "LHZ",
    location_code = "",
    latitude = 40.7666,
    longitude = -111.8460824,
    elevation = 1700.0,
    azimuth = 0,
    dip = -90,
    sample_rate = 1000,
    depth = 0)

#########get response##############
manufacturer = 'DTCC (manufacturers of SmartSolo'
device = 'SmartSolo IGU-16HR3C'
sampleRate = '1000'
preampGain = '18 dB (4)'
filterType = 'Linear Phase'
IIR_Low_Cut = 'Off'
low_freq = '5 Hz'

sensorKeys = ['DTCC (manuafacturers of SmartSolo)','5 Hz','Rc=1850, Rs=430000']
dataloggerKeys = [manufacturer, device, preampGain,sampleRate, filterType, IIR_Low_Cut]
nrl = NRL()
response = nrl.get_response( # doctest: +SKIP
    sensor_keys=sensorKeys,
    datalogger_keys= dataloggerKeys)
cha_x.response = response
cha_y.response = response
cha_z.response = response
sta.channels.append(cha_x)
sta.channels.append(cha_y)
sta.channels.append(cha_z)
net.stations.append(sta)
inv.networks.append(net)
inv.write("staition.xml", format = "stationxml",validate = True)