import obspy
from obspy.core.inventory import Inventory, Network, Station, Channel, Site 
from obspy.clients.nrl import NRL

manufacturer = 'DTCC (manufacturers of SmartSolo'
device = 'SmartSolo IGU-16HR3C'
sampleRate = '1000'
preampGain = '12 dB (4)'
filterType = 'Linear Phase'
IIR_Low_Cut = 'Off'
low_freq = '5 Hz'

inv = Inventory(
    networks =[],
    source = "FiveCrows"
) 
net = Network(
    #creation_date =obspy.UTCDateTime(2023,9,14),
    code = "XX",
    stations = [],
    description = "A test station"
)

sta = Station(
    code = 'TEST',
    latitude = 40.7666,
    longitude = -111.8460824,
    elevation = 1700.0,
    creation_date =obspy.UTCDateTime(2023,5,1),
    site = Site(name = "paradox")
)

cha_x = Channel(
    code = "LHX",
    location_code = "",
    latitude = 40.7666,
    longitude = -111.8460824,
    elevation = 1700.0,
    azimuth = 90,
    dip = 0,
    sample_rate = 1000,
    depth = 0)

cha_y = Channel(
    code = "LHY",
    location_code = "",
    latitude = 40.7666,
    longitude = -111.8460824,
    elevation = 1700.0,
    azimuth = 2,
    dip = -90,
    sample_rate = 1000,
    depth = 0)

cha_z = Channel(
    code = "LHZ",
    location_code = "",
    latitude = 40.7666,
    longitude = -111.8460824,
    elevation = 1700.0,
    azimuth =0,
    dip = 0,
    sample_rate = 1000,
    depth = 0)

#########get response##############
sensorKeys = ['DTCC (manuafacturers of SmartSolo)','5 Hz','Rc=1850, Rs=430000']
dataloggerKeys = [manufacturer, device, preampGain,sampleRate, filterType, IIR_Low_Cut]
nrl = NRL()
response = nrl.get_response( # doctest: +SKIP
    sensor_keys=['Streckeisen', 'STS-1', '360 seconds'],
    datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])
st._rotate_to_zne(inventory = inv,components=("ZXY","123")).plot()
cha_x.response = response
cha_y.response = response
cha_z.response = response
sta.channels.append(cha_x)
sta.channels.append(cha_y)
sta.channels.append(cha_z)
net.stations.append(sta)
inv.networks.append(net)
#inv.write("stations.xml", format = "stationtxt",validate = True)