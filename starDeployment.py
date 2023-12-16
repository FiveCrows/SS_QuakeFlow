import numpy as np
import utm
import simplekml
R = 2500
n_arms = 8
exDist=200
arm_count = 6
origin = utm.from_latlon(38.850321,-109.985915)#xwgs84
zone = origin[2:]
origin = origin[:2]
rad_off = -np.pi*2*(9/360)
r = np.linspace(0,R,arm_count+1)[1:arm_count+1]
x_vals = list(np.concatenate([np.add(origin[0],r*np.cos(np.pi*2*n/n_arms+rad_off)) for n in range(n_arms)]))
x_vals = x_vals+[np.add(origin[0],exDist*np.cos(2*np.pi*(2*n+1)/n_arms+rad_off)) for n in range(4)]
y_vals = list(np.concatenate([np.add(origin[1],r*np.sin(np.pi*2*n/n_arms+rad_off)) for n in range(n_arms)]))
y_vals = y_vals+[np.add(origin[1],exDist*np.sin(2*np.pi*(2*n+1)/n_arms+rad_off)) for n in range(4)]
utmvals =[origin] + list(zip(x_vals,y_vals)) 

latlon = [utm.to_latlon(val[0],val[1], zone[0],zone[1]) for val in utmvals]
kml = simplekml.Kml()
[kml.newpoint(name = "#{}: arm= {}, pos= {}".format(i,(i-1)//arm_count +1,(i-1)%arm_count+1),coords = [(latlon[i][1],latlon[i][0])]) for i  in range(1,len(latlon)-4)]
[kml.newpoint(name = "#{}:extra".format(i),coords = [(latlon[i][1],latlon[i][0])]) for i  in range(len(latlon)-4,len(latlon))]
kml.newpoint(name = "deployment{}".format(0),coords = [(latlon[0][1],latlon[0][0])]) 

kml.save("starDeployment.kml")