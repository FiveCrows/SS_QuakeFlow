
import re as re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def readTime(line):
    ###to read lines with time in solo logs
    line = re.findall('"([^"]*)"', line)[0]
    return datetime.strptime(line, "%Y/%m/%d,%H:%M:%S")
    ### read lines with floats in solo logs


logDir = 'SS_inventory/UofUmockTest1/453015647/DigiSolo.LOG'
with open(logDir) as f:
    f = f.readlines()
blockHeads = [n for n in range(len(f)) if f[n].startswith("[")]
dataDict = {}#associates keys to values
typeKeys = {}
readKeys = {"int":       lambda x: int(x),
            "string":    lambda x: x.replace("\"",""),#remove quotes
            "doubleList":lambda x: [float(x) for x in s],
            "double":    lambda x: float(x),
            "unknown":   lambda x: x
            }#how to read values for different sorts of keys
######build data into organized dicts
for i in range(len(blockHeads)):    
    header = re.match(r"([A-za-z]+)", f[blockHeads[i]][1:]  , re.I).group()
    top = blockHeads[i]+1
    blockHeads.append(len(f))
    bot = blockHeads[i+1]    
    splits = [line.split("=")   for line in f[top:bot] if "=" in line]
    splits = {split[0].strip().lower().replace(" ","_"):split[1].strip() for split in splits}    
    #so time can be seperated as the index 
    if "utc_time" in splits.keys():        
        time = readTime(splits.pop("utc_time"))
    else:
        time = None
    for (key,value) in splits.items():        
        #identify what sort of value the key pairs to
        if key not in dataDict:        
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
            dataDict[key]  = []
        #finally, read and store the value where it belongs
        else:
            parsedValue = readKeys[typeKeys[key][1]](value)
            dataDict[key].append((time, parsedValue))





