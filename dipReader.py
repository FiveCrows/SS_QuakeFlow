import numpy as np
import math
import cv2
import statistics as stats
import os
import matplotlib.pyplot as plt
import pandas as pd 
#import pytesseract
from scipy.ndimage import convolve
from scipy.signal import find_peaks
import re

###color names
white = (255,255,255)
black = (0,0,0)
red = (0,0,255)
green = (0,255,0)

class dipmeterLog():
      def __init__(self, dir):  
        raw_img = cv2.imread(dir)
        greyImg = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)#cv2 reads this format
        blurDist = 30
        grad =[1/2,0,-1/2]
        k_hLine=np.outer(np.convolve(grad,grad),np.convolve(np.ones(blurDist)/blurDist,np.ones(5)/5))
        h = cv2.threshold(cv2.filter2D(greyImg,ddepth = -1,kernel = k_hLine),100,255,cv2.THRESH_BINARY)[1]
        v = cv2.threshold(cv2.filter2D(greyImg,ddepth = -1,kernel = k_hLine.transpose()),100,255,cv2.THRESH_BINARY)[1]
        #plt.imshow(cv2.morphologyEx(v, cv2.MORPH_CLOSE, np.ones((round(blurDist)))))        
        ######################
        #locate and cutout the grid position
        vSums = list(cv2.reduce(v.transpose(),dim=0,rtype=cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0])
        cut1T = findFirstPeak(vSums,300) 
        cut1B = len(vSums) -findFirstPeak(np.array(list(reversed(vSums))))
        hSums = list(cv2.reduce(h[cut1T:cut1B],dim=0,rtype=cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0])
        cut1L = findFirstPeak(hSums) 
        cut1R = len(hSums) -findFirstPeak(np.array(list(reversed(hSums))))
        cut1 = (slice(cut1T,cut1B),slice(cut1L,cut1R))
        imgCut1 = greyImg[cut1]
        vCut1 = v[cut1]
        hCut1 = h[cut1]

        #display results

        bottom = 4937
        top = 3023
        dist = bottom-top
        y_v = vSums[top:bottom]/max(vSums)
        x_v = np.array(range(dist))*(3/dist)
        y_h = hSums[cut1L:cut1R]/max(hSums)
        x_h = np.array(range(cut1L,cut1R))*(3/h.shape[1])

        dropImg_v = cv2.morphologyEx(v, cv2.MORPH_DILATE, np.ones([3,3]))[top:bottom].transpose()
        dropImg_h = cv2.morphologyEx(h, cv2.MORPH_DILATE, np.ones([3,3]))[top:bottom]
        fig, ax = plt.subplots(2, figsize = (18,12))
        fig.suptitle("Picking Boundaries at the top of a Filtered Tapole Plot", fontsize = 20)
        ax[0].imshow(dropImg_v, extent = (0,3,0,1), cmap = 'binary')
        ax[0].set_ylabel('Percent Max Horizontal Sum',fontsize = 15)
        ax[0].set_xlabel('Vertical Distance (Transposed)',fontsize = 15)
        xy = ((cut1T-top)*3/dist,y_v[cut1T-top])
        ax[0].scatter([xy[0]],[xy[1]])
        ax[0].axhline(y = 0.4, color = 'forestgreen', linestyle = '--',label = 'threshold') 
        ax[0].annotate('Top',xy = xy,xytext =(xy[0]*0.95, xy[1]+0.1), arrowprops = dict(facecolor ='forestgreen', shrink = 0.05),size = 15)
        ax[0].set_xticks([])
        ax[0].plot(x_v, y_v, color = 'firebrick')
        ax[0].legend()
        ax[1].imshow(dropImg_h, extent = (0,3,0,1), cmap = 'binary')
        xy_l = (cut1L*3/h.shape[1], y_v[cut1L])
        xy_r = (cut1R*3/h.shape[1], y_v[cut1R])
        ax[1].annotate('Left', xy = xy_l, xytext =(xy_l[0]*0.95, xy_l[1]+0.1), arrowprops = dict(facecolor ='forestgreen', shrink = 0.05),size = 15)
        ax[1].annotate('Right', xy = xy_r, xytext =(xy_r[0]*1.05, xy_r[1]+0.1), arrowprops = dict(facecolor ='forestgreen', shrink = 0.05),size = 15)
        ax[1].axhline(y = 0.4, color = 'forestgreen', linestyle = '--',label = 'threshold') 
        ax[1].axvline(x = xy[0], color = 'forestgreen', linestyle = '--',label = 'leftbound') 
        ax[1].set_ylabel('Percent Max Vertical Sum',fontsize = 15)
        ax[1].set_xlabel('Horizontal Distance',fontsize = 15)
        ax[1].set_xticks([])
        ax[1].plot(x_h,y_h,color = 'firebrick')
        ax[1].legend(fontsize = 10)
        plt.tight_layout()
        plt.show()

        def getCircleAngle(img, c, extra = 15):
                """getCircleAngle finds the angle of the tail of a circle in an image

                Args:
                img (_type_): the image containing the circle
                c (list): the x,y,r of the circle
                extra (int, optional): _description_. Defaults to 15.

                Returns:
                _type_: _description_
                """
                x = c[0]
                y = c[1]
                r = c[2]        
                #box around circle
                l = int(np.floor(x-r)-extra)
                if l<0:
                        l=0
                right = int(np.ceil(x+r)+extra)
                t = int(np.floor(y-r)-extra)
                b = int(np.ceil(y+r)+extra)
                c_imgG =img[slice(t,b),slice(l,right)]        
                x = x-l
                y = y-t
                ####draw a series of lines,
                #rotated a pixel around the circle tosee how much overlap it has to the image,
                #pick the angle where line overlaps the least
                angleStep = 1/(r+extra)
                angle =0
                matchAngle=0
                best = 10000      
                r = r+1#make sure line is out of circle
                while angle<2*np.pi:
                        x_frac = np.sin(angle)
                        y_frac = -np.cos(angle)
                        x0 = int(np.around(x+r*x_frac))
                        x1 = int(np.around(x+(r+extra)*x_frac))
                        x12 = int(np.around(x+(r+2)*x_frac))
                        y0 = int(np.around(y+r*y_frac))
                        y1 = int(np.around(y+(r+extra)*y_frac))                
                        y12 = int(np.around(y+(r+2)*y_frac))                
                        #lined = cv2.line(c_imgG.copy(),(x0,y0),(x1,y1),255,4)
                        #lined = cv2.line(c_imgG.copy(),(x0,y0),(x12,y12),255,6)
                        lined = cv2.line(c_imgG.copy(),(x0,y0),(x1,y1),0,1)                
                        diff = cv2.subtract(c_imgG,lined)
                        diff = np.sum(diff**2)
                        if diff<best:
                                best=diff
                                matchAngle = angle
                        angle=angle+angleStep                
                return matchAngle

        def openCmismatch(img,c):
                x = int(np.around(c[0]))
                y = int(np.around(c[1]))       
                r = c[2]
                box = circleBoxRegion(img,c)
                box =cv2.threshold(box,127,255,cv2.THRESH_BINARY)[1]#black or white
                #diff = cv2.subtract(box,cv2.circle(cv2.circle(box,[y,x],r,white,-1),[y,x],r,black,1))#compare to white circle with outer black        
                return err
        
        def findFirstPeak(sums,smoothDist = 51):        
                """findFirstPeak finds the first peak in a 1d array of sums

                Args:
                sums (list): a list of sums
                smoothDist (int, optional): the distance to smooth the sums over. Defaults to 50.
                                                                                                                                
                Returns:
                _type_: _description_
                """                                                                                                                                                                     
                sums = np.convolve(sums, np.ones(smoothDist))
                maxSum = sums.max()
                val =0
                pos = 0 
                while val< 0.4*maxSum:
                        val =sums[pos]
                        pos = pos+1
                firstPeak = pos - smoothDist#offset convolution distance
                return firstPeak

        def reinterpolateOutliers(l):
                """A script that detects outliers in a list downmatches and replaces them with nearby values"""
                diffs = np.diff(l)
                for i  in range(len(diffs)):                                                           
                        if abs(diffs[i])>2:
                                print("outlier found in straightening")
                                diffs[i]=0
                return l[0]+np.cumsum(diffs)
                                

        def straighten_v(img_in,img_vf, matchDist_h = 100, return_xShifts = False):   
                """Straighten_v transform detects shift in the columns of vertical lines
                from top to bottom,
                and returns the image with a correction applied

                Args:
                img_in (2d array): The image to be straightened
                img_vf (2d array): image to determine the straightening shifts from, an image filtered for vertical lines
                matchDist_h (int, optional): half The fraction of the image to be used as template 

                Returns:
                2d Array: straightened image, same shape as input image
                """
                matchDist = matchDist_h*2+1
                matchStart = 0
                downMatches = cv2.filter2D(img_vf.astype(float),ddepth=-1,kernel = img_vf[matchStart:matchStart+matchDist])
                xShifts = [np.argmax(row) for row in downMatches]
                xShifts = xShifts-min(xShifts)
                xShifts = reinterpolateOutliers(xShifts)
                maxShift = max(xShifts)
                straight= np.vstack([np.array(list((img_in[i,xShifts[i]: -maxShift+xShifts[i]-1])))for i in range(0, len(xShifts))])
                straight = np.pad(straight,[(0,0),(0,maxShift+1)], constant_values = 255)
                if return_xShifts:
                        return (straight,xShifts)
                else:
                        return straight


        def straighten_h(img_in,img_hf, matchDist_h = 50, return_yShifts = False):   
                """Straighten_h  detects shift in the rows of horizontal lines
                from left to right,
                and returns the image with a correction applied

                Args:
                img_in (2d array): The image to be straightened
                img_hf (2d array): image to determine the straightening shifts from, an image filtered for horizontal lines
                matchDist_h (int, optional): half the portion of the image to be used as template
                return_yShifts (bool, optional): _description_. Defaults to False.

                Returns:
                2d Array: straightened image, same shape as input image
                """                                                                                                                                             
                matchDist = matchDist_h*2+1
                
                img_hfCut= img_hf.transpose()# transpose to make horizontal lines vertical
                downMatches = cv2.filter2D(img_hfCut.astype(float),ddepth=-1,kernel = img_hfCut[:matchDist])#compare template from top to bottom                                                                                                                                                                                                                                                                                         
                yShifts=  [np.argmax(row) for row in downMatches] #find the shift in each column by the best template matchlocation
                yShifts = yShifts-min(yShifts)#make shifts relative
                maxShift = max(yShifts) #find the maximum shift
                straight = np.array([img_in[yShifts[i]: -maxShift+yShifts[i]-1,i] for i in range(0, len(yShifts))]) #apply the shift to each column
                straight = straight.transpose() #transpose back to horizontal
                straight = np.pad(straight,[(0,maxShift+1),(0,0)], constant_values = 255) #pad the image to make up for the shift
                if return_yShifts: #
                        return (straight,yShifts)
                else:                                                                                                                                                                                           
                        return straight

        def straighten(img_in, img_vf, img_hf, matchDist_h = 100, return_shifts = True):
                """Straighten straightens an image by detecting the shift in the columns of vertical lines
                Args:
                img_in (_type_): the image to be straightened
                img_hf (_type_): the image filtered for horizontal lines to predict the shift in the vertical direction
                img_vf (_type_): the image filtered for vertical lines to predict the shift in the horizontal direction
                matchDist_h (int, optional): half the portion of the image to be used as template for matching in the horizontal direction
                """
                if return_shifts:
                        img_in, shifts_v = straighten_h(img_in,img_hf, matchDist_h = matchDist_h,return_yShifts=return_shifts)
                        img_in, shifts_h = straighten_v(img_in,img_vf, matchDist_h = matchDist_h,return_xShifts=return_shifts)    
                        return (img_in, shifts_h, shifts_v)
                else:
                        img_in = straighten_v(img_in,img_vf, matchDist_h = matchDist_h,return_xShifts=return_shifts)
                        img_in = straighten_h(img_in,img_hf, matchDist_h = matchDist_h,return_yShifts=return_shifts)        
                        return img_in


        def applyShifts(img_in, shifts, axis = 0):
                """apply the shifts to an image

                Args:
                img_in (_type_): _description_
                shifts (_type_): _description_
                """
                shifts = reinterpolateOutliers(shifts)
                maxShift = max(shifts) 
                if axis == 0:
                        straight = np.array([img_in[i,shifts[i]: -maxShift-1+shifts[i]] for i in range(len(shifts))])
                        straight = np.pad(straight,[(0,0),(0,maxShift+1)], mode = 'median')
                else: 
                        straight = np.array([img_in[shifts[i]: -maxShift[i]-1+shifts[i],i] for i in range(len(shifts))])
                        straight = np.pad(straight,[(0,maxShift+1),(0,0)], mode = 'median')
                return straight


                                
                [reCircle(straight3, c, canparam = 1, sensScale = 0.5) for c in circles]
                

        def showGridClearing(img_grid,img_clear):
                top = 1600
                bottom = 2100


        def unifyRegions(reg1,reg2):
                """addRegions combines two regions

                Args:
                reg1 (list): the first region
                reg2 (list): the second region

                Returns:
                list: the combined region
                """
                return [(min(r1[0],r2[0]),max(r1[1],r2[1])) for r1,r2 in zip(reg1,reg2)]


        def sub(cut1,cut2):
                """subtracts the inner cut from the outer cut
                Args:
                cut1 (list): a list of slices
                cut2 (list): a list of slices

                Returns:
                list: a list of slices
                """
                outer = cut1
                inner = cut2
                if cut1[0].start>cut2[0].start:
                        outer = cut2
                        inner = cut1
                elif cut1[0].start==cut2[0].start and cut1[1].start>cut2[1].start:
                        outer = cut2
                        inner = cut1
                return [slice(c1.start-c2.start,c1.stop-c2.start) for c1,c2 in zip(inner,outer)]



raw_img = cv2.imread( 'dipMeters/dipmeter.jpg')
greyImg = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)#cv2 reads this format






############identify the vertical gridlines on cut1
straight1, shifts_h,shifts_v = straighten(imgCut1, vCut1, hCut1)
vStraight1 = applyShifts(vCut1, shifts_h)
#############use filtered images to find peaks so that we can separate grids
vSum1 = cv2.reduce(vStraight1,dim=0,rtype=cv2.REDUCE_SUM,dtype=cv2.CV_32S)[0]
vPeaks1 = list(find_peaks(vSum1,height = max(vSum1)/5,width = 1))
peaks_x1 = vPeaks1[0]
diffs = np.diff(peaks_x1)
diffMode_x = stats.mode(diffs)  
gridStart = peaks_x1[0]
maxDiff = max(diffs)
if maxDiff>diffMode_x*4:
        gridStart = np.argmax(diffs)+1
        diffMode_x = stats.mode(diffs[gridStart:])
firstPeak_x1 = peaks_x1[gridStart]
cut2 = (cut1[0],slice(cut1[1].start+firstPeak_x1-diffMode_x,cut1[1].stop))## narrowed to the grid with extra tadpole space on the left      
gridLeft = diffMode_x  # update leftbound to new left for whole image processing later
straight2 = straight1[tuple(sub(cut2,cut1))]
############################get the vertical lines from unfiltered image 
strtSum2 = cv2.reduce(255-straight2,dim=0,rtype=cv2.REDUCE_SUM,dtype=cv2.CV_32S)[0]
#use unfiltered cut to find peaks  with their widths
vPeaks2 = list(find_peaks(strtSum2,height = max(strtSum2)/5,width = 1,distance = diffMode_x*4/5))
peaks_x2 = vPeaks2[0]
##############################identify the horizontal gridlines
strtSum = cv2.reduce(255-straight2.transpose(),dim=0,rtype=cv2.REDUCE_SUM,dtype=cv2.CV_32S)[0]
hPeaks2 = list(find_peaks(strtSum,height = max(strtSum)/5,width = 1))
firstPeak_y2 = hPeaks2[0][0]
cut3 = (slice(cut2[0].start+firstPeak_y2,cut2[0].stop),cut2[1])## cut the space on the top  
straight3 = straight2[tuple(sub(cut3,cut2))]
straight3, shifts_h,shifts_v = straighten(greyImg[cut3], v[cut3], h[cut3],matchDist_h = 50)

vPeaks3 = vPeaks2
hPeaks3 = hPeaks2.copy()
hPeaks3[0] = hPeaks3[0]-firstPeak_y2


noGrid = straight3.copy()
for i in range(len(vPeaks3[0])):
        y = (vPeaks3[0][i])
        dy = int(np.ceil(vPeaks3[1]['widths'][i]*1.5))
        slc = slice(y-dy,y+dy)
        lClose = cv2.morphologyEx(noGrid[:,slc], cv2.MORPH_CLOSE, np.ones((1,dy)))#remove line thinly 
        noGrid[:,slc]=lClose
for i in range(len(hPeaks3[0])):
        x = (hPeaks3[0][i])
        d = int(np.ceil(hPeaks3[1]['widths'][i]*1.5))
        if (x-d) < 0:
                slc = slice(0,x+d)
        else:
                slc = slice(x-d,x+d)
        lClose = cv2.morphologyEx(noGrid[slc], cv2.MORPH_CLOSE, np.ones(d))#restore cirles
        noGrid[slc]=lClose

plotSlice = slice(1600,2100)

cv2.imwrite('straight3ex.jpg',straight3[plotSlice])
cv2.imwrite('noGrid.jpg',noGrid[plotSlice])

plt.imshow(noGrid)
plt.show()

######################
#Hough Circle Detection params#
#################first pass finding filled circles
minDist = 20 #the minimum distance between circles
cannyCircleParam = 10 #a parameter for canny edge  detection
circRadMax = 8
circRadMin = 4
circSensitivityParam = 3
ringWidth = 2

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)) #circle kernel
closeImg = noGrid.copy()
closeImg = cv2.morphologyEx(closeImg, cv2.MORPH_CLOSE, k)#plot_temprestore cirles
#cv2.imwrite('noGridClose.jpg',closeImg[plotSlice])
circles_c = cv2.HoughCircles(closeImg,cv2.HOUGH_GRADIENT,
                        dp=1,
                        minDist = minDist, 
                        param1 = cannyCircleParam,
                        param2 =circSensitivityParam,
                        minRadius = circRadMin, 
                        maxRadius = circRadMax)[0]
#############################to pick any missed circles
[cv2.circle(closeImg, np.around((c[:2])).astype(int),int(c[2])+6,255,-1) for c in circles_c]
#cv2.imwrite('noCirc1.jpg',noCirc[plotSlice])
#cv2.imwrite('noCircClose.jpg',noCirc[plotSlice])
circles_c2 = cv2.HoughCircles(closeImg,cv2.HOUGH_GRADIENT,
                        dp=1,
                        minDist = minDist, 
                        param1 = cannyCircleParam,
                        param2 =circSensitivityParam,
                        minRadius = circRadMin, 
                        maxRadius = circRadMax)[0]
circles_c = np.concatenate((circles_c,circles_c2))
##########get the open circles from an image with no closed circles
noCirc = noGrid.copy()
[cv2.circle(noCirc, np.around((c[:2])).astype(int),int(c[2])+6,255,-1) for c in circles_c]#clear the already detected
#cv2.imwrite('noCirc2.jpg',noCirc[plotSlice])
notCircles =  np.array(list(filter(lambda c: c[0]>250,circles_c)))
[cv2.circle(noCirc, np.around((c[:2])).astype(int),int(c[2])+50,255,-1) for c in notCircles]#clear the already detected
circles_o = cv2.HoughCircles(noCirc,cv2.HOUGH_GRADIENT,
                        dp=1,
                        minDist = minDist, 
                        param1 = cannyCircleParam,
                        param2 =circSensitivityParam+4,
                        minRadius = circRadMin+2, 
                        maxRadius = circRadMax+1)[0]
#################combine the closed and open circles
circles_c =  np.array(list(filter(lambda c: c[0]<300,circles_c)))
circles = np.concatenate((circles_c,circles_o))

##########convert pixel to depth/dip and plot
depth_t = 1340
depth_b = 2350

dfT = pd.DataFrame(circles,columns = ['xPixel','yPixel','R'])

yDiff = dfT["yPixel"].sort_values().diff().mode()[0]#find the y separation between tadpoles


hPeaks = hPeaks3[0]
dfT['upLine'] = [np.argmax(1/(y-hPeaks)) for y in dfT['yPixel']]
dfT['leftLine'] = [np.argmax(1/(x-peaks_x2)) for x in dfT['xPixel']]
#dfT['xPixel'] =( np.around(dfT['xPixel'])).astype(int)
#dfT = dfT.set_index('depth')
dfT['leftLine'] = dfT['leftLine'].replace(40,0)
dfT['depth'] = np.around(dfT.apply(lambda x:depth_t+ 10*(x['upLine']+(x['yPixel']- hPeaks[int(x['upLine'])])/(hPeaks[int(x['upLine'])+1]-hPeaks[int(x['upLine'])])),axis = 1)*2)/2
peaks_x2 = np.append(peaks_x2,straight3.shape[1])
dfT['dip'] = dfT.apply(lambda x: 2*(x['leftLine']+   (x['xPixel']-peaks_x2[int(x['leftLine'])])/(peaks_x2[int(x['leftLine'])+1]-peaks_x2[int(x['leftLine'])])),axis = 1)
l1 = dfT.apply(lambda row: (row.xPixel+row.R*np.sin(row.dip_azimuth*np.pi/180),(row.yPixel-row.R*np.cos(row.dip_azimuth*np.pi/180))),axis = 1)
l2 = dfT.apply(lambda row: (row.xPixel+(row.R+15)*np.sin(row.dip_azimuth*np.pi/180),(row.yPixel-(row.R+15)*np.cos(row.dip_azimuth*np.pi/180))),axis = 1)
hl = pd.read_csv('hardListed.csv')
plt.scatter(hl.depth,hl.dip,label = "source")
plt.scatter(dfT.depth,dfT.dip,marker="x",label = "interpolation")
plt.legend()
plt.show()

#################clear out circles to get dip azimuth
#circShow = straight3.copy()
#[cv2.circle(circShow, np.around((c[:2])).astype(int),int(c[2])+5,0,2) for c in circles]
#cv2.imwrite('circShow.jpg',circShow[plotSlice])         
noCirc = noGrid.copy()

[cv2.circle(noCirc, np.around((c[:2])).astype(int),int(c[2])+2,255,-1) for c in circles]
#cv2.imwrite('noCirc3.jpg',noCirc[plotSlice])
dfT['dip_azimuth'] = dfT.apply(lambda x: (getCircleAngle(noCirc, [x.xPixel,x.yPixel, x.R],extra=14)*180/np.pi),axis=1)
#################
################# check for intersections
#################
dfT = dfT.sort_values('yPixel')
yDiff = -dfT['yPixel'].diff(periods = -1)
yMode = yDiff.mode()
xDiff = -dfT['xPixel'].diff(periods = -1)
dfT['ang_nxt'] =(np.pi-np.arctan(xDiff/yDiff))*180/np.pi
dfT['ang_prev'] = (dfT['ang_nxt'].shift(1)+180)%360
dfT['dist_nxt'] = np.sqrt(yDiff**2+xDiff**2)#distance to next circle)
dfT['dist_prev'] = dfT.dist_nxt.shift(1)
#####get distance of tail to next circle
tailLen = 20
rMean = dfT['R'].mean()+1
####assuming D>R-r 
dfT['cosHit_nxt'] = (dfT.dist_nxt**2+(tailLen+dfT.R)**2-rMean**2)/(2*(dfT.R+tailLen)*dfT.dist_nxt)
dfT['cosHit_prev'] = dfT.cosHit_nxt.shift(1)
dfT['an_diff'] = abs((dfT['ang_nxt']-dfT['dip_azimuth']+180)%360-180)
dfT['ap_diff'] = abs((dfT['ang_prev']-dfT['dip_azimuth']+180)%360-180)
dfT['hitNext'] = dfT.apply(lambda dfT: (np.cos(dfT.an_diff*np.pi/360)>dfT['cosHit_nxt']) | ((dfT.dist_nxt < (dfT.R+tailLen)) and (np.arctan(rMean/dfT.dist_nxt)*180/np.pi)>dfT.an_diff),axis=1)
dfT['hitPrev'] = dfT.apply(lambda dfT: (np.cos(dfT.ap_diff*np.pi/360)>dfT['cosHit_prev']) | ((dfT.dist_prev < (dfT.R+tailLen)) and (np.arctan(rMean/dfT.dist_prev)*180/np.pi)>dfT.ap_diff),axis=1)
noHit = dfT[~(dfT.hitNext | dfT.hitPrev)]

(dfT[dfT.hitPrev])
#plt.imshow(noGrid)
#plt.show()
#### see if there is a possible hit

#dfT['pltAngle'] = 
bothT = pd.merge_asof(dfT.sort_values("depth"),hl.sort_values("depth"),on='depth')
bothT['err'] = abs((bothT.dip_azimuth_x-bothT.dip_azimuth_y+180)%360-180)/180

err = abs(bothT.err).sum()/len(bothT)
plt.scatter(np.log(bothT.err+0.001), bothT.dip_azimuth_y,label = 'source')
plt.scatter(np.log(bothT.err+0.001), bothT.dip_azimuth_x,marker = 'x',label = 'interpolation')
plt.labelsize = 30
plt.titlesize = 50
plt.xlabel('log10 percent error in dip azimuth')
plt.ylabel('dip azimuth')
plt.title("original azimuth record")
#plt.text("avg percent error = {}".format(err))
plt.show()
errCircs = (bothT[bothT.err>0.1][["xPixel","yPixel","R"]])
errLines = (bothT[bothT.err>0.1][["xPixel","yPixel","R"]])
l1 = dfT.apply(lambda row: (round(row.xPixel+row.R*np.sin(row.dip_azimuth*np.pi/180)),round(row.yPixel-row.R*np.cos(row.dip_azimuth*np.pi/180))),axis = 1)
l2 = dfT.apply(lambda row: (round(row.xPixel+(row.R+15)*np.sin(row.dip_azimuth*np.pi/180)),round(row.yPixel-(row.R+15)*np.cos(row.dip_azimuth*np.pi/180))),axis = 1)
c1 = bothT.dropna().apply(lambda row: (round(row.xPixel+(row.R)*np.sin(row.dip_azimuth_y*np.pi/180)),round(row.yPixel-(row.R)*np.cos(row.dip_azimuth_y*np.pi/180))),axis = 1)
c2 = bothT.dropna().apply(lambda row: (round(row.xPixel+(row.R+15)*np.sin(row.dip_azimuth_y*np.pi/180)),round(row.yPixel-(row.R+15)*np.cos(row.dip_azimuth_y*np.pi/180))),axis = 1)

color = cv2.cvtColor(noCirc.copy(),cv2.COLOR_GRAY2RGB)
showErrs = color.copy()
[cv2.circle(showErrs, np.around((c[:2])).astype(int),int(c[2])+5,[255,0,0],2) for c in errCircs.values]
[cv2.line(showErrs, l[0],l[1],[0,0,255],2) for l in zip(l1,l2)]
[cv2.line(showErrs, l[0],l[1],[0,255,0],2) for l in zip(c1,c2)]
plt.imshow(cv2.addWeighted(showErrs,0.5,color,0.5,0))
cv2.imwrite('showErrs.jpg',showErrs)
cv2.imwrite('showErrsCut.jpg',showErrs[plotSlice])







###outdatedCode
#dfT["hasFill"] = dfT.apply(lambda x: hasFill(straightCut,x),axis=1)
#[np.argmax(1/(x-peaks_x2)) for x in dfT['xPixel']]
#diffs = np.diff(df.iloc[:,1].sort_values())#find spacing between circles
#regSpace = stats.mode(diffs) #find the mode of the spacing
#loc = list(filter(lambda i: lines[i].startswith('*'),range(len(lines))))
#update = [re.findall(r'\d+\.\d+',lines[l]) for l in (loc)]###pick out number from text
#depths = depths +[float(depth[0]) for depth in update if len(depth)==1]#convert to float
#filter(lambda i: ex[i].startswith('*'),range(len(ex)))
#depths = []

#depthTag = list(filter(lambda l: l[-2] == 'DEPTH', lines))
#depthTagCenter = (depthTags[0][0][0][0]+depthTags[0][0][1][0])/2
#depthLoc = [i for i in range(len(lines)) if (lines[i][0][0][0]<depthTagCenter and lines[i][0][1][0]>depthTagCenter and lines[i][-2].replace('.','',1).replace(',','',1).replace('/','1',1).isnumeric())]
#d_dip_dist = 
#plt.imshow(erosion)
#plt.imshow(cv2.canny(erosion,50,100,3,False))
#stats.mode(np.diff(df.iloc[:,1].sort_values()))

#def isCircle(c_img,c):
#        """isCircle checks if a circle is a circle
#
#        Args:
#            c_img (2d array): the image of the circle
#            c (tuple): details of the circle (x,y,r)
#        """
#        c_img = cv2.threshold(c_img,127,255,cv2.THRESH_BINARY)[1]# black or white
#        height,width = c_img.shape
#        #get rotated img
#        M = cv2.getRotationMatrix2D((c[0], c[1]), 45, 1.0)
#        rotated = cv2.warpAffine(c_img, M, (height,width))
#        rotated = cv2.threshold(rotated,127,255,cv2.THRESH_BINARY)[1]# black or white
#        c_img = cv2.bitwise_and(c_img, c_img, mask=cv2.circle(rotated.copy(), (round(c[0]),round(c[1])), math.ceil(c[2]),white,-1 ))        
#        #copy rotated image only in circle with mask 
#                #setup mask                                                
#        print(pse(c_img,rotated))


#def getXaxis(rowSums):
#        #image should be single channel!
#        inv_img = 255-img
#        #find line separation distance
#        modeFound = False
#                                 
#        tops = [0]+[i+top for i in range(1,len(rowSum)) if (rowSum[i-1]/rowSum[i]>(1.1))]        #find where white suddenly drops
#        bots = [i+top for i in range(1,len(rowSum)) if (rowSum[i]/rowSum[i-1]>(1.1))]+[bottom]        #find where white suddenly drops        
#        bots = [bots[i] for i in range(0,len(bots)-1) if (bots[i+1]-bots[i])>10]+[bots[-1]]#filter out adjacent values        
#        tops = [tops[0]]+[tops[i] for i in range(1,len(tops)) if tops[i]-tops[i-1]>10]+[bots[-1]]#filter out adjacent values        
#
#        b_diffs = np.diff(bots)
#        t_diffs = np.diff(tops)        
#        t_mode = statistics.mode(t_diffs)#get line spacing
#        b_mode = statistics.mode(b_diffs)#get line spacing
#        #consensus for mode 
#        if t_mode == b_mode:
#                mode = t_mode
#                modeFound =True
#        elif abs(1-t_mode/b_mode)<0.1:
#                mode = int(np.around((t_mode+b_mode)/2))
#                modeFound =True
#        else:
#                if n>10:
#                        print("no mode found")
#                n = n*2              
#                
#        ### remove lines too close to others here, get differences
#        
#        tops = [tops[0]]+[tops[i] for i in range(1,len(tops)-1) if (t_diffs[i-1]*1.1> mode and t_diffs[i]*1.1> mode)]+[tops[-1]]#remove lines that create too small regions
#        bots = [bots[0]]+[bots[i] for i in range(1,len(bots)-1) if (b_diffs[i-1]*1.1> mode and b_diffs[i]*1.1> mode)]+[bots[-1]]#remove lines that create too small regions                        
#        b_diffs = np.diff(bots)
#        t_diffs = np.diff(tops)        
#        t_misses = [(i, int(np.round(t_diffs[i]/mode))) for i in range(len(tops)-1) if t_diffs[i]>mode*1.5]#look for missing lines
#        b_misses = [(i, int(np.round(b_diffs[i]/mode))) for i in range(len(bots)-1) if b_diffs[i]>mode*1.5]#look for missing lines
#        
#        ###replace missing top lines
#        for miss in reversed(t_misses):#insert missing lines                
#                i = miss[0]                
#                factor = miss[1]
#                for n in reversed(range(1,factor)):                        
#                        tops.insert(i+1,int(np.around(tops[i]+n*t_diffs[i]/factor)))
#        #### replace missing bottom lines                
#        for miss in reversed(b_misses):#insert missing lines                
#                i = miss[0]                
#                factor = miss[1]
#                for n in reversed(range(1,factor)):                                
#                        bots.insert(i+1,int(np.around(bots[i]+n*b_diffs[i]/factor)))
#        if len(tops) == len(bots):                
#                return list(zip(tops,bots))
#        else:
#                print("more tops than bots oh no!")


#def pse(img1, img2):
#   h, w = img1.shape
#   img1=(255-img1)/255
#   img2=(255-img2)/255
#   diff = cv2.subtract(img1.astype(int), img2.astype(int))
#   err = np.sum(diff**2)/(np.sum(img2**2))
#   return err


##########################################
####Finding the location, spacing, and rotation of the tadpole grid
##########################################
####find first sum vaue at half or more of max value to identify start of grid

################
#replace line with ramp and find gradient to compute the rotation angle 
#This code has been replaced because lines are not straight, even at an angle, but maybe it'll find a use later 
################
#half_shift = 1
#shift = half_shift*2
#xyScalar = 1
#blur = shift
#original = vCut[:-shift*xyScalar,:-shift].astype(float)
#half_shifted_y =  vCut[half_shift*xyScalar:-half_shift*xyScalar,:-shift]
#half_shifted_x =  vCut[:-shift*xyScalar,half_shift:-half_shift]
##kern = np.array( [[blur- np.sqrt((i - blur)**2+(j-blur)**2) for j in range(blur*2+1)] for i in range(blur*2+1)])
##kern[kern<0] = 0
#kern = np.array([(i+1) for i in reversed(range((blur*2+1)))]).transpose()#linear pyramid
##kern = np.outer(kern,kern)
#convKern = kern/np.sum(kern)
#blurred_x = cv2.filter2D(half_shifted_x.astype(float),ddepth=-1,kernel = convKern)
#blurred_y = cv2.filter2D(half_shifted_y.astype(float),ddepth=-1,kernel = convKern)
##blurred = cv2.filter2D(original.astype(float),ddepth=-1,kernel = convKern)
#blurred=cv2.filter2D(original,ddepth=-1,kernel = convKern)/255
#upShift =   vCut[shift*xyScalar:,:-shift]
#leftShift = vCut[:-shift*xyScalar,shift:]
######################## find slope of vertical lines
#regionScale = np.sum(leftShift)/np.sum(upShift)
#up = sum(np.multiply(blurred_x,(original-upShift)).transpose())
#lef = (sum(np.multiply(blurred,(original-leftShift)).transpose()))
##upDiffs =   list(accumulate(up))
##leftDiffs =   list(accumulate(lef))
##plt.plot(list(accumulate(sum(np.multiply(blurred_x,(original-upShift)).transpose())))) check to see the shifting in the image
#slope = regionScale*sum(up)/sum(lef)/xyScalar
#np.array([[blur*np.sqrt(2.5) - np.sqrt((i - blur+0.5)**2+(j-blur)**2) for j in range(blur*2)] for i in range(blur*2+1)])
#np.vstack([zeros,[[blur- np.sqrt((i - blur+0.5)**2+(j-blur)**2) for j in range(blur*2+1)] for i in range(blur*2)]])
###########rotate image straight and refilter straightened
#
#c_x = round((bottom-top)/2)
#c_y = round((right-left)/2)
#degrees = np.arctan(slope)*180/np.pi
#alpha = np.cos(angle)
#Beta = np.sin(angle)
#R = cv2.getRotationMatrix2D(np.float32([c_y,c_x]),degrees, 1)
#img_rot = cv2.warpAffine(imgCut,R,shape)
#hLineK = np.ones(30)/30
#h = (cv2.threshold(cv2.filter2D(img_rot,ddepth = -1,kernel = hLineK),100,255,cv2.THRESH_BINARY)[1])
#v = (cv2.threshold(cv2.filter2D(img_rot,ddepth = -1,kernel = hLineK.transpose()),100,255,cv2.THRESH_BINARY)[1])
#hStraight = cv2.warpAffine(hCut,R,shape)
#plt.imshow(cv2.merge([straight,imgCut,imgCut])) #plot image with rotation


#########################################
#filter to isolate gridLines
#########################################
### custom convolution kernels
#pk_h = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])/6#prewitt horizontal edge detector
#k_hLine = convolve(pk_h,pk_h)#kernal for detecting horizontal lines
#k_hLine20 = convolve([np.ones(20)/20],k_hLine)
##### filter



#def reCircle(img_in,c,canparam = 1, sensScale = 1):
#        """confirms is a circle and recenters it
#
#        Args:
#            img_in (_type_): the image containing the circle
#            c (list): [x,y,r]
#        """
#        x = c[0]
#        y = c[1]
#        r = c[2]
#        extra = 7
#        l = int(np.floor(x-r-extra))
#        if l<0:
#                l = 0
#        t = int(np.floor(y-r-extra))
#        if t<0:
#                t = 0
#        right = int(np.ceil(x+r+extra))
#        b = int(np.ceil(y+r+extra))
#        c_img = img_in[slice(t,b),slice(l,right)]
#        #c_img = cv2.threshold(c_img,0.5,1,cv2.THRESH_BINARY)[1]# black or white at 1 or 0
#        x = (x-l)
#        y = (y-t)        
#        short_r = math.floor(r-3)
#        filled = cv2.circle(c_img.copy(), (round(x),round(y)),short_r,0,-1)
#        diff = np.sum(cv2.subtract(c_img,filled))
#        isFilled=True
#        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) #circle kernal
#        if diff> short_r**2*np.pi*255*1/2:
#                isFilled = False                
#                c_img = cv2.morphologyEx(c_img, cv2.MORPH_OPEN, k)#restore circles
#        else:
#                c_img =(cv2.morphologyEx(c_img, cv2.MORPH_CLOSE, k))                        
#                c_img = (c_img + (255 - cv2.morphologyEx(c_img, cv2.MORPH_DILATE, k)))#restore cirles)
#        #c_img = np.uint16(c_img)
#        c2 = cv2.HoughCircles(c_img,
#                                cv2.HOUGH_GRADIENT,
#                                dp=1,
#                                minDist = round(r+extra), 
#                                param1 = canparam,
#                                param2 =round(r*sensScale),
#                                minRadius = round(r-2),
#                                maxRadius = round(r+2))
#        try: 
#                c2 = c2[0]
#        except: return None
#        if len(c2)>1:
#                print("more than one circle found oh no!")
#                best =np.argmax([ex[2] for ex in c2])
#                c2 = c2[best]
#        else:
#                c2 = c2[0]
#        dx = c2[0]-x
#        dy = c2[1]-y
#        #cleared = cv2.circle(c_img.copy(), (x,y),short_r,1,-1)
#        #diff = np.sum(cv2.subtract(cleared,c_img))
#        return [c[0]+dx,c[1]+dy,c2[2],isFilled]
#

#def getYaxis(img_in,top,bottom,left,right):
#        return getXaxis(cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY).transpose(),left,right,top,bottom)
#         
#def drawContour(img, c):
#        c_img = circleBoxRegion(img,c,20)
#        c_imgG = cv2.cvtColor(c_img,cv2.COLOR_BGR2GRAY)#cv2 reads this format
#        cnt = cv2.findContours(c_imgG,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[0]
#        c_img = cv2.drawContours(c_img, cnt, -1, (0,255,0), 1)  
#        cv2.imwrite('countour.jpg',c_img)
        
#def circleBoxRegion(img,c,extra = 0):
#        x = c[0]
#        y = c[1]
#        r = c[2]
#        #box around circle
#        l = int(np.floor(x-r)-extra)
#        right = int(np.ceil(x+r)+extra)
#        t = int(np.floor(y-r)-extra)
#        b = int(np.ceil(y+r)+extra)
#        return img[slice(t,b),slice(l,right)],(x-l,y-t,r)#return new circle
#



#############get image with a cleared out grid

#########outdated code for removing lines
#[cv2.line(noGrid,(x,0), (x,len(noGrid)), 255, round(width-0.5)) for x,width in zip(vPeaks3[0],vPeaks3[1]['widths'])]#remove line thinly
#[cv2.line(noGrid,(0,y), (noGrid.shape[1],y), 255, round(width-0.5)) for y,width in zip(hPeaks3[0],hPeaks3[1]['widths'])]#remove line thinly
#midWidth_v = round(stats.median(vPeaks3[1]['widths']))
#midWidth_h = round(stats.median(hPeaks3[1]['widths'])) 
#midWidth = min(midWidth_v,midWidth_h)                                                                                                                                
#noGrid = cv2.morphologyEx(noGrid, cv2.MORPH_OPEN, np.ones((midWidth,midWidth)))#restore circles
#noGrid = cv2.morphologyEx(noGrid, cv2.MORPH_CLOSE, np.ones((1,midWidth_v)))#clean up      
#noGrid = cv2.morphologyEx(noGrid, cv2.MORPH_CLOSE, np.ones(midWidth_h))#clean up 

#noGrid = cv2.morphologyEx(noGrid, cv2.MORPH_OPEN, np.ones((4)))#restore circles
#noGrid = cv2.morphologyEx(noGrid, cv2.MORPH_ERODE, np.ones((2,2)))#restore circles
#plt.imshow(noGrid)
#plt.show()
######################


#def morphoImage(img_in, erode_kern, close_kern):
#        """plots a close followed by an erosion
#
#        Args:
#            img_in (_type_): _description_
#            erode_kern (_type_): _description_
#            dilate_kern (_type_): _description_
#        """
#        eroded = cv2.erode(img_in,erode_kern)
#        closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, close_kern)
#        plt.imshow(closed)
#        plt.show()
#        return closed

