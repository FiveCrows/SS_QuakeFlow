import numpy as np
import cv2
import statistics
import os

#Hough Circle Detection params#
minDist = 10 #the minimum distance between circles
cannyCircleParam = 500 #a parameter for canny edge  detection
cannyEdgeParam = 5
cannyEdgeParam2 = 5
circRadMax = 10
circRadMin = 7
circSensitivityParam =5
ringWidth = 2
xLineWidth = 3
lineWidth = 1
lineLength =0

###color names
white = (255,255,255)
black = (0,0,0)
red = (0,0,255)
green = (0,255,0)
#image section

topPoint = [849,3768.5]

top = 3700
bottom=14200
left = 150
right = 1050
tl = [np.float32(449), float(3769)]
tr = [np.float32(1034),float(3768)]
br = [np.float32(1046),float(14148)]
bl = [np.float32(461), float(14151)]
yJump = 1287
(os.path.dirname(os.path.abspath(__name__)))

img = cv2.imread( 'dipMeters/dipmeter.jpg')
print(img)
#identify circles


############Find the lines
#cannyImg = cv2.Canny(greyImg,cannyEdgeParam,cannyEdgeParam2)#edge detection to help identify lines

#lines = cv2.HoughLinesP(greyImg,rho = 1,theta = 1*np.pi/180,threshold = 100000,minLineLength = 500,maxLineGap = 100)
#for line in lines: cv2.line(img,(line[0,0],line[0,1]),(line[0,2],line[0,3]),red) 
#find circles

def straightenRegion(img,tl,tr,br,yJump, top,bottom):
        left = int(tl[0])#get left of region
        right = int(max(tr[0],br[0]))#get right of region
        dydx= (tr[1]-tl[1])/(tr[0]-tl[0])# calculate next pixel jump
        adj = 0.5 - dydx*(yJump-tl[0])        #calc next pixel jump
        src = np.array([[tl[0],tl[1]+adj],[tr[0],tr[1]],br],dtype= np.float32)#actual triangle of image 
        dest = np.array([tl,[tr[0],tl[1]],[tr[0],br[1]]],dtype= np.float32)   #to a right triangle
        trans = cv2.getAffineTransform(src,dest)#get transform carry source triangle to destination triangle
        straight =cv2.warpAffine(img[top:bottom,:],trans,(img.shape[1],img.shape[0]),0)#transform
        straight = cv2.threshold(straight,127,255,cv2.THRESH_BINARY)[1][top:bottom,left:right,:]#black or white 
        return straight
        
def hasFill(img,crosspoint):
        x = crosspoint[0]
        y = crosspoint[1]
        t = int(y-0.5)
        b = int(y+0.5)
        l = int(x-0.5)
        r = int(x+0.5)
        blck = [0,0,0]
        
        return (np.sum(img[t,l] +img[t,r]+img[b,l]+img[b,r])<500)                


def getXaxis(img_in):
        left = 0 
        top = 0
        bottom = img_in.shape[0]
        right = img_in.shape[1]
        #image should be single channel!
        try:
                img = cv2.cvtColor(img_in.copy(),cv2.COLOR_BGR2GRAY)#singleChannel
        except:
                img = img_in.copy()
        warp =cv2.threshold(warp,127,255,cv2.THRESH_BINARY)[1]#black or white
        rowSum = [np.sum(img[row,left:right]) for row in range(top,bottom)]#get the amount of white in each row        
        tops = [i+top for i in range(1,len(rowSum)) if (rowSum[i]/rowSum[i-1]>(5/4))]        #find where white suddenly drops
        tops = [tops[i] for i in range(0,len(tops)-1) if tops[i+1]-tops[i]>10]#filter out adjacent values        
        diffs = np.diff(tops)
        mode = statistics.mode(diffs)#get line spacing
        ### remove too small diffs here!
        #get length of lines
        for top in tops():
                i = 1
                
        misses = [(i, int(np.round(diffs[i]/mode))) for i in range(len(tops)-1) if diffs[i]>mode*1.5]#look for missing lines
        
        print(misses)
        for miss in reversed(misses):#insert missing lines                
                i = miss[0]                
                factor = miss[1]
                for n in reversed(range(1,factor)):
                        print(n)
                        tops.insert(i+1,int(np.around(tops[i]+n*diffs[i]/factor)))
        return tops
        #fill missing lines
        #botts = [i+top for i in range(0,len(rowSum)-1) if (rowSum[i]/rowSum[i+1]>(4/3))]      #bottom of line, white suddenly jumps
        #botts = [botts[i] for i in range(0,len(botts)-1) if botts[i+1]-botts[i]>5]#filter out adjacent values


        
def getYaxis(img_in,top,bottom,left,right):
        return getXaxis(cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY).transpose(),left,right,top,bottom)
        
def colorXaxis(img,top,bottom,left,right,color):
        lines = getXaxis(img,top,bottom,left,right)
        for line in lines:
                pass                                        
        return img
        
        
def drawContour(img, c):
        c_img = circleBoxRegion(img,c,20)
        c_imgG = cv2.cvtColor(c_img,cv2.COLOR_BGR2GRAY)#cv2 reads this format
        cnt = cv2.findContours(c_imgG,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[0]
        c_img = cv2.drawContours(c_img, cnt, -1, (0,255,0), 1)  
        cv2.imwrite('countour.jpg',c_img)
        
def circleBoxRegion(img,c,extra = 0):
        x = c[0]
        y = c[1]
        r = c[2]
        
        try:
                img = cv2.cvtColor(img_in.copy(),cv2.COLOR_BGR2GRAY)#singleChannel
        except:
                img = img_in.copy()

        #box around circle
        l = int(np.floor(x-r)-extra)
        right = int(np.ceil(x+r)+extra)
        t = int(np.floor(y-r)-extra)
        b = int(np.ceil(y+r)+extra)
        return img[slice(t,b),slice(l,right)],(x-l,y-t)#return new circle center too


def getCircleAngle(img, c):
        extra=10
        x = c[0]
        y = c[1]
        r = c[2]
        #box around circle
        l = int(np.floor(x-r)-extra)
        right = int(np.ceil(x+r)+extra)
        t = int(np.floor(y-r)-extra)
        b = int(np.ceil(y+r)+extra)
        c_img =img[slice(t,b),slice(l,right)]
        c_imgG = cv2.cvtColor(c_img,cv2.COLOR_BGR2GRAY)#cv2 reads this format
        x = x-l
        y = y-t
        ####draw a series of lines,
        #rotated a pixel around the circle tosee how much overlap it has to the image,
        #pick the angle where line overlaps the least
        angleStep = 1/r
        angle =0
        matchAngle=0
        best = 10000      
        r = r+1#make sure line is out of circle
        while angle<2*np.pi:
                x_frac = np.cos(angle)
                y_frac = np.sin(angle)
                x0 = int(np.around(x+r*x_frac))
                x1 = int(np.around(x+(r+10)*x_frac))
                x12 = int(np.around(x+(r+2)*x_frac))
                y0 = int(np.around(y+r*y_frac))
                y1 = int(np.around(y+(r+10)*y_frac))                
                y12 = int(np.around(y+(r+2)*y_frac))                
                #lined = cv2.line(c_imgG.copy(),(x0,y0),(x1,y1),255,4)
                lined = cv2.line(c_imgG.copy(),(x0,y0),(x12,y12),255,6)
                lined = cv2.line(c_imgG.copy(),(x0,y0),(x1,y1),0,2)
                
                diff = cv2.subtract(c_imgG,lined)
                diff = np.sum(diff**2)
                if diff<best:
                        best=diff
                        matchAngle = angle
                angle=angle+angleStep                
        return matchAngle

def lineToCircle(img,c,angle,length,color):
        x=c[0]
        y=c[1]
        r = c[2]
        x_frac = np.cos(angle)
        y_frac = np.sin(angle)
        x0 = int(np.around(x+r*x_frac))
        x1 = int(np.around(x+(r+length)*x_frac))
        y0 = int(np.around(y+r*y_frac))
        y1 = int(np.around(y+(r+length)*y_frac))                
        #lined = cv2.line(img,(x0,y0),(x1,y1),255,4)
        return cv2.line(img,(x0,y0),(x1,y1),color,3)        

def openCmismatch(img,c):
        x = int(np.around(c[0]))
        y = int(np.around(c[1]))       
        r = c[2]
        box = circleBoxRegion(img,c)
        
        box =cv2.threshold(box,127,255,cv2.THRESH_BINARY)[1]#black or white
        #diff = cv2.subtract(box,cv2.circle(cv2.circle(box,[y,x],r,white,-1),[y,x],r,black,1))#compare to white circle with outer black        
        return diff


img = straightenRegion(img,tl,tr,br,yJump, top,bottom)
greyImg = cv2.cvtColor(straight,cv2.COLOR_BGR2GRAY)#cv2 reads this format
blurImg = cv2.blur(greyImg,(3,3))# to help identify circles
circles = cv2.HoughCircles(blurImg,cv2.HOUGH_GRADIENT,
                        dp=1,
                        minDist = minDist, 
                        param1 = cannyCircleParam,
                        param2 =circSensitivityParam,
                        minRadius = circRadMin, 
                        maxRadius = circRadMax)[0]

for c in circles:
        if hasFill(img,c[:2]):                                
                angle = getCircleAngle(img,c)                
                print('ok')
                img = lineToCircle(img,c,angle,10,red)                
                c = (np.around(c)).astype(int)
                img = cv2.circle(img,(c[0],c[1]),c[2],green,-1)
        else: 
                pass
for x in yLines: img =  cv2.line(img,(x,tops[1]),(x,tops[10]),(255,0,0),2)

cv2.imwrite('warp.jpg',img)