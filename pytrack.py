from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkFileDialog import *

def file_save():
    global dist,t 
    fout = asksaveasfile(mode='w', defaultextension=".txt")
    
    for i in range(dist.shape[0]):
        text2save = str(dist[i])+'\t'+str(t[i])
        fout.write("%s\n" % text2save)
        
    fout.close()    

vidcap = cv2.VideoCapture('figure3.avi')
s=0 										#time start
vidcap.set(cv2.CAP_PROP_POS_MSEC,s)  
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps= vidcap.get(cv2.CAP_PROP_FPS)


ret ,frame = vidcap.read() #read the video
rows,cols,ch = frame.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),90-52.38,0.8) #rotate the video to get the roi align with the moving boundary
frame = cv2.warpAffine(frame,M,(cols,rows))


# setup initial location of window
r,h,c,w =218-100,300,433-20,40  
track_window = (c,r,w,h)
r1,h1,c1,w1 =525-10,15,278-5,15  
track_window1 = (c1,r1,w1,h1)
roi = frame[r:r+h, c:c+w] #roi for the boundary
fixed = frame[r1:r1+h1, c1:c1+w1] #roi for the fixed point


#plt.imshow(frame)
#plt.show()
d=[]

while(1):
	ret ,frame = vidcap.read()
	if ret == True:
		frame = cv2.warpAffine(frame,M,(cols,rows))
		res=cv2.matchTemplate(frame, roi, cv2.TM_CCORR_NORMED) #template matching for the roi
		res2=cv2.matchTemplate(frame, fixed, cv2.TM_CCORR_NORMED) #for the fixed point
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		top_left = max_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)
		cv2.rectangle(frame,top_left, bottom_right, 255, 2) #draw rectangle
		
		min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
		top_left2 = max_loc2
		bottom_right2 = (top_left2[0] + w1, top_left2[1] + h1)
		cv2.rectangle(frame,top_left2, bottom_right2, 255, 2) #draw rectangle
		d=d+[top_left2[0]-top_left[0]] #calculate the distance between the two rectangles along the x moving direction
		
		cv2.imshow('img2',frame)
		k = cv2.waitKey(1) & 0xff
	else:
		break
 
dist=np.array(d)

t=np.linspace(s/1000,length/fps,length-s*fps/1000-1) #time scale


file_save() #save distance and time


plt.plot(t,dist,'bo') #plot 
plt.show()

