from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator
import math


def affineTransform3d(A, B):
	#Perform getAffineTransform of two vertices to get scale, orientation, translation witht he openCV function
	M=cv2.getAffineTransform(A, B)
	t=np.array([M[0][2],M[1][2]])
	s_x=np.sqrt(np.power(M[0][0],2)+np.power(M[1][0],2))
	s_y=np.sqrt(np.power(M[0][1],2)+np.power(M[1][1],2))
	s=np.array([[s_x,0],[0,s_y]])
	R=np.array([[M[0][0]/s_x,M[0][1]/s_y],[M[1][0]/s_x,M[1][1]/s_y]])
	return R,t,s

def applyRotationTo(points, frame_x, frame_y, x, y, teta, scale=1):
    center_x=x+frame_x/2
    center_y=y+frame_y/2
    teta=deg2rad(teta)
    teta=np.float32(teta)
    print type(teta)
    new_points=np.zeros(np.shape(points),np.float32)
    for i in range(np.shape(points)[0]):
        new_points[i][0]=np.float32((points[i][0]-center_x)*np.cos(teta,dtype=np.float32))-np.float32((points[i][1]-center_y)*np.sin(teta,dtype=np.float32)+center_x)
        new_points[i][1]=np.float32((points[i][0]-center_x)*np.sin(teta,dtype=np.float32)+(points[i][1]-center_y)*np.cos(teta,dtype=np.float32)+center_y)
        print type(new_points[i][0])
    return new_points

def deg2rad(deg):
	rad=deg*3.14/180
	return rad

#Scaling=1.2, translation=0, rotation=0
#WORKS
# A=np.array([(10,10),(15,11),(13,13)],np.float32)
# B=np.array([(12,12),(18,13.2),(15.6,15.6)],np.float32)

#Scaling=1.2, translation=10, rotation=0
#WORKS
# A=np.array([(10,10),(15,11),(13,13)],np.float32)
# B=np.array([(22,22),(28,23.2),(25.6,25.6)],np.float32)

#Scaling=1.2, translation=10, rotation=10
A=np.array([(10,10),(15,11),(13,13)],np.float32)
B=A*1.2 #Scaling of 1.2
B=applyRotationTo(B,0,0,0,0,10) #Rotation of 10
B=B+np.array([(10,10),(10,10),(10,10)],np.float32) #Translation of 10
print B


#Rotation=0
#translation=0

R,t,s=affineTransform3d(A, B)

print "R = ",R
print "angle = ", np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64)*180/3.14
print "s = ",s
print "t = ",t

