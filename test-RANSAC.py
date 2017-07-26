from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator
import math

def affineTransform(A, B, frame_x=600, frame_y=600,margin=0):
	assert len(A) == len(B)
	N = A.shape[0]
	centroid_A = np.mean(A, axis=0, dtype=np.float64)
	centroid_B = np.mean(B, axis=0, dtype=np.float64)
	AA = A - np.tile(centroid_A, (N, 1))
	BB = B - np.tile(centroid_B, (N, 1))
	H = np.dot(np.transpose(AA),BB)
	U, S, Vt = np.linalg.svd(H)
	R = np.dot(Vt.T,U.T)
	recenter=centroid_B-[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
	#recenter=np.dot(np.linalg.inv(R),recenter)+[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
	recenter=np.dot(np.dot(U.T,Vt.T),recenter)+[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
	t=-recenter+centroid_A.T
	return R, t

def applyRotationTo(points, frame_x, frame_y, teta, x=0, y=0, scale=1):
	center_x=x+frame_x/2
	center_y=y+frame_y/2
	teta=deg2rad(teta)
	# Rotation Matrix
	new_points=np.zeros(np.shape(points))
	for i in range(np.shape(points)[0]):
		new_points[i][0]=(points[i][0]-center_x)*np.cos(teta)-(points[i][1]-center_y)*np.sin(teta)+center_x
		new_points[i][1]=(points[i][0]-center_x)*np.sin(teta)+(points[i][1]-center_y)*np.cos(teta)+center_y
	return new_points

def deg2rad(deg):
	rad=deg*3.14/180
	return rad

def rad2deg(rad):
	deg=rad*180/3.14
	return deg

A=np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]],np.float64)#,[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],[19,19],[20,20],[21,21],[22,22],[23,23],[24,24],[25,25],[26,26],[27,27],[28,28],[29,29]],np.float64)
B=np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]],np.float64)#,[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],[19,19],[20,20],[21,21],[22,22],[23,23],[24,24],[25,25],[26,26],[27,27],[28,28],[29,29],[30,30]])


B=applyRotationTo(A, 0, 0, 10, x=0, y=0, scale=1)
B=B+np.tile([1,1],(np.shape(B)[0],1))
print B

for i in range(5): #BEWARE THE NUMBER 30 IS TO BE CALCULATED!!!!!!!
	### Select randomly 3 features
	print "ITERATION NUMBER = ", i
	#Create array of index and select the three first value of the shuffled array
	idx=np.asarray(list(range(np.shape(A)[0])))
	np.random.shuffle(idx)
	idx=idx[:3]
	A_rand=A[idx]
	B_rand=B[idx]
	print idx

	### Select all the other features
	mask=np.ones(np.shape(A)[0],np.bool)
	mask[idx]=0
	idx_test=np.asarray(list(range(np.shape(A)[0])))
	idx_test=idx_test[mask]
	#USELESS TO SHUFFLE
	#np.random.shuffle(idx_test)
	A_test=A[idx_test]
	B_test=B[idx_test]

	### Call affineTransform
	R,t=affineTransform(A_rand,B_rand)
	print "R = ",R
	print "t = ",t
	print "angle = ",rad2deg(np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64))

	#B_estimated=np.dot(R,A_test.T)-np.tile(t.T,((np.shape(A_test)[0]),1)).T
	B_estimated=np.dot(np.linalg.inv(R),A_test.T)-np.tile(t.T,((np.shape(A_test)[0]),1)).T
	#B_estimated=np.dot(A_test,R)+np.tile(t,(np.shape(A_test)[0],1))
	B_estimated=B_estimated.T


	# print B_estimated
	# print np.shape(B_estimated)

	### Calculate distance from point B_bar to B_estimated, if within a predefined distance, include as inlier,if not outlier. Count inlier.
	#print B_test[:,0]
	#print "###################"
	#print B_estimated[:,0]
	distances=np.sqrt(np.power(B_test[:,0]-B_estimated[:,0],2)+np.power(B_test[:,1]-B_estimated[:,1],2))
	#print distances
	total_distance=np.sum(distances)
	print "TOTAL DISTANCE >>>>>>>>>>>>>>>>> ",total_distance,"<<<<<<<<<<<<<<<<<"
	#print distances
	#Compate B_bar and B_estimated
	#count the maximum number of point within a predefined radius!



