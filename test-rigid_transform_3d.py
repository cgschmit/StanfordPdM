from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator

# Apply axis on the A frame
def rigid_transform_3D(A, B, frame_x, frame_y, margin):
    assert len(A) == len(B)
    N = np.shape(A)[0]; # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.dot(np.transpose(AA),BB)
    U, S, V = np.linalg.svd(H)
    R = np.dot(V.T,U.T)
    recenter=centroid_B-[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    recenter=np.dot(np.linalg.inv(R),recenter)+[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    t=recenter-centroid_A.T
    #t = -R*centroid_A.T + centroid_B.T
    return R, t

# source: ETHZ: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
def weighted_rigid_transform_3D(A, B,weights, frame_x, frame_y, margin):
    #weights should be linked to the distance. The shorter the distance is the higher the value is.
    assert len(A) == len(B)
    N = np.shape(A)[0]
    weight_enlarged=np.matlib.repmat(weights,1,2)
    centroid_A = np.sum(weight_enlarged*A, axis=0)
    centroid_B = np.sum(weight_enlarged*B, axis=0)
    centroid_A = centroid_A / np.sum(weights,axis=0)
    centroid_B = centroid_B / np.sum(weights,axis=0)
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    W = np.diagflat(weights)
    S = np.dot(np.dot(A.T,W),B)
    U,S,V = np.linalg.svd(S)
    M=np.shape(U)[0]
    id_=np.identity(M)
    id_[M-1,M-1]=np.linalg.det(np.dot(V.T,U.T))
    R=np.dot(np.dot(V.T,id_),U.T)
    recenter=centroid_B-[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    recenter=np.dot(np.linalg.inv(R),recenter)+[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    t=-recenter+centroid_A.T
    return R,t

def normalize_array(distance,norm_type="L2", method="inverse"):
    sum_=0
    if norm_type=="L2":
        if method=="euler":
            weights=np.exp(-distance)
            for i in range(np.shape(weights)[0]):
                sum_+=np.power(weights[i],2)
            sum_=np.sqrt(sum_)
            weights_norm=weights/sum_
            return weights_norm
        elif method =="inverse":
            weights=1/distance
            for i in range(np.shape(weights)[0]):
                sum_+=np.power(weights[i],2)
            sum_=np.sqrt(sum_)
            weights_norm=weights/sum_
            return weights_norm
        else:
            print "Wrong method entered. Enter \"inverse\" or \"euler\"."
            return None
    elif norm_type=="L1":
        if method=="euler":
            weights=np.exp(-distance)
            sum_=np.sum(weights)
            weights_norm=weights/sum_
            return weights_norm
        elif method =="inverse":
            weights=1/distance
            sum_=np.sum(weights)
            weights_norm=weights/sum_
            return weights_norm
        else:
            print "Wrong method entered. Enter \"inverse\" or \"euler\"."
            return None
    else:
        print "Wrong Normalization Name Input. Enter \"L1\" or \"L2\"."
        return None

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
    deg=180*rad/3.14
    return deg


A=np.array([[1,2],[1,1],[4,1]])
#B=np.array([[6,2],[6,1],[10,1]])
#B=np.array([[1.2,2],[1.2,1],[4.2,1]])
B=np.array([[6,30],[6,29],[9,29]])

#weight=np.ones((3,1))/3
distances=np.array([[18],[12],[8]])
weight=normalize_array(distances,norm_type="L2", method="inverse")

R_p,t_p=weighted_rigid_transform_3D(A,B,weight,0,0,0)

print "========================="

print "TEST WEIGHTED AFFINE TRANSFORM - 0 degree, trans_x=5, trans_y=28 --> DOES NOT WORK!"

print "R = ",R_p

print "angle = ", rad2deg(np.arctan2(-R_p.item([1][0]),R_p.item([0][0])))

print "t = ",t_p

R,t=rigid_transform_3D(A,B,0,0,0)

print "========================="

print "NO ROTATION - TRANSLATION OF 5PX ON X-AXIS, 28PX ON Y-AXIS"

print "R = ",R

print "angle = ", rad2deg(np.arctan2(-R.item([1][0]),R.item([0][0])))

print "t = ",t

print "========================="

print "ROTATION OF 20 - TRANSLATION OF 5PX ON X-AXIS, 28PX ON Y-AXIS"

# WEIRD COMPORTMENT --> WHEN I SET FRAME_X AND FRAME_Y TO ZEROS, RESIDUAL IS ALMOST EQUAL TO ZERO, 
# WHEN I CHANGE THE VALUE THE RESIDUAL IS GROWING WITH THE VALUE OF FRAME_X AND FRAME_Y

B_10=applyRotationTo(B,0,0,30)

R_10,t_10=rigid_transform_3D(A,B_10,0,0,0)

print "R = ",R_10

print "angle = ", rad2deg(np.arctan2(-R_10.item([1][0]),R_10.item([0][0])))

print "t = ",t_10


print "========================="

print "ROTATION OF 20 - TRANSLATION OF 0"

R_20,t_20=rigid_transform_3D(B,B_10,0,0,0)

print "R = ",R_20

print "angle = ", rad2deg(np.arctan2(-R_20.item([1][0]),R_20.item([0][0])))

print "t = ",t_20


