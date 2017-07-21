from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator


def local2global(local_x, local_y, global_teta=0, global_position_frame_x=0, global_position_frame_y=0):
    # local_x/y: position of feature in frame referential
    # global_teta: total angle of the frame (global referential)
    # position_frame_x/y: position of the top-left corner of the frame regarding the total displacement (global referential)
    global_x=global_position_frame_x+np.sqrt(np.power(local_x,2)+np.power(local_y,2))*np.sin(np.arctan2(local_x,local_y)-deg2rad(global_teta))
    global_y=global_position_frame_y+np.sqrt(np.power(local_x,2)+np.power(local_y,2))*np.cos(np.arctan2(local_x,local_y)-deg2rad(global_teta))
    return global_x, global_y

def global2local(global_x, global_y, global_teta=0, global_position_frame_x=0, global_position_frame_y=0):
    # global_x/y: position of feature in global referential
    # global_teta: total angle of the frame (global referential)
    # global_position_frame_x/y: position of the top-left corner of the frame regarding the total displacement (global referential)
    global_x_frame=global_x-global_position_frame_x
    global_y_frame=global_y-global_position_frame_y
    # local_x=np.sqrt(np.power(global_x_frame,2)+np.power(global_y_frame,2))
    # print local_x
    # angle=global_teta+np.arctan2(global_y_frame,global_x_frame)
    # local_x=local_x*np.cos(angle)
    local_x=np.sqrt(np.power(global_x_frame,2)+np.power(global_y_frame,2))*np.cos(deg2rad(global_teta)+np.arctan2(global_y_frame,global_x_frame))
    local_y=np.sqrt(np.power(global_x_frame,2)+np.power(global_y_frame,2))*np.sin(deg2rad(global_teta)+np.arctan2(global_y_frame,global_x_frame))
    return local_x, local_y


def formatting(tupleTab):
    totalTab=[]
    for i in range(np.size(tupleTab)):
        tab = [[np.float32(tupleTab[i].pt[0]), np.float32(tupleTab[i].pt[1])]]
        totalTab.append(tab)
    myarray = np.asarray(totalTab)
    return myarray

def array2tuple(array1D,ret_int):
    if ret_int==True:
        return (int(array1D[0]),int(array1D[1]))
    else:
        return (array1D[0],array1D[1])

def tuple2array(tup_le,ret_int):
    if ret_int==True:
        arr_ay=[int(tup_le[0]),int(tup_le[1])]
    else:
        arr_ay=[tup_le[0],tup_le[1]]
    return arr_ay

def normalize_array(distance,norm_type="L2", method="inverse"):
    sum_=0
    if norm_type=="L2":
        if method=="euler":
            weights=np.exp(-distance/100)
            for i in range(np.shape(weights)[0]):
                sum_+=np.power(weights[i],2)
            sum_=np.sqrt(sum_)
            weights_norm=weights/sum_
            return weights_norm
        elif method =="inverse":
            # TO AVOID SINGULARITY WITH DISTANCE==0, WE ADDED +0.05
            weights=1/(distance+0.05)
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
            weights=np.exp(-distance/100)
            sum_=np.sum(weights)
            weights_norm=weights/sum_
            return weights_norm
        elif method =="inverse":
            weights=1/(distance+0.05)
            sum_=np.sum(weights)
            weights_norm=weights/sum_
            return weights_norm
        else:
            print "Wrong method entered. Enter \"inverse\" or \"euler\"."
            return None
    else:
        print "Wrong Normalization Name Input. Enter \"L1\" or \"L2\"."
        return None

def rigid_transform_3D(A, B, frame_x, frame_y, margin):
    assert len(A) == len(B)
    N = A.shape[0]; # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.dot(np.transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T,U.T)
    recenter=centroid_B-[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    recenter=np.dot(np.linalg.inv(R),recenter)+[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    t=-recenter+centroid_A.T
    #t = -R*centroid_A.T + centroid_B.T
    return R, t

# source: ETHZ: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
def weighted_rigid_transform_3D(A, B,weights, frame_x, frame_y, margin):
    #weights should be linked to the distance. The shorter the distance is the higher the value is.
    N = np.shape(A)[0]
    weight_enlarged=np.matlib.repmat(weights,1,2)
    centroid_A = np.sum(weight_enlarged*A, axis=0)
    centroid_B = np.sum(weight_enlarged*B, axis=0)
    centroid_A = centroid_A / np.sum(weights,0)
    centroid_B = centroid_B / np.sum(weights,0)
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    W = np.diagflat(weights)
    S = np.dot(np.dot(A.T,W),B)
    U,S,Vt = np.linalg.svd(S)
    M=np.shape(U)[0]
    id_=np.identity(M)
    id_[M-1,M-1]=np.linalg.det(np.dot(Vt.T,U.T))
    R=np.dot(np.dot(Vt.T,id_),U.T)
    recenter=centroid_B-[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    recenter=np.dot(np.linalg.inv(R),recenter)+[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    t=-recenter+centroid_A.T

    return R,t

def createMovingMask(x=0, y=0,teta=0,scale=1, frame_x=300, frame_y=300,i=0):
    frame=cv2.imread("Total3.jpg",0)
    (rows,cols)=np.shape(frame)

    #Rotation of image
    M=cv2.getRotationMatrix2D((x+frame_x/2,y+frame_y/2), teta, scale)
    frame_rotated=cv2.warpAffine(frame,M,(cols,rows))

    # # WRITE BEFORE ROT
    # cv2.circle(frame,(int(x+frame_x/2),int(x+frame_y/2)),30,(0,0,0),5)
    # cv2.imwrite("debug/1_center_before_rot"+str(i)+".jpg",frame)
    # mask=frame[y:y+frame_y,x:x+frame_x]
    # cv2.circle(mask,(int(frame_x/2),int(frame_y/2)),30,(0,0,0),5)
    # cv2.imwrite("debug/1_mask_before_rot"+str(i)+".jpg",mask)
    # # WRITE AFTER ROT
    # cv2.circle(frame_rotated,(int(x+frame_x/2),int(x+frame_y/2)),30,(255,255,255),5)
    # cv2.imwrite("debug/2_center_after_rot"+str(i)+".jpg",frame_rotated)
    # mask_rotated=frame_rotated[y:y+frame_y,x:x+frame_x]
    # cv2.circle(mask_rotated,(int(frame_x/2),int(frame_y/2)),30,(255,255,255),5)
    # cv2.imwrite("debug/2_mask_after_rot"+str(i)+".jpg",mask_rotated)

    mask_rotated=frame_rotated[y:y+frame_y,x:x+frame_x]    
    
    return mask_rotated

def applyRotationTo(points, frame_x, frame_y, teta, x=0, y=0, scale=1):
    center_x=x+frame_x/2
    center_y=y+frame_y/2
    teta=deg2rad(teta)
    # Rotation Matrix
    new_points=np.zeros(np.shape(points))
    for i in range(np.shape(points)[0]):
        new_points[i][0]=int((points[i][0]-center_x)*np.cos(teta)-(points[i][1]-center_y)*np.sin(teta)+center_x)
        new_points[i][1]=int((points[i][0]-center_x)*np.sin(teta)+(points[i][1]-center_y)*np.cos(teta)+center_y)
    return new_points

def deg2rad(deg):
    rad=deg*3.14/180
    return rad

img=cv2.imread("Total3.jpg")

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

coords_0 = [2000, 2000, 0]
coords_1 = [2100, 2100, 0]
size = 600

# img_0 = img[coords_0[0]:coords_0[0]+size, coords_0[1]:coords_0[1]+size, :]
# img_1 = img[coords_1[0]:coords_1[0]+size, coords_1[1]:coords_1[1]+size, :]

img_0 = createMovingMask(coords_0[0], coords_0[1], 0, scale=1, frame_x=size, frame_y=size,i=1)
img_1 = createMovingMask(coords_1[0], coords_1[1], 10, scale=1, frame_x=size, frame_y=size,i=2)



#print "np.shape(img_0)",np.shape(img_0)
#print "np.shape(img_1)",np.shape(img_1)

# cv2.imshow('img_0',img_0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('img_1',img_1)
# cv2.waitKey(0)
#cv2.destroyAllWindows()

# -----------------------------
# Load ORB stuff
# -----------------------------

feature_params_orb = dict ( nfeatures=100,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=40,
                            firstLevel=0,
                            WTA_K=4,
                            patchSize=60
                            )


orb = cv2.ORB_create(**feature_params_orb)

# -----------------------------
# Feature detection
# -----------------------------

kp_0, des_0 = orb.detectAndCompute(img_0,mask=None)
feats_0 = formatting(kp_0)

kp_1, des_1 = orb.detectAndCompute(img_1,mask=None)
feats_1 = formatting(kp_1)

# -----------------------------
# Feature matching
# -----------------------------

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des_0, des_1, k=2)
good = []
for m,n in matches:
    if m.distance < 0.65*n.distance:
        good.append([m])


# print 'img_0.shape = ', img_0.shape

# print len(kp_0), ' features in img_0'
# print len(kp_1), ' features in img_1'
# print len(good), ' features in good'
# print 'type(good) = ', type(good[0][0])

# ------------------------------------
# Reorder features according to matches
# ------------------------------------

coords_reordered_0 = np.zeros((np.shape(good)[0],2))
coords_reordered_1 = np.zeros((np.shape(good)[0],2))
for i in range(np.shape(good)[0]):
    coords_reordered_0[i]=feats_0[good[i][0].queryIdx]
    coords_reordered_1[i]=feats_1[good[i][0].trainIdx]

# print 'type(coords_reordered_0) = ', type(coords_reordered_0)
# print 'coords_reordered_0.shape = ', coords_reordered_0.shape
# print 'coords_reordered_1.shape = ', coords_reordered_1.shape
#print "coords_reordered_0 = ",coords_reordered_0
#print "coords_reordered_1 = ",coords_reordered_1


matches_orb=cv2.drawMatchesKnn(img_0,kp_0,img_1,kp_1,good,None,flags=2)
cv2.imwrite("ORB_test1.png",matches_orb)


# ------------------------------------
# Draw circles on masks
# ------------------------------------

fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale1=2

#img contains the big frame

#Realign points with the axis of General frame
coords_reordered_0_global=applyRotationTo(coords_reordered_0, size, size, 0, x=0, y=0, scale=1)
coords_reordered_1_global=applyRotationTo(coords_reordered_1, size, size, 40, x=0, y=0, scale=1)


# for j in range(coords_reordered_0.shape[0]):
# for j in range(np.shape(coords_reordered_0)[0]):
#     #print 'j = ', j
#     center_0=coords_reordered_0_global[j]
#     center_1=coords_reordered_1_global[j]+[50,50] #displ
#     center_0 = array2tuple(center_0, ret_int=True)
#     center_1 = array2tuple(center_1, ret_int=True)

#     cv2.circle(img_0, array2tuple(coords_reordered_0[j],True), 30, (0,0,0), 5)
#     cv2.circle(img_1, array2tuple(coords_reordered_1[j],True), 35, (255,255,255), 5)
#     text_j0=tuple(map(operator.add, array2tuple(coords_reordered_0[j],True), (40,-40)))
#     cv2.putText(img_0, str(j), (int(text_j0[0]),int(text_j0[1])), fontFace, fontScale1, (0,0,0),thickness=3)
#     text_j1 = tuple(map(operator.add, array2tuple(coords_reordered_0[j],True), (40,40)))
#     cv2.putText(img_1, str(j), (int(text_j1[0]),int(text_j1[1])), fontFace, fontScale1, (255,255,255),thickness=3)

#     center_0_write=tuple(map(operator.add,center_0,(2000,2000)))
#     center_1_write=tuple(map(operator.add,center_1,(2000,2000)))
#     cv2.circle(img, center_0_write, 30, (0,255,0), 5)
#     cv2.circle(img, center_1_write, 30, (255,255,255), 5)
#     text_j0=tuple(map(operator.add, center_0, (2050,2050)))
#     cv2.putText(img, str(j), (int(text_j0[0]),int(text_j0[1])), fontFace, fontScale1, (0,255,0),thickness=3)
#     text_j1 = tuple(map(operator.add, center_1, (2050,2000-50)))
#     cv2.putText(img, str(j), (int(text_j1[0]),int(text_j1[1])), fontFace, fontScale1, (255,255,255),thickness=3)

#print 'type(coords_reordered_centers_0) = ', type(coords_reordered_centers_0)

#cv2.imshow('img_0',img_0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#cv2.imshow('img_1',img_1)
#cv2.waitKey(0)
# cv2.destroyAllWindows()


cv2.imwrite("image_test.png",img)


# ------------------------------------
# Compute displacement (x, y q)
# ------------------------------------

# print 'type(A) = ', type(A)

# for i in range(5):
#     print 'i = ', i
#     coords_reordered_0[i]=tuple2array(coords_reordered_0[i], False)
#     coords_reordered_1[i]=tuple2array(coords_reordered_1[i], False)

#l_0=np.float32([coords_reordered_0[0],coords_reordered_0[1],coords_reordered_0[2]])
#l_1=np.float32([coords_reordered_1[0],coords_reordered_1[1],coords_reordered_1[2]])

### TEST AFFINE TRANSFORM FROM OPENCV
#M=cv2.getAffineTransform(l_0, l_1)
#print M
#print M[0:2,0:2]
#R_alt=M[0:2,0:2]
#t_alt_x=M[0,2]
#t_alt_y=M[1,2]



R, t = rigid_transform_3D(coords_reordered_0, coords_reordered_1, frame_x=600, frame_y=600, margin=0)

new_t=local2global(t[0],t[1],0,0,0)
print "HEREEEEEE",new_t

angle1 = 180/3.14*(np.arctan2(R.item([1][0]),R.item([0][0])))

print "=========================================="
print "CORRECT : angle=10 / pos_x = 100 / pos_y = 100 / displ_x = 100 / displ_y = 100"
print 'angle =', angle1
print "pos_x = ",new_t[0]
print "pos_y = ",new_t[1]
print "displ_x = ",new_t[0]
print "displ_y = ",new_t[1]

















#TEST SECOND FRAME
feature_params_orb = dict ( nfeatures=50,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=20,
                            firstLevel=0,
                            WTA_K=4,
                            patchSize=60
                            )


orb = cv2.ORB_create(**feature_params_orb)

matches=[]
size=600
coords_1=[2100,2100,0]
coords_2=[2150,2150,0]
img_1 = createMovingMask(coords_1[0], coords_1[1], teta=10, scale=1, frame_x=size, frame_y=size)
#cv2.imshow("img_1",img_1)
img_2 = createMovingMask(coords_2[0], coords_2[1], teta=20, scale=1, frame_x=size, frame_y=size)
#cv2.imshow("img_2",img_2)
#cv2.waitKey(0)

kp_1, des_1 = orb.detectAndCompute(img_1,mask=None)
feats_1 = formatting(kp_1)
kp_2, des_2 = orb.detectAndCompute(img_2,mask=None)
feats_2 = formatting(kp_2)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des_1, des_2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.4*n.distance:
        good.append([m])

matches_orb=cv2.drawMatchesKnn(img_1,kp_1,img_2,kp_2,good,None,flags=2)
cv2.imwrite("ORB_test2.png",matches_orb)

coords_reordered_1 = np.zeros((np.shape(good)[0],2))
coords_reordered_2 = np.zeros((np.shape(good)[0],2))
distance=np.zeros((np.shape(good)[0],1))
for i in range(np.shape(good)[0]):
    coords_reordered_1[i]=feats_1[good[i][0].queryIdx]
    coords_reordered_2[i]=feats_2[good[i][0].trainIdx]
    distance[i]=good[i][0].distance

R2, t2 = rigid_transform_3D(coords_reordered_1, coords_reordered_2, frame_x=600, frame_y=600, margin=0)

new_t2=local2global(t2[0],t2[1],angle1,0,0)

angle2=180/3.14*(np.arctan2(R2.item([1][0]),R2.item([0][0])))

print "=========================================="
print "CORRECT : total_angle= 20 / angle=10 / pos_x = 150 / pos_y = 150 / displ_x = 50 / displ_y = 50"
print 'angle =', angle2
#print t2
print "pos_x = ", new_t[0]+new_t2[0]*np.cos(deg2rad(angle1))
print "pos_y = ", new_t[1]+new_t2[1]*np.cos(deg2rad(angle1))
print "displ_x = ",new_t2[0]*np.cos(deg2rad(angle1))
print "displ_y = ",new_t2[1]*np.cos(deg2rad(angle1))

angle2=angle1+angle2
print "angle total = ", angle2






















#TEST THIRD FRAME
feature_params_orb = dict ( nfeatures=50,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=20,
                            firstLevel=0,
                            WTA_K=4,
                            patchSize=60
                            )


orb = cv2.ORB_create(**feature_params_orb)

matches=[]
size=600
coords_2=[2150,2150,0]
coords_3=[2200,2200,0]
img_2 = createMovingMask(coords_2[0], coords_2[1], teta=20, scale=1, frame_x=size, frame_y=size)
#cv2.imshow("img_1",img_1)
img_3 = createMovingMask(coords_3[0], coords_3[1], teta=10, scale=1, frame_x=size, frame_y=size)
#cv2.imshow("img_2",img_2)
#cv2.waitKey(0)

kp_2, des_2 = orb.detectAndCompute(img_2,mask=None)
feats_2 = formatting(kp_2)
kp_3, des_3 = orb.detectAndCompute(img_3,mask=None)
feats_3 = formatting(kp_3)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des_2, des_3, k=2)

good = []
for m,n in matches:
    if m.distance < 0.4*n.distance:
        good.append([m])

matches_orb=cv2.drawMatchesKnn(img_2,kp_2,img_3,kp_3,good,None,flags=2)
cv2.imwrite("ORB_test3.png",matches_orb)

coords_reordered_2 = np.zeros((np.shape(good)[0],2))
coords_reordered_3 = np.zeros((np.shape(good)[0],2))
distance=np.zeros((np.shape(good)[0],1))
for i in range(np.shape(good)[0]):
    coords_reordered_2[i]=feats_2[good[i][0].queryIdx]
    coords_reordered_3[i]=feats_3[good[i][0].trainIdx]
    distance[i]=good[i][0].distance

weights_norm=normalize_array(distance,"L2","euler")

#R2, t2 = weighted_rigid_transform_3D(coords_reordered_1, coords_reordered_2,weights_norm, frame_x=600, frame_y=600, margin=0)

R3, t3 = rigid_transform_3D(coords_reordered_2, coords_reordered_3, frame_x=600, frame_y=600, margin=0)

new_t3=local2global(t3[0],t3[1],angle2,0,0)

angle3=180/3.14*(np.arctan2(R3.item([1][0]),R3.item([0][0])))

print "=========================================="
print "CORRECT : total_angle= 10 / angle=10 / pos_x = 200 / pos_y = 200 / displ_x = 50 / displ_y = 50"
print 'angle =', angle3
#print t2
print "pos_x = ", new_t[0]+new_t2[0]*np.cos(deg2rad(angle1))+new_t3[0]*np.cos(deg2rad(angle2))
print "pos_y = ", new_t[1]+new_t2[1]*np.cos(deg2rad(angle1))+new_t3[1]*np.cos(deg2rad(angle2))
print "displ_x = ",new_t3[0]*np.cos(deg2rad(angle2))
print "displ_y = ",new_t3[1]*np.cos(deg2rad(angle2))

angle3=angle2+angle3
print "angle total = ", angle3















#TEST FOURTH FRAME
feature_params_orb = dict ( nfeatures=50,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=20,
                            firstLevel=0,
                            WTA_K=4,
                            patchSize=60
                            )

orb = cv2.ORB_create(**feature_params_orb)

matches=[]
size=600
coords_3=[2200,2200,0]
coords_4=[2300,2300,0]
img_3 = createMovingMask(coords_3[0], coords_3[1], teta=10, scale=1, frame_x=size, frame_y=size)
#cv2.imshow("img_1",img_1)
img_4 = createMovingMask(coords_4[0], coords_4[1], teta=30, scale=1, frame_x=size, frame_y=size)
#cv2.imshow("img_2",img_2)
#cv2.waitKey(0)

kp_3, des_3 = orb.detectAndCompute(img_3,mask=None)
feats_2 = formatting(kp_3)
kp_4, des_4 = orb.detectAndCompute(img_4,mask=None)
feats_4 = formatting(kp_4)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des_3, des_4, k=2)

good = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append([m])

matches_orb=cv2.drawMatchesKnn(img_3,kp_3,img_4,kp_4,good,None,flags=2)
cv2.imwrite("ORB_test4.png",matches_orb)

coords_reordered_3 = np.zeros((np.shape(good)[0],2))
coords_reordered_4 = np.zeros((np.shape(good)[0],2))
distance=np.zeros((np.shape(good)[0],1))
for i in range(np.shape(good)[0]):
    coords_reordered_3[i]=feats_3[good[i][0].queryIdx]
    coords_reordered_4[i]=feats_4[good[i][0].trainIdx]
    distance[i]=good[i][0].distance

weights_norm=normalize_array(distance,"L2","euler")

#R2, t2 = weighted_rigid_transform_3D(coords_reordered_1, coords_reordered_2,weights_norm, frame_x=600, frame_y=600, margin=0)

R4, t4 = rigid_transform_3D(coords_reordered_3, coords_reordered_4, frame_x=600, frame_y=600, margin=0)

new_t4=local2global(t4[0],t4[1],angle3,0,0)

angle4=180/3.14*(np.arctan2(R4.item([1][0]),R4.item([0][0])))

print "=========================================="
print "CORRECT : total_angle= 10 / angle=10 / pos_x = 200 / pos_y = 200 / displ_x = 50 / displ_y = 50"
print 'angle =', angle4
#print t2
print "pos_x = ", new_t[0]+new_t2[0]*np.cos(deg2rad(angle1))+new_t3[0]*np.cos(deg2rad(angle2))+new_t4[0]*np.cos(deg2rad(angle3))
print "pos_y = ", new_t[1]+new_t2[1]*np.cos(deg2rad(angle1))+new_t3[1]*np.cos(deg2rad(angle2))+new_t4[1]*np.cos(deg2rad(angle3))
print "displ_x = ",new_t4[0]*np.cos(deg2rad(angle3))
print "displ_y = ",new_t4[1]*np.cos(deg2rad(angle3))

angle4=angle3+angle4
print "angle total = ", angle4

