from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator


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

# def rigid_transform_3D(A, B, frame_x, frame_y, margin=0):
#     # DONT CALL WITH 1 FEATURE MATCHED, SUM WILL BE DONE OVER X AND Y instead of all the X and all the Y.
#     assert len(A) == len(B)
#     N = A.shape[0]; # total points
#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
#     # centre the points to their center of mass 
#     AA = A - np.tile(centroid_A, (N, 1))
#     BB = B - np.tile(centroid_B, (N, 1))
#     print "MEAN RECENTERED INPUT",np.mean(AA,axis=0)
#     print "MEAN RECENTERED OUTPUT",np.mean(BB,axis=0)
#     H = np.dot(np.transpose(AA),BB)
#     U, S, Vt = np.linalg.svd(H)
#     R = np.dot(Vt.T,U.T)
#     #Recenter rotation to centre of frame and not to the left top corner.
#     #Positioning to the origin of the axes
#     recenter = centroid_A - [(1-2*margin) / 2 * frame_x, (1-2*margin) / 2 * frame_y]
#     #Applying Rotation of the recentered points
#     recenter_1 =np.dot(R,recenter)
#     #Adding the translation back
#     recenter_2 = recenter_1+[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
#     if np.linalg.det(R) < 0:
#        print "Reflection detected"
#     #   Vt[2,:] *= -1
#     #   R = Vt.T * U.T
#     t = centroid_B - recenter_2
#     return R, t

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
    t=recenter-centroid_A.T
    #t = -R*centroid_A.T + centroid_B.T
    return R, t

def createMovingMask(x=0, y=0,teta=0,scale=1, frame_x=300, frame_y=300):
    frame=cv2.imread("Total3.jpg",0)
    (rows,cols)=np.shape(frame)
    M=cv2.getRotationMatrix2D((x+frame_x/2,y+frame_y/2), teta, scale)
    frame=cv2.warpAffine(frame,M,(rows,cols))
    mask=frame[y:y+frame_y,x:x+frame_x]
    return mask

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
coords_1 = [2020, 2050, 0]
size = 600

# img_0 = img[coords_0[0]:coords_0[0]+size, coords_0[1]:coords_0[1]+size, :]
# img_1 = img[coords_1[0]:coords_1[0]+size, coords_1[1]:coords_1[1]+size, :]

img_0 = createMovingMask(coords_0[0], coords_0[1], teta=0, scale=1, frame_x=size, frame_y=size)
img_1 = createMovingMask(coords_1[0], coords_1[1], teta=30, scale=1, frame_x=size, frame_y=size)

print "np.shape(img_0)",np.shape(img_0)
print "np.shape(img_1)",np.shape(img_1)

#cv2.imshow('img_0',img_0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#cv2.imshow('img_1',img_1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# -----------------------------
# Load ORB stuff
# -----------------------------

feature_params_orb = dict ( nfeatures=50,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=20,
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
    if m.distance < 0.4*n.distance:
        good.append([m])


print 'img_0.shape = ', img_0.shape

print len(kp_0), ' features in img_0'
print len(kp_1), ' features in img_1'
print len(good), ' features in good'
print 'type(good) = ', type(good[0][0])

# ------------------------------------
# Reorder features according to matches
# ------------------------------------

coords_reordered_0 = np.zeros((np.shape(good)[0],2))
coords_reordered_1 = np.zeros((np.shape(good)[0],2))
for i in range(np.shape(good)[0]):
    coords_reordered_0[i]=feats_0[good[i][0].queryIdx]
    coords_reordered_1[i]=feats_1[good[i][0].trainIdx]

print 'type(coords_reordered_0) = ', type(coords_reordered_0)
print 'coords_reordered_0.shape = ', coords_reordered_0.shape
print 'coords_reordered_1.shape = ', coords_reordered_1.shape
#print "coords_reordered_0 = ",coords_reordered_0
#print "coords_reordered_1 = ",coords_reordered_1


matches_orb=cv2.drawMatchesKnn(img_0,kp_0,img_1,kp_1,good,None,flags=2)
cv2.imwrite("ORB_test.png",matches_orb)


# ------------------------------------
# Draw circles on masks
# ------------------------------------

fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale1=2

#img contains the big frame

#Realign points with the axis of General frame
coords_reordered_0_global=applyRotationTo(coords_reordered_0, size, size, 0, x=0, y=0, scale=1)
coords_reordered_1_global=applyRotationTo(coords_reordered_1, size, size, 20, x=0, y=0, scale=1)


# for j in range(coords_reordered_0.shape[0]):
for j in range(np.shape(coords_reordered_0)[0]):
    #print 'j = ', j
    center_0=coords_reordered_0_global[j]
    center_1=coords_reordered_1_global[j]+[50,50] #displ
    center_0 = array2tuple(center_0, ret_int=True)
    center_1 = array2tuple(center_1, ret_int=True)

    cv2.circle(img_0, array2tuple(coords_reordered_0[j],True), 30, (0,0,0), 5)
    cv2.circle(img_1, array2tuple(coords_reordered_1[j],True), 35, (255,255,255), 5)
    text_j0=tuple(map(operator.add, array2tuple(coords_reordered_0[j],True), (40,-40)))
    cv2.putText(img_0, str(j), (int(text_j0[0]),int(text_j0[1])), fontFace, fontScale1, (0,0,0),thickness=3)
    text_j1 = tuple(map(operator.add, array2tuple(coords_reordered_0[j],True), (40,40)))
    cv2.putText(img_1, str(j), (int(text_j1[0]),int(text_j1[1])), fontFace, fontScale1, (255,255,255),thickness=3)

    center_0_write=tuple(map(operator.add,center_0,(2000,2000)))
    center_1_write=tuple(map(operator.add,center_1,(2000,2000)))
    cv2.circle(img, center_0_write, 30, (0,255,0), 5)
    cv2.circle(img, center_1_write, 30, (255,255,255), 5)
    text_j0=tuple(map(operator.add, center_0, (2040,2040)))
    cv2.putText(img, str(j), (int(text_j0[0]),int(text_j0[1])), fontFace, fontScale1, (0,255,0),thickness=3)
    text_j1 = tuple(map(operator.add, center_1, (2040,2000-40)))
    cv2.putText(img, str(j), (int(text_j1[0]),int(text_j1[1])), fontFace, fontScale1, (255,255,255),thickness=3)

#print 'type(coords_reordered_centers_0) = ', type(coords_reordered_centers_0)

cv2.imshow('img_0',img_0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cv2.imshow('img_1',img_1)
cv2.waitKey(0)
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

print 'R = ', R
print 'angle =', 180/3.14*(np.arctan2(-R.item([1][0]),R.item([0][0])))
print 't = ', t
