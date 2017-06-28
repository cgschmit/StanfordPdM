from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator

### COMMENTAIRES:
# rigid_transform_3D appears not to be working regarding the results we receive even though checking the matching is good.
# TO DO: find a new function to get Rotation and Translation from two images.

# EDITED: Looking at DrawMatches, the matches are good but looking at the plot we do with the position of the features, the matching is not good.
# Problem is remaining in the matching features, but also rigid_transform_3D apparently.

# FINAL EDIT: FEATURE MATCHING DIDN'T WORK FOR THE SIMPLE REASON THAT CURRENT_FEAT HAS TO BE DONE EVERY LOOP. MISTAKE WAS TO TRY TO OPTIMIZE BY 
# doing only one orb and compute per loop by assigning the new value to current.
###

# def rigid_transform_3D(A, B):
#     assert len(A) == len(B)
#     N = A.shape[0];
#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
#     AA = A - np.tile(centroid_A, (N, 1))
#     BB = B - np.tile(centroid_B, (N, 1))
#     H = np.dot(np.transpose(AA), BB)
#     U, S, Vt = np.linalg.svd(H)
#     R = np.dot(U.T ,Vt.T)
#     t = -np.dot(R,centroid_A.T) + centroid_B.T
#     return R, t

def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0]; # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    print "centroid_A",centroid_A
    print "centroid_B",centroid_B

    # centre the points to their center of mass
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T,U.T)
    recenter=centroid_A-[270,270]
    recenter=np.dot(R,recenter)+[270,270]
    # special reflection case
    if np.linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T
    t= centroid_B-recenter
    return R, t, centroid_A, centroid_B, AA, BB

def array2tuple(array1D,ret_int):
    if ret_int==True:
        return (int(array1D[0]),int(array1D[1]))
    else:
        return (array1D[0],array1D[1])

def formatting(tupleTab):
    totalTab=[]
    for i in range(np.size(tupleTab)):
        tab = [[np.float32(tupleTab[i].pt[0]), np.float32(tupleTab[i].pt[1])]]
        totalTab.append(tab)
    myarray = np.asarray(totalTab)
    return myarray

def applyRotationTo(points, frame_x, frame_y, x, y, teta, scale=1):
    center_x=x+frame_x/2
    center_y=y+frame_y/2
    teta=deg2rad(teta)
    new_points=np.zeros(np.shape(points))
    for i in range(np.shape(points)[0]):
        new_points[i][0]=int((points[i][0]-center_x)*np.cos(teta)-(points[i][1]-center_y)*np.sin(teta)+center_x)
        new_points[i][1]=int((points[i][0]-center_x)*np.sin(teta)+(points[i][1]-center_y)*np.cos(teta)+center_y)
    return new_points

def rad2deg(rad):
    return rad*180/3.14

def deg2rad(deg):
    return deg*3.14/180

# VARIABLES
#-----------------------------
iterator=0
all_feats_current=[]
all_feats_new=[]
total_angle=0
total_translation_x=0
total_translation_y=0
R_tab=[]
t_tab=[]
centroid_A_tab=[]
centroid_B_tab=[]
AA_tab=[]
BB_tab=[]
#-----------------------------

# PARAMS FUNCTIONS OPENCV
#-----------------------------
feature_params_orb = dict ( nfeatures=20,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=31, 
                            firstLevel=0, 
                            WTA_K=4, 
                            patchSize=31
                            )

feature_param_sift = dict( )

feature_param_surf = dict( )

feature_param_brief = dict( )
#-----------------------------

frame_gray=cv2.imread("img-rot/image_0.png",0)
(m_frame_x,m_frame_y)=np.shape(frame_gray)
edge1=int(m_frame_x*0.1)
edge2=int(m_frame_x*0.9)
edge3=int(m_frame_y*0.1)
edge4=int(m_frame_y*0.9)
ROI_current=frame_gray[edge1:edge2, edge3:edge4]

orb = cv2.ORB_create(**feature_params_orb)


while(iterator<23):
    iterator+=1
    print ""
    print "------------------------------------------"
    print ""
    print "IMAGE NUMBER:",iterator
    frame_gray=cv2.imread("img-rot/image_"+str(iterator)+".png",0)
    if frame_gray is not None:   
        ROI_new=frame_gray[edge1:edge2, edge3:edge4]

        kp_current_orb, des_current_orb = orb.detectAndCompute(ROI_current,mask=None)
        feats_current_orb = formatting(kp_current_orb)

        kp_new_orb, des_new_orb = orb.detectAndCompute(ROI_new,mask=None)
        feats_new_orb = formatting(kp_new_orb)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des_current_orb,des_new_orb, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.45*n.distance:
                good.append([m])

        #------------------
        #DRAW MATCHES
        matches_orb=cv2.drawMatchesKnn(ROI_current,kp_current_orb,ROI_new,kp_new_orb,good,None,flags=2)
        cv2.imwrite("test-rigidTransform/DrawMatches/ORB"+str(iterator)+".png",matches_orb)
        #------------------

        #------------------
        #ORDER MATCHED FEATURES
        inpu_t = np.zeros((np.shape(good)[0],2))
        outpu_t = np.zeros((np.shape(good)[0],2))
        for i in range(np.shape(good)[0]):
            inpu_t[i]=feats_current_orb[good[i][0].queryIdx]
            outpu_t[i]=feats_new_orb[good[i][0].trainIdx]
        #------------------

        all_feats_current.append(inpu_t)
        all_feats_new.append(outpu_t)

        R, t, centroid_A, centroid_B, AA, BB=rigid_transform_3D(inpu_t, outpu_t)

        R_tab.append(R)
        t_tab.append(t)
        AA_tab.append(AA)
        BB_tab.append(BB)
        centroid_A_tab.append(centroid_A)
        centroid_B_tab.append(centroid_B)


        angle=rad2deg(np.arctan2(-R.item([1][0]),R.item([0][0])))
        total_angle+=angle
        #projected on the orthnormale axis of the first frame
        # TO RESEE SHOULD I DO COS(ANGLE) TO X AND Y OR COS AND SIN TO X AND Y RESPECTIVELY
        translation_x=t[0]*np.cos(total_angle)
        translation_y=t[1]*np.cos(total_angle)
        total_translation_x+=translation_x
        total_translation_y+=translation_y

        print "TRANSLATION TOTALE X:",total_translation_x
        print "TRANSLATION TOTALE Y:",total_translation_y
        print "ANGLE:", angle
        print "TOTAL ANGLE:", total_angle

        ROI_current=ROI_new
    
    else:
        break


# Parameters applyRotationTo
#--------------
angle=[0,10,12,27,36]
#BEWARE TAKE INTO CONSIDERATION TO RECENTER THE CENTER TO THE SIZE OF THE ROI TO PERFORM "applyRotationTo"!
margin=0.2 #0.1 a droite et 0.1 a gauche
frame_x=600-margin*600
frame_y=600-margin*600
x=0
y=0
#--------------

fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale1=0.7

# i itere sur les frames
# j itere sur les features du i-th frame
for i in range(np.shape(all_feats_new)[0]):
    #print "ITERATION NUMBER I:",i
    img_current=cv2.imread("img-rot/image_"+str(i)+".png",0)
    img_new=cv2.imread("img-rot/image_"+str(i+1)+".png",0)
    (m_frame_x,m_frame_y)=np.shape(img_current)
    edge1=int(m_frame_x*0.1)
    edge2=int(m_frame_x*0.9)
    edge3=int(m_frame_y*0.1)
    edge4=int(m_frame_y*0.9)
    ROI_current=img_current[edge1:edge2, edge3:edge4]
    ROI_new=img_new[edge1:edge2, edge3:edge4]

    # NEW ARE ALREADY IN THE RIGHT REFERENTIAL OF THE FRAME
    # CURRENT IS IN THE REFERENTIAL OF THE FIRST FRAME SO WE NEED TO APPLY THE AFFINE TRANSFORM TO PUT IT AT THE RIGHT PLACE
    # center_current_alt=applyRotationTo(all_feats_current[i], frame_x, frame_y, x, y, -angle[i], scale=1)

    for j in range(np.shape(all_feats_current[i])[0]):
        #print "FEATURES NUMBER:", j
        center_current=array2tuple(all_feats_current[i][j],ret_int=True)
        center_new=array2tuple(all_feats_new[i][j],ret_int=True)
        cv2.circle(ROI_current,center_current,5,(0,255,0),2)
        cv2.circle(ROI_new,center_new,5,(0,255,0),2)

        cv2.circle(ROI_current,array2tuple(centroid_A_tab[i],True),10,(255,255,255),4)
        cv2.circle(ROI_new,array2tuple(centroid_B_tab[i],True),10,(255,255,255),4)
        cv2.circle(ROI_current,array2tuple(np.dot(R_tab[i],centroid_A_tab[i].T),True),10,(0,0,0),4)

        text_j0=tuple(map(operator.add,center_current,(5,5)))
        text_j1=tuple(map(operator.add,center_new,(5,-5)))
        text_j0=array2tuple(text_j0,True)
        text_j1=array2tuple(text_j1,True)
        cv2.putText(ROI_current, str(j), (int(text_j0[0]),int(text_j0[1])), fontFace, fontScale1, (0,255,0),thickness=3)
        cv2.putText(ROI_new, str(j), (int(text_j1[0]),int(text_j1[1])), fontFace, fontScale1, (0,255,0),thickness=3)


    cv2.imwrite("test-rigidTransform/current_"+str(i+1)+".png",ROI_current)
    cv2.imwrite("test-rigidTransform/new_"+str(i+1)+".png",ROI_new)







