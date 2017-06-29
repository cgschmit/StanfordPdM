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

#margin define the margin we take of each side.
def rigid_transform_3D(A, B, frame_x, frame_y, margin):
    assert len(A) == len(B)
    N = A.shape[0]; # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points to their center of mass
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T,U.T)
    #Recenter rotation to centre of frame and not to the left top corner.
    recenter=centroid_A-[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    recenter=np.dot(R,recenter)+[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
    #if np.linalg.det(R) < 0:
    #   print "Reflection detected"
    #   Vt[2,:] *= -1
    #   R = Vt.T * U.T
    t= centroid_B-recenter
    return R, t

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

def formatting(tupleTab):
    totalTab=[]
    for i in range(np.size(tupleTab)):
        tab = [[np.float32(tupleTab[i].pt[0]), np.float32(tupleTab[i].pt[1])]]
        totalTab.append(tab)
    myarray = np.asarray(totalTab)
    return myarray

def applyRotationTo(points, frame_x, frame_y, teta, x=0, y=0, scale=1):
    center_x=x+frame_x/2
    center_y=y+frame_y/2
    teta=deg2rad(teta)
    new_points=np.zeros(np.shape(points))
    for i in range(np.shape(points)[0]):
        if np.shape(points)==(2,):
            new_points[0]=int((points[0]-center_x)*np.cos(teta)-(points[1]-center_y)*np.sin(teta)+center_x)
            new_points[1]=int((points[0]-center_x)*np.sin(teta)+(points[1]-center_y)*np.cos(teta)+center_y)
        else:
            new_points[i][0]=int((points[i][0]-center_x)*np.cos(teta)-(points[i][1]-center_y)*np.sin(teta)+center_x)
            new_points[i][1]=int((points[i][0]-center_x)*np.sin(teta)+(points[i][1]-center_y)*np.cos(teta)+center_y)
            
    return new_points

def rad2deg(rad):
    deg=rad*180/3.14
    return deg

def deg2rad(deg):
    rad=deg*3.14/180
    return rad

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
feature_params_orb = dict ( nfeatures=50,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=20, 
                            firstLevel=0, 
                            WTA_K=4, 
                            patchSize=30
                            )

feature_param_sift = dict( )

feature_param_surf = dict( )

feature_param_brief = dict( )
#-----------------------------

frame_gray=cv2.imread("img-2D-rot/image_0.png",0)
(m_frame_x,m_frame_y)=np.shape(frame_gray)
margin=0.1
edge1=int(m_frame_x*margin)
edge2=int(m_frame_x*(1-margin))
edge3=int(m_frame_y*margin)
edge4=int(m_frame_y*(1-margin))
ROI_current=frame_gray[edge1:edge2, edge3:edge4]

orb = cv2.ORB_create(**feature_params_orb)


while(1):
    iterator+=1
    print ""
    print "------------------------------------------"
    print ""
    print "IMAGE NUMBER:",iterator
    frame_gray=cv2.imread("img-2D-rot/image_"+str(iterator)+".png",0)
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
            if m.distance < 0.5*n.distance:
                good.append([m])

        #------------------
        #DRAW MATCHES
        matches_orb=cv2.drawMatchesKnn(ROI_current,kp_current_orb,ROI_new,kp_new_orb,good,None,flags=2)
        cv2.imwrite("DrawMatches/ORB"+str(iterator)+".png",matches_orb)
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

        if np.shape(inpu_t)[0]==0:
            #Prediction avec precedants points!
            angle=0
            translation_x=0
            translation_y=0
        elif np.shape(inpu_t)[0]==1:
            #Prediction avec precedants points!
            angle=0
            translation_x=0
            translation_y=0
        elif np.shape(inpu_t)[0]==2:
            #Prediction avec precedants points!
            angle=0
            translation_x=0
            translation_y=0
        else:
            R, t=rigid_transform_3D(inpu_t, outpu_t, m_frame_x, m_frame_y, margin)
            angle=rad2deg(np.arctan2(-R.item([1][0]),R.item([0][0])))
            translation_x=t[0]*np.cos(total_angle+angle)
            translation_y=t[1]*np.cos(total_angle+angle)



        
        total_angle+=angle
        #projected on the orthnormale axis of the first frame
        # TO RESEE SHOULD I DO COS(ANGLE) TO X AND Y OR COS AND SIN TO X AND Y RESPECTIVELY
        
        total_translation_x+=translation_x
        total_translation_y+=translation_y

        print "TRANSLATION TOTALE X:",total_translation_x
        print "TRANSLATION TOTALE Y:",total_translation_y
        print "ANGLE:", angle
        print "TOTAL ANGLE:", total_angle

        ROI_current=ROI_new
    
    else:
        break



# Variables for Visualization
img_2D_rot_posFrame=[(200,200),(300,300),(400,400),(500,500),(600,600),(700,700),(800,800),(900,900),(1000,1000),(1100,1100),(1200,1200),(1300,1300),(1400,1400),(1500,1500),(1600,1600),(1700,1700),(1800,1800),(1900,1900),(2000,2000),(2100,2100),(2200,2200),(2300,2300),(2400,2400)]
img_2D_rot_displ=[(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100)]
img_2D_rot_angle=[0,10,0,-10,5,15,20,0,20,10,-10,-5,-2,0,10,20,40,30,10,0,-5,-10,0]
img_2D_rot_cummDispl=[(100,100),(200,200),(300,300),(400,400),(500,500),(600,600),(700,700),(800,800),(900,900),(1000,1000),(1100,1100),(1200,1200),(1300,1300),(1400,1400),(1500,1500),(1600,1600),(1700,1700),(1800,1800),(1900,1900),(2000,2000),(2100,2100),(2200,2200),(2300,2300)]

frame_x_ROI=(1-2*margin)*m_frame_x
frame_y_ROI=(1-2*margin)*m_frame_y

mask_x=(frame_x_ROI,0)
mask_y=(0,frame_y_ROI)
mask_xy=(frame_x_ROI,frame_y_ROI)

fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale1=2

pt1=[0,0]
pt2=[frame_x_ROI,0]
pt3=[frame_x_ROI,frame_y_ROI]
pt4=[0,frame_y_ROI]



for i in range(np.shape(all_feats_new)[0]):
    img=cv2.imread("Total3.jpg")

    # Print of Frame time T-1
    pt1_current=tuple(map(operator.add,img_2D_rot_posFrame[i],(margin*m_frame_x,margin*m_frame_y)))
    pt2_current=tuple(map(operator.add,mask_x,tuple(map(operator.add,img_2D_rot_posFrame[i],(margin*m_frame_x,margin*m_frame_y)))))
    pt3_current=tuple(map(operator.add,mask_xy,tuple(map(operator.add,img_2D_rot_posFrame[i],(margin*m_frame_x,margin*m_frame_y)))))
    pt4_current=tuple(map(operator.add,mask_y,tuple(map(operator.add,img_2D_rot_posFrame[i],(margin*m_frame_x,margin*m_frame_y)))))
    pt1_current=tuple2array(pt1_current,False)
    pt2_current=tuple2array(pt2_current,False)
    pt3_current=tuple2array(pt3_current,False)
    pt4_current=tuple2array(pt4_current,False)
    current_frame=np.array([pt1_current,pt2_current,pt3_current,pt4_current])
    current_point_frame_rotated=applyRotationTo(current_frame, frame_x_ROI, frame_y_ROI, img_2D_rot_angle[i], img_2D_rot_posFrame[i][0]+margin*m_frame_x, img_2D_rot_posFrame[i][1]+margin*m_frame_y)

    # Print of Frame time T
    pt1_new=tuple(map(operator.add,img_2D_rot_posFrame[i+1],(margin*m_frame_x,margin*m_frame_y)))
    pt2_new=tuple(map(operator.add,mask_x,tuple(map(operator.add,img_2D_rot_posFrame[i+1],(margin*m_frame_x,margin*m_frame_y)))))
    pt3_new=tuple(map(operator.add,mask_xy,tuple(map(operator.add,img_2D_rot_posFrame[i+1],(margin*m_frame_x,margin*m_frame_y)))))
    pt4_new=tuple(map(operator.add,mask_y,tuple(map(operator.add,img_2D_rot_posFrame[i+1],(margin*m_frame_x,margin*m_frame_y)))))
    pt1_new=tuple2array(pt1_new,False)
    pt2_new=tuple2array(pt2_new,False)
    pt3_new=tuple2array(pt3_new,False)
    pt4_new=tuple2array(pt4_new,False)
    new_frame=np.array([pt1_new,pt2_new,pt3_new,pt4_new])
    new_point_frame_rotated=applyRotationTo(new_frame, frame_x_ROI, frame_y_ROI, img_2D_rot_angle[i+1], img_2D_rot_posFrame[i+1][0]+margin*m_frame_x, img_2D_rot_posFrame[i+1][1]+margin*m_frame_y)

    # Print of both Frame on Image and Legend
    cv2.polylines(img,np.int32([current_point_frame_rotated]),True,(0,255,0),thickness=5)
    cv2.polylines(img,np.int32([new_point_frame_rotated]),True,(255,0,0),thickness=5)
    cv2.putText(img, 'frame'+str(i), array2tuple(current_point_frame_rotated[0],True), fontFace, fontScale1, (0,255,0),thickness=5)
    cv2.putText(img, 'frame'+str(i+1), array2tuple(new_point_frame_rotated[2],True), fontFace, fontScale1, (255,0,0),thickness=5)

    # Printing each frame with legend on big map
    for j in range(np.shape(all_feats_current[i])[0]):
        center_current=array2tuple(all_feats_current[i][j],ret_int=True)
        center_new=array2tuple(all_feats_new[i][j],ret_int=True)
        center_current=tuple(map(operator.add,center_current,tuple(map(operator.add,img_2D_rot_posFrame[i],(margin*m_frame_x,margin*m_frame_y)))))
        center_new=tuple(map(operator.add,center_new,tuple(map(operator.add,img_2D_rot_posFrame[i+1],(margin*m_frame_x,margin*m_frame_y)))))
        
        center_current=applyRotationTo(tuple2array(center_current,True), frame_x_ROI, frame_y_ROI, img_2D_rot_angle[i],img_2D_rot_posFrame[i][0]+margin*m_frame_x, img_2D_rot_posFrame[i][1]+margin*m_frame_y)
        center_new=applyRotationTo(tuple2array(center_new,True), frame_x_ROI, frame_y_ROI, img_2D_rot_angle[i+1],img_2D_rot_posFrame[i][0]+margin*m_frame_x, img_2D_rot_posFrame[i][1]+margin*m_frame_y)

        cv2.circle(img,array2tuple(center_current,True),30,(0,255,0),5)
        cv2.circle(img,array2tuple(center_new,True),30,(255,0,0),5)
        
        text_j0=tuple(map(operator.add,center_current,(40,40)))
        text_j1=tuple(map(operator.add,center_new,(40,40)))
        text_j0=array2tuple(text_j0,True)
        text_j1=array2tuple(text_j1,True)

        cv2.putText(img, str(j), (int(text_j0[0]),int(text_j0[1])), fontFace, fontScale1, (0,255,0),thickness=3)
        cv2.putText(img, str(j), (int(text_j1[0]),int(text_j1[1])), fontFace, fontScale1, (255,0,0),thickness=3)

    # Write all to the big map
    cv2.imwrite("test-rigidTransform-orientationtranslation/image_"+str(i)+".png",img)

