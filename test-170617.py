from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator

# VARIABLES INIT
angle_all=[]
tot_angle=0
t_tot_x=0
t_tot_y=0
dist_tot_alter_x=[]
dist_tot_alter_y=[]
dist_x_=[]
dist_y_=[]
iterator=0
all_feats_current=[]
all_feats_new=[]
mask_xy=(600,600)
mask_x=(600,0)
mask_y=(0,600)
# FOR TESTING
color=[(0,0,255),(0,0,150),(0,255,0),(0,150,0),(255,0,0),(150,0,0),(0,255,255),(255,0,255),(255,255,0),(255,255,255),(0,0,255),(0,0,150),(0,255,0),(0,150,0),(255,0,0),(150,0,0),(0,255,255),(255,0,255),(255,255,0),(255,255,255),(0,0,255),(0,0,150),(0,255,0),(0,150,0),(255,0,0),(150,0,0),(0,255,255),(255,0,255),(255,255,0),(255,255,255)]
mask=(600,600)
frame_pos=[(2200,1700),(2250,1700),(2300,1700),(2350,1700),(2370,1700),(2390,1700),(2420,1700),(2450,1700),(2500,1700),(2700,1700)]
displ=[(50,0),(50,0),(50,0),(20,0),(20,0),(30,0),(30,0),(50,0),(200,0)]
frame_pos_2=[(2200,1700),(2250,1900),(2300,1800),(2350,1900),(2370,2000),(2390,1850),(2420,2000),(2450,2200),(2500,2400),(2700,2500)]
displ_2=[(50,200),(50,-100),(50,100),(20,100),(20,-150),(30,150),(30,200),(50,200),(200,100)]
cummulative_displ2= [(50,200),(100,100),(150,200),(170,300),(190,150),(220,300),(250,500),(300,700),(500,800)]
#frame_pos_3=[(200,200),(300,300),(400,400),(500,500),(600,600),(700,700),(800,800),(900,900),(1000,1000),(1100,1100),(1200,1200),(1300,1300),(1400,1400),(1500,1500),(1600,1600),(1700,1700),(1800,1800),(1900,1900),(2000,2000),(2100,2100),(2200,2200),(2300,2300),(2400,2400)]
#displ_3=[(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100)]
#cummulative_displ3= [(100,100),(200,200),(300,300),(400,400),(500,500),(600,600),(700,700),(800,800),(900,900),(1000,1000),(1100,1100),(1200,1200),(1300,1300),(1400,1400),(1500,1500),(1600,1600),(1700,1700),(1800,1800),(1900,1900),(2000,2000),(2100,2100),(2200,2200),(2300,2300)]
#angle_3=[0,10,0,-10,5,15,20,0,20,10,-10,-5,-2,0,10,20,40,30,10,0,-5,-10,0]
#angle_3=[0, 0,0,0  ,0,0 ,0 ,0, 0, 0,  0, 0, 0,0, 0, 0, 0, 0, 0,0, 0,  0,0]

frame_pos_3=[(2000,2000),(2000,2000),(2000,2000),(2000,2000),(2000,2000)]
displ_3=[(0,0),(0,0),(0,0),(0,0),(0,0)]
cummulative_displ3=[(0,0),(0,0),(0,0),(0,0),(0,0)]
angle_img_rot=[0,10,12,27,36]

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

feature_params_gftt = dict( maxCorners=20, 
                            qualityLevel=0.01, 
                            minDistance=10)#, 
                            #corners, 
                            #mask, 
                            #blockSize, 
                            #useHarrisDetector
                           #)

# FUNCTIONS
#-----------------------------
def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    if np.shape(A)[0]>0:
        N = A.shape[0];
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
        H = np.dot(np.transpose(AA), BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T ,U.T)
        t = -np.dot(R,centroid_A.T) + centroid_B.T
    elif np.shape(A)[0]==0:
        R=np.identity(2)
        t=np.array([0,0])
    return R, t

def array2tuple(arr_ay):
    tup=(arr_ay[0],arr_ay[1])
    return tup

def array2tuple2(arr_ay):
    tup=(arr_ay[0][0],arr_ay[0][1])
    return tup

def array2tuple_int(arr_ay):
    return (int(arr_ay[0]),int(arr_ay[1]))

def array2tuple_int2(arr_ay):
    return (int(arr_ay[0][0]),int(arr_ay[0][1]))

def tuple2array(tup_le):
    arr_ay=[tup_le[0],tup_le[1]]
    return arr_ay

def tuple2array2(tup_le):
    arr_ay=np.array([[tup_le[0],tup_le[1]]])
    return arr_ay

def rad2deg(rad):
    ang=rad*180/3.14
    return ang

def array2array2D(arr_ay):
    return np.array([[arr_ay[0],arr_ay[1]]])

def deg2rad(deg):
    rad=deg*3.14/180
    return rad

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

# INIT
#-----------------------------
#cap = cv2.VideoCapture('OptiTrack-Video/version3-current/NewGood-alignedToAxis-1s-43ms.mov')
#ret, frame = cap.read()
#frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
###TEST
frame_gray=cv2.imread("img-rot/image_0.png",0)
(m_frame_x,m_frame_y)=np.shape(frame_gray)
edge1=int(m_frame_x*0.1)
edge2=int(m_frame_x*0.9)
edge3=int(m_frame_y*0.1)
edge4=int(m_frame_y*0.9)
ROI_current=frame_gray[edge1:edge2, edge3:edge4]

# IDEA IMPROVEMENT --> SELECT ROI with mask in the middle of the frame to not get features that 
# have a big chance to disappear on the frame in order to avoid wrong feature matching.
# 1)
orb = cv2.ORB_create(**feature_params_orb)
kp_current, des_current = orb.detectAndCompute(ROI_current,mask=None)
feats_current = formatting(kp_current)
# 2)
# gftt=cv2.goodFeaturesToTrack(ROI_current, **feature_params_gftt)
# print "FEATURES",gftt



# WHILE CAP IS OPENED
#-----------------------------
#while(cap.isOpened()):
###TEST
while(iterator<5):
    #ret, frame = cap.read()
    ###TEST
    iterator+=1
    print "IMAGE NUMBER:",iterator
    frame_gray=cv2.imread("img-rot/image_"+str(iterator)+".png",0)
    if frame_gray is not None:   
        ROI_new=frame_gray[edge1:edge2, edge3:edge4]
        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_new, des_new = orb.detectAndCompute(ROI_new,mask=None)
        feats_new = formatting(kp_new)

        # --> FEATURE MATCHING: 1) OR 2)
        ####################################################
        ### 1) MATCHING BRUTE FORCE
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des_current,des_new)
        ### 2) MATCHING K-NN
        # bf = cv2.BFMatcher()
        # matches1 = bf.knnMatch(des_old,des, k=2)
        # matches = []
        # for m,n in matches1:
        #   if m.distance < 0.8*n.distance:
        #       matches.append(m)

        ### 3) IMPROVING PERFORMANCE FEATURE MATCHING by choosing only the 
        sum_distance=0
        size_tab=0
        index_matches=[]
        for i in range(np.size(matches)):
            sum_distance+=matches[i].distance
        if np.size(matches)>0:
            avg_distance=sum_distance/np.size(matches)
        else:
            avg_distance=0
        for i in range(np.size(matches)):
            #if matches[i].distance>inf and matches[i]<sup:
            if matches[i].distance<=avg_distance:
                size_tab+=1
                index_matches.append(i)
        inpu_t = np.zeros((size_tab,2))
        outpu_t = np.zeros((size_tab,2))
        for j in range(np.size(index_matches)):
            inpu_t[j] = feats_current[matches[index_matches[j]].queryIdx]
            outpu_t[j] = feats_new[matches[index_matches[j]].trainIdx]


        matches_img=cv2.drawMatches(ROI_current,kp_current,ROI_new,kp_new,matches,None)
        matche_name="DrawMatches/match_"+str(iterator)+".png"
        cv2.imwrite(matche_name,matches_img)
        ####################################################

        # inpu_t = np.zeros((np.size(matches),2))
        # outpu_t = np.zeros((np.size(matches),2))
        # # Rearrange the tab to order the features to perform Affine Transform 
        # for i in range(np.size(matches)):
        #     inpu_t[i] = feats_current[matches[i].queryIdx]
        #     outpu_t[i] = feats_new[matches[i].trainIdx]

        all_feats_current.append(inpu_t)
        all_feats_new.append(outpu_t)

        R,t=rigid_transform_3D(inpu_t, outpu_t)

        angle=np.arctan2(-R.item([1][0]),R.item([0][0]))
        angle_all.append(angle)
        tot_angle+=angle

        print "ANGLE:",angle

        # --> COMPUTE DISTANCE COVERED: 1) OR 2)
        #################################################### 
        ### 1) WITH CAMERA ROTATION
        t_axed_x=t.item([0][0])*np.cos(tot_angle)
        t_axed_y=t.item([1][0])*np.cos(tot_angle)
        t_tot_x+=t_axed_x
        t_tot_y+=t_axed_y
        print "TRANS X:",t_axed_x
        print "TRANS Y:",t_axed_y
        dist_tot_alter_x.append(t_tot_x)
        dist_tot_alter_y.append(t_tot_y)
        dist_x_.append(t_axed_x)
        dist_y_.append(t_axed_y)

        ### 2) WITHOUT CAMERA ROTATION
        # t_tot_x+=t.item([0][0])
        # t_tot_y+=t.item([1][0])
        # dist_tot_alter_x.append(t_tot_x)
        # dist_tot_alter_y.append(t_tot_y)
        # dist_x_.append(t.item([0][0]))
        # dist_y_.append(t.item([1][0]))
        #################################################### 
        ROI_current=ROI_new
        des_current=des_new
        kp_current=kp_new
        feats_current=feats_new
    else:
        break

# PLOT RESULTS
##################################################
fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale1=3
fontScale2=1
pt1=frame_pos_3[0] #(200,200)
pt2=tuple(map(operator.add,frame_pos_3[0],mask_x))
pt3=tuple(map(operator.add,frame_pos_3[0],mask_xy))
pt4=tuple(map(operator.add,frame_pos_3[0],mask_y))
pt1_rotated=tuple2array(pt1)
pt2_rotated=tuple2array(pt2)
pt3_rotated=tuple2array(pt3)
pt4_rotated=tuple2array(pt4)
current_point=np.array([pt1_rotated,pt2_rotated,pt3_rotated,pt4_rotated])
current_point_rotated=applyRotationTo(current_point, mask_x[0], mask_y[1], frame_pos_3[0][0], frame_pos_3[0][1], -angle_img_rot[0])

for i in range(4):
    img=cv2.imread("Total3.jpg")

    pt1_1=frame_pos_3[i+1]
    pt2_1=tuple(map(operator.add,frame_pos_3[i+1],mask_x))
    pt3_1=tuple(map(operator.add,frame_pos_3[i+1],mask_xy))
    pt4_1=tuple(map(operator.add,frame_pos_3[i+1],mask_y))
    pt1_1_rotated=tuple2array(pt1_1)
    pt2_1_rotated=tuple2array(pt2_1)
    pt3_1_rotated=tuple2array(pt3_1)
    pt4_1_rotated=tuple2array(pt4_1)
    new_point=np.array([pt1_1_rotated,pt2_1_rotated,pt3_1_rotated,pt4_1_rotated])
    #ADD MARGIN BLACK TO CHECK IF FEATURES ARE WITHIN
    tuple1=tuple(map(operator.add,frame_pos_3[i],(0.1*mask_xy[0],0.1*mask_xy[1])))
    tuple2=tuple(map(operator.add,frame_pos_3[i],(0.9*mask_xy[0],0.1*mask_xy[1])))
    tuple3=tuple(map(operator.add,frame_pos_3[i],(0.9*mask_xy[0],0.9*mask_xy[1])))
    tuple4=tuple(map(operator.add,frame_pos_3[i],(0.1*mask_xy[0],0.9*mask_xy[1])))
    margin=np.array([tuple1,tuple2,tuple3,tuple4])
    margin_rotated=applyRotationTo(margin, mask_x[0], mask_y[1], frame_pos_3[i][0], frame_pos_3[i][1], -angle_img_rot[i])
    new_point_rotated=applyRotationTo(new_point, mask_x[0], mask_y[1], frame_pos_3[i+1][0], frame_pos_3[i+1][1], -angle_img_rot[i+1])

    #TEST TO SEE IF ROTATION WORKS!
    # #PLOT CENTRE CURRENT
    cv2.circle(img,(int(mask_x[0]/2+frame_pos_3[i][0]),int(mask_y[1]/2+frame_pos_3[i][1])),60,(0,0,0),5)
    # #PLOT CENTRE NEW
    cv2.circle(img,(int(mask_x[0]/2+frame_pos_3[i+1][0]),int(mask_y[1]/2+frame_pos_3[i+1][1])),60,(0,0,0),20)
    # #PLOT in green current frame
    # cv2.polylines(img,np.int32([current_point]),True,(0,0,0),thickness=5)
    # #PLOT in blue new frame
    # cv2.polylines(img,np.int32([new_point]),True,(0,0,0),thickness=20)

    cv2.polylines(img,np.int32([current_point_rotated]),True,color[i],thickness=5)
    # 20% MARGIN FEATURES!
    cv2.polylines(img,np.int32([margin_rotated]),True,(0,0,0),thickness=8)

    cv2.polylines(img,np.int32([new_point_rotated]),True,color[i+1],thickness=5)
    cv2.putText(img, 'frame'+str(i), array2tuple_int(current_point_rotated[0]), fontFace, fontScale1, color[i],thickness=5)
    cv2.putText(img, 'frame'+str(i+1), array2tuple_int(new_point_rotated[2]), fontFace, fontScale1, color[i+1],thickness=5)

    for j in range(np.shape(all_feats_current[i])[0]):

        all_feats_current_2D=array2array2D(all_feats_current[i][j])
        all_feats_new_2D=array2array2D(all_feats_new[i][j])
        print all_feats_current_2D
        center_current=applyRotationTo(all_feats_current_2D, mask_x[0], mask_y[1], 0, 0, angle_img_rot[i])
        center_new=applyRotationTo(all_feats_current_2D, mask_x[0], mask_y[1], 0, 0, angle_img_rot[i+1])

        #add 0.1x 0.1y, because of ROI!
        center_current=tuple(map(operator.add,array2tuple2(center_current),tuple(map(operator.add,(0.1*mask_xy[0],0.1*mask_xy[1]),frame_pos_3[i]))))
        #center_new=tuple(map(operator.add,tuple(map(operator.add,array2tuple2(center_new),frame_pos_3[i])),displ_3[i]))
        center_new=tuple(map(operator.add,array2tuple2(center_new),tuple(map(operator.add,(0.1*mask_xy[0],0.1*mask_xy[1]),frame_pos_3[i]))))

        cv2.circle(img,(int(center_current[0]),int(center_current[1])),20,color[i],3)
        cv2.circle(img,(int(center_new[0]),int(center_new[1])),40,color[i+1],5)

        #center_new=tuple(map(operator.add,all_feats_new[i][j],frame_pos_3[i]))
        text_j0=tuple(map(operator.add,center_current,(40,40)))
        text_j1=tuple(map(operator.add,center_new,(40,-40)))
        text_j0=array2tuple(text_j0)
        text_j1=array2tuple(text_j1)
        text_j0=applyRotationTo(tuple2array2(text_j0), mask_x[0], mask_y[1], frame_pos_3[i][0], frame_pos_3[i][1], angle_img_rot[i])
        text_j1=applyRotationTo(tuple2array2(text_j1), mask_x[0], mask_y[1], frame_pos_3[i][0], frame_pos_3[i][1], angle_img_rot[i+1])
        cv2.putText(img, str(j), array2tuple_int2(text_j0), fontFace, fontScale2, color[i],thickness=3)
        cv2.putText(img, str(j), array2tuple_int2(text_j1), fontFace, fontScale2, color[i+1],thickness=3)



    cv2.imwrite("result-rot/match_"+str(i)+".png",img)

    center_current=center_new
    current_point=new_point
    current_point_rotated=new_point_rotated
    #
##################################################

ax=plt.plot(np.absolute(dist_tot_alter_x),np.absolute(dist_tot_alter_y),'r',label="Calculated")
x_data=[]
y_data=[]
for i in cummulative_displ3:
    x_data.append(i[0])
    y_data.append(i[1])
ax1=plt.plot(x_data[:np.size(x_data)-1],y_data[:np.size(x_data)-1],'bo',label="Real")
plt.legend(loc='upper left')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()

cv2.imread
cv2.destroyAllWindows()
#FILTERING DO AN AVERAGE OF THE % LAST VALUE, IS NEW VALUE COMPUTED IS NOT IN BETWEEN, PUT AVERAGE!
#FOR ANGLE AND DISTANCES
