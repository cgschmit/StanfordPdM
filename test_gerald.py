import numpy as np
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


# Beware that the rotation is not working for frame that are not squared. So keep the frame squared.
def createMovingMask(x=0, y=0,teta=0,scale=1, frame_x=300, frame_y=300):
    frame=cv2.imread("Total3.jpg",0)
    (rows,cols)=np.shape(frame)
    M=cv2.getRotationMatrix2D((x+frame_x/2,y+frame_y/2), teta, scale)
    frame=cv2.warpAffine(frame,M,(rows,cols))
    mask=frame[x:x+frame_x,y:y+frame_y]
    return mask


img=cv2.imread("Total3.jpg")


# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

coords_0 = [100, 200, 0]
coords_1 = [100, 200, 0]
size = 600

# img_0 = img[coords_0[0]:coords_0[0]+size, coords_0[1]:coords_0[1]+size, :]
# img_1 = img[coords_1[0]:coords_1[0]+size, coords_1[1]:coords_1[1]+size, :]

img_0 = createMovingMask(coords_0[0], coords_0[1], teta=0, scale=1, frame_x=size, frame_y=size)
img_1 = createMovingMask(coords_1[0], coords_1[1], teta=10, scale=1, frame_x=size, frame_y=size)



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
                            patchSize=30
                            )


# frame_gray=cv2.imread("img-2D-rot/image_0.png",0)
# (m_frame_x,m_frame_y)=np.shape(frame_gray)
# margin=0.1
# edge1=int(m_frame_x*margin)
# edge2=int(m_frame_x*(1-margin))
# edge3=int(m_frame_y*margin)
# edge4=int(m_frame_y*(1-margin))
# ROI_current=frame_gray[edge1:edge2, edge3:edge4]

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
    if m.distance < 0.5*n.distance:
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

# ------------------------------------
# Draw circles on masks
# ------------------------------------

fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale1=2

# for j in range(coords_reordered_0.shape[0]):
for j in range(5):
    print 'j = ', j
    coords_reordered_centers_0 = array2tuple(coords_reordered_0[j], ret_int=True)
    coords_reordered_centers_1 = array2tuple(coords_reordered_1[j], ret_int=True)

    cv2.circle(img_0, coords_reordered_centers_0, 30, (0,255,0), 5)
    cv2.circle(img_1, coords_reordered_centers_1, 30, (0,255,0), 5)
    cv2.circle(img_0, (300,300), 10, (255,255,255), 2)
    cv2.circle(img_1, (300,300), 10, (255,255,255), 2)

    text_j0=tuple(map(operator.add, coords_reordered_centers_0, (40,40)))
    cv2.putText(img_0, str(j), (int(text_j0[0]),int(text_j0[1])), fontFace, fontScale1, (0,255,0),thickness=3)

    text_j1 = tuple(map(operator.add, coords_reordered_centers_1, (40,40)))
    cv2.putText(img_1, str(j), (int(text_j1[0]),int(text_j1[1])), fontFace, fontScale1, (0,255,0),thickness=3)


print 'type(coords_reordered_centers_0) = ', type(coords_reordered_centers_0)

cv2.imshow('img_0',img_0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cv2.imshow('img_1',img_1)
cv2.waitKey(0)
# cv2.destroyAllWindows()


# ------------------------------------
# Compute displacement (x, y q)
# ------------------------------------

print 'coords_reordered_0 = ', coords_reordered_0.shape

A = coords_reordered_0[:5]
B = coords_reordered_1[:5]

# print 'type(A) = ', type(A)

for i in range(5):
    print 'i = ', i
    coords_reordered_0[i]=tuple2array(coords_reordered_0[i], False)
    coords_reordered_1[i]=tuple2array(coords_reordered_1[i], False)


R, t = rigid_transform_3D(A, B, frame_x=600, frame_y=600, margin=0)

print 'R = ', R
print 't = ', t



222











