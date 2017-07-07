from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator

class OpticalFlow:

	iterator=0
	def __init__(self, margin, frame_x, frame_y,angle,pos_x,pos_y,scale,result_filepath=None, data_filepath=None):
		self.margin=margin
		self.frame_x=frame_x
		self.frame_y=frame_y
		self.angle=angle
		self.pos_x=pos_x
		self.pos_y=pos_y
		self.scale=scale

	def formatting(self, key_points):
		totalTab=[]
		for i in range(np.size(key_points)):
			tab = [[np.float32(key_points[i].pt[0]), np.float32(key_points[i].pt[1])]]
			totalTab.append(tab)
		myarray = np.asarray(totalTab)
		return myarray
		
	def rad2deg(self, rad):
		deg=rad*180/3.14
		return deg

	def update(self, old_frame, current_frame):
		# Feature Extraction
		kp_old,des_old,feats_old=self.ExtractingFeaturesORB(old_frame)
		kp_current,des_current,feats_current=self.ExtractingFeaturesORB(current_frame)

		# Feature Matching
		inpu_t,outpu_t=self.MatchingFeatures(des_old, des_current, feats_old, feats_current)

		# Affine Transform
        R, t=self.AffineTransform(inpu_t, outpu_t)

        # Estimation Angle and Position
        angle_calc=self.rad2deg(np.arctan2(-R.item([1][0]),R.item([0][0])))
        self.angle+=angle_calc
        translation_x=t[0]*np.cos(self.angle)
        translation_y=t[1]*np.cos(self.angle)
        self.pos_x+=translation_x
        self.pos_y+=translation_y

		return (total_translation_x,total_translation_y), total_angle

    def ExtractingFeaturesORB(self, frame):
    	working_frame=frame[self.margin*self.frame_x:(1-self.margin)*self.frame_x , self.margin*self.frame_y:(1-self.margin)*self.frame_y]
        kp, des = orb.detectAndCompute(working_frame,mask=None)
        feats = self.formatting(kp)
    	return kp,des,feats

    def MatchingFeatures(self, des_old, des_current, feats_old, feats_current):
    	# Feature Matcher Init
    	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des_old,des_current, k=2)
        good = []
        # KNN Matching using David Lowe Criteria
        for m,n in matches:
            if m.distance < 0.45*n.distance:
                good.append([m])
        # Ordering Feature Matched
        inpu_t = np.zeros((np.shape(good)[0],2))
        outpu_t = np.zeros((np.shape(good)[0],2))
        for i in range(np.shape(good)[0]):
            inpu_t[i]=feats_old[good[i][0].queryIdx]
            outpu_t[i]=feats_current[good[i][0].trainIdx]
        return inpu_t,outpu_t

    def AffineTransform(self, inpu_t, outpu_t):
    	# Credit goes to Nghiaho, modification for this case made by me
	    assert len(inpu_t) == len(outpu_t)
	    N = inpu_t.shape[0]; # total points
	    centroid_input = np.mean(inpu_t, axis=0)
	    centroid_output = np.mean(outpu_t, axis=0)
	    # centre the points to their center of mass
	    AA = inpu_t - np.tile(centroid_A, (N, 1))
	    BB = outpu_t - np.tile(centroid_B, (N, 1))
	    H = np.dot(np.transpose(AA),BB)
	    U, S, Vt = np.linalg.svd(H)
	    R = np.dot(Vt.T,U.T)
	    # Recenter rotation to centre of frame and not to the left top corner
	    recenter=centroid_input-[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
	    recenter=np.dot(R,recenter)+[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
	    t= centroid_output-recenter
	    return R, t

	def RandT_2_angleAndProjection(R,t):
		angle=rad2deg(np.arctan2(-R.item([1][0]),R.item([0][0]))))
		self.angle+=angle
		self.pos_x=t[0]*cos(self.angle)
		self.pos_y=t[1]*cos(self.angle)
		estimated_pos_x_tab.append(projected_trans_x)
		estimated_pos_y_tab.append(projected_trans_y)
		estimated_orientation_tab.append(self.angle)
		return estimated_pos_x_tab,estimated_pos_y_tab,estimated_orientation_tab

	def create_dataframe(self, pos_x=0, pos_y=0, angle=0):
	    frame=cv2.imread("Total3.jpg",0)
	    (rows,cols)=np.shape(frame)
	    M=cv2.getRotationMatrix2D((pos_x+self.frame_x/2,y+pos_self.frame_y/2), angle, self.scale)
	    frame=cv2.warpAffine(frame,M,(rows,cols))
	    mask=frame[y:y+self.frame_y,x:x+self.frame_x]
	    return mask

	def run_test(self, pos_x_tab, pos_y_tab ,orientation_tab):
		for i in range(np.shape(pos_x_tab)[0]):
			img_0 = self.create_dataframe(pos_x_tab[i], pos_y_tab[i], orientation_tab[i])
			img_1 = self.create_dataframe(pos_x_tab[i+1], pos_y_tab[i+1], orientation_tab[i+1])

			kp_0,des_0,feats_0 = self.ExtractingFeaturesORB(img_0)
			kp_1,des_1,feats_1 = self.ExtractingFeaturesORB(img_1)

			matched_0,matched_1 = MatchingFeatures(self, des_0, des_1, feats_0, feats_1)

			R,t = self.AffineTransform(matched_0, matched_1)

			estimated_pos_x_tab,estimated_pos_y_tab,estimated_orientation_tab = RandT_2_angleAndProjection(R,t)

		plot_visualization(pos_x_tab,pos_y_tab,orientation_tab, estimated_pos_x_tab, estimated_pos_y_tab, estimated_orientation_tab)



	def plot_visualization(self, pos_x_tab,pos_y_tab,orientation_tab, estimated_pos_x_tab, estimated_pos_y_tab, estimated_orientation_tab):
		# ADD VISUALIZATION



# INIT VARIABLES
old_frame=cv2.imread("img-2D-rot/image_"+str(0)+".png",0)
(frame_x,frame_y)=np.shape(img)
margin_ROI=0.1
angle=0
pos_x=0
pos_y=0
scale=1

#Creation object of Class OpticalFlow
optflow=OpticalFlow(margin_ROI, frame_x, frame_y, angle, pos_x, pos_y,scale) 

iterator=-1
all_pos_robot=[]

while(1):
	iterator+=1
	frame=cv2.imread("img-2D-rot/image_"+str(iterator)+".png",0)
	if frame is not None:
		pos_robot, orientation_robot=optflow.update(old_frame,new_frame)
		all_pos_robot.append(pos_robot)
		all_orientation_robot.append(orientation_robot)
		old_frame=new_frame
	else:
		break

optflow.plot_visualization(all_pos_robot, all_orientation_robot)
