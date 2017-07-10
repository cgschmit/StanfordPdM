from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator

class OpticalFlow:

	def __init__(self, margin, frame_x, frame_y,angle,pos_x,pos_y,scale,result_filepath=None, data_filepath=None):
		self.margin=margin
		self.frame_x=frame_x
		self.frame_y=frame_y
		self.angle=angle
		self.pos_x=pos_x
		self.pos_y=pos_y
		self.scale=scale

		feature_params_orb = dict ( nfeatures=50,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=20, 
                            firstLevel=0, 
                            WTA_K=4, 
                            patchSize=30
                            )

		self.orb = cv2.ORB_create(**feature_params_orb)


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
		working_frame=frame[int(self.margin*self.frame_x):int((1-self.margin)*self.frame_x) , int(self.margin*self.frame_y):int((1-self.margin)*self.frame_y)]
		kp, des = self.orb.detectAndCompute(working_frame,mask=None)
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
		AA = inpu_t - np.tile(centroid_input, (N, 1))
		BB = outpu_t - np.tile(centroid_output, (N, 1))
		H = np.dot(np.transpose(AA),BB)
		U, S, Vt = np.linalg.svd(H)
		R = np.dot(Vt.T,U.T)
		# Recenter rotation to centre of frame and not to the left top corner
		recenter=centroid_input-[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
		recenter=np.dot(R,recenter)+[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
		t= centroid_output-recenter
		return R, t

	def actualize_pos_orientation(self,R,t):
		angle=self.rad2deg(np.arctan2(-R.item([1][0]),R.item([0][0])))
		self.angle+=angle
		self.pos_x=t[0]*np.cos(self.angle)
		self.pos_y=t[1]*np.cos(self.angle)
		return 

	def create_dataframe(self, pos_x=0, pos_y=0, angle=0):
		frame=cv2.imread("Total3.jpg",0)
		(rows,cols)=np.shape(frame)
		M=cv2.getRotationMatrix2D((pos_x+self.frame_x/2,pos_y+self.frame_y/2), angle, self.scale)
		frame=cv2.warpAffine(frame,M,(rows,cols))
		mask=frame[pos_y:pos_y+self.frame_y,pos_x:pos_x+self.frame_x]
		return mask

	def run_test(self, pos_x_tab, pos_y_tab ,orientation_tab):
		estimated_pos_x_tab=[]
		estimated_pos_y_tab=[]
		estimated_orientation_tab=[]
		for i in range(np.shape(pos_x_tab)[0]-1):
			img_0=self.create_dataframe(pos_x_tab[i], pos_y_tab[i], orientation_tab[i])
			img_1=self.create_dataframe(pos_x_tab[i+1], pos_y_tab[i+1], orientation_tab[i+1])

			kp_0,des_0,feats_0=self.ExtractingFeaturesORB(img_0)
			kp_1,des_1,feats_1=self.ExtractingFeaturesORB(img_1)

			matched_0,matched_1=self.MatchingFeatures(des_0, des_1, feats_0, feats_1)

			if np.shape(matched_0)[0]==0:
				self.estimating()
			else:
				R,t=self.AffineTransform(matched_0, matched_1)
				self.actualize_pos_orientation(R,t)

			estimated_pos_x_tab.append(self.pos_x)
			estimated_pos_y_tab.append(self.pos_y)
			estimated_orientation_tab.append(self.angle)

		self.plot_visualization(pos_x_tab,pos_y_tab,orientation_tab, estimated_pos_x_tab, estimated_pos_y_tab, estimated_orientation_tab)


	def estimating(self):
		# DO SOME KALMAN FILTERING TO ESTIMATED VALUE OF UNFOUND MATHCING FEATURES DISPLACEMENT'S
		self.pos_x+=0
		self.pos_y+=0
		self.angle+=0
		return 

	def plot_visualization(self, pos_x_tab,pos_y_tab,orientation_tab, estimated_pos_x_tab, estimated_pos_y_tab, estimated_orientation_tab):
		img=cv2.imread("Total3.jpg",1)
		fontFace=cv2.FONT_HERSHEY_SIMPLEX
		fontScale=1.5

		color_real=(0,255,0)
		color_estimation=(255,0,0)

		### PLOT REAL DATA
		cv2.circle(img,(int(pos_x_tab[0]),int(pos_y_tab[0])),30,color_real,5)
		#Draw orientation of robot
		orientation_line=tuple(map(operator.add,(pos_x_tab[0],pos_y_tab[0]),(30*np.cos(orientation_tab[0]),30*np.sin(orientation_tab[0]))))
		cv2.line(img,(int(pos_x_tab[0]),int(pos_y_tab[0])),(int(orientation_line[0]),int(orientation_line[1])),color_real, 3)
		for i in range(np.shape(pos_x_tab)[0]-1):
			cv2.circle(img,(int(pos_x_tab[i+1]),int(pos_y_tab[i+1])),30,color_real,5)
			# Add text next to each circle plot
			text_pos=tuple(map(operator.add,(int(pos_x_tab[i+1]),int(pos_y_tab[i+1])),(40,40)))
			cv2.putText(img, str(i+1), text_pos, fontFace, fontScale, color_real,thickness=3)
			#Draw line from one point to an other
			#cv2.line(img, (int(pos_x_tab[i]),int(pos_y_tab[i])), (int(pos_x_tab[i+1]),int(pos_y_tab[i+1])), color_real, 3) 
			#Draw orientation of robot
			orientation_line=tuple(map(operator.add,(pos_x_tab[i+1],pos_y_tab[i+1]),(30*np.cos(orientation_tab[i+1]),30*np.sin(orientation_tab[i+1]))))

			cv2.line(img,(int(pos_x_tab[i+1]),int(pos_y_tab[i+1])),(int(orientation_line[0]),int(orientation_line[1])),color_real, 3)

		### PLOT ESTIMATED DATA
		cv2.circle(img,(int(estimated_pos_x_tab[0]),int(estimated_pos_y_tab[0])),30,color_estimation,5)
		#Draw orientation of robot
		orientation_line=tuple(map(operator.add,(estimated_pos_x_tab[0],estimated_pos_y_tab[0]),(30*np.cos(estimated_orientation_tab[0]),30*np.sin(estimated_orientation_tab[0]))))
		cv2.line(img,(int(estimated_pos_x_tab[0]),int(estimated_pos_y_tab[0])),(int(orientation_line[0]),int(orientation_line[1])),color_estimation, 3)
		# print "np.shape(estimated_pos_x_tab)[0]-1",np.shape(estimated_pos_x_tab)[0]-1
		for i in range(np.shape(estimated_pos_x_tab)[0]-1):
			# print "estimated_pos_x_tab[i+1] = ",estimated_pos_x_tab[i+1]
			# print "estimated_pos_y_tab[i+1] = ",estimated_pos_y_tab[i+1]
			# print "estimated_orientation_tab[i+1] = ",estimated_orientation_tab[i+1]
			cv2.circle(img,(int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])),30,color_estimation,5)
			# Add text next to each circle plot
			text_pos=tuple(map(operator.add,(int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])),(40,40)))
			cv2.putText(img, str(i+1), text_pos, fontFace, fontScale, color_estimation,thickness=3)
			#Draw line from one point to an other
			#cv2.line(img, (int(estimated_pos_x_tab[i]),int(estimated_pos_y_tab[i])), (int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])), color_estimation, 3) 
			#Draw orientation of robot
			orientation_line=tuple(map(operator.add,(estimated_pos_x_tab[i+1],estimated_pos_y_tab[i+1]),(30*np.cos(estimated_orientation_tab[i+1]),30*np.sin(estimated_orientation_tab[i+1]))))
			cv2.line(img,(int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])),(int(orientation_line[0]),int(orientation_line[1])),color_estimation, 3)

		cv2.imshow("img",img)
		cv2.waitKey(0)

# INIT VARIABLES
frame_x=600
frame_y=600
margin_ROI=0.1
angle=0
pos_x=0
pos_y=0
scale=1

#Creation object of Class OpticalFlow
optflow=OpticalFlow(margin_ROI, frame_x, frame_y, angle, pos_x, pos_y, scale) 

#Data
pos_x_tab=np.array([100,200,300,400,500,600,700,800,900,1000])
pos_y_tab=np.array([100,200,300,400,500,600,700,800,900,1000])
orientation_tab=np.array([10,20,30,20,10,0,10,20,30,20])

#Run Tets
optflow.run_test(pos_x_tab, pos_y_tab, orientation_tab)


