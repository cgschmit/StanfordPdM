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
		self.estimated_pos_x_tab=[]
		self.estimated_pos_y_tab=[]
		self.estimated_orientation_tab=[]

		feature_params_orb = dict ( nfeatures=100,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=40, 
                            firstLevel=0, 
                            WTA_K=4, 
                            patchSize=60
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

	def deg2rad(self, deg):
		rad=deg*3.14/180
		return rad

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
			if m.distance < 0.4*n.distance:
				good.append([m])
		# Ordering Feature Matched
		inpu_t = np.zeros((np.shape(good)[0],2))
		outpu_t = np.zeros((np.shape(good)[0],2))
		for i in range(np.shape(good)[0]):
			inpu_t[i]=feats_old[good[i][0].queryIdx]
			outpu_t[i]=feats_current[good[i][0].trainIdx]
		return inpu_t,outpu_t,good

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
		#R = np.dot(Vt.T,U.T)
		R = np.dot(Vt.T,U.T)
		# Recenter rotation to centre of frame and not to the left top corner
		recenter=centroid_output-[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
		recenter=np.dot(np.linalg.inv(R),recenter)+[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
		t=centroid_input.T-recenter
		return R, t

	def actualize_pos_orientation(self,R,t):
		local_angle=np.arctan2(R.item([1][0]),R.item([0][0]))
		local_x=t[0]*np.cos(self.angle)
		local_y=t[1]*np.cos(self.angle)
		self.pos_x+=local_x
		self.pos_y+=local_y
		self.angle+=local_angle
		

	def local2global(self,local_x, local_y, global_teta=0, global_position_frame_x=0, global_position_frame_y=0):
	    # local_x/y: position of feature in frame referential
	    # global_teta: total angle of the frame (global referential)
	    # position_frame_x/y: position of the top-left corner of the frame regarding the total displacement (global referential)
	    global_x=global_position_frame_x+np.sqrt(np.power(local_x,2)+np.power(local_y,2))*np.sin(np.arctan2(local_x,local_y)-self.deg2rad(global_teta))
	    global_y=global_position_frame_y+np.sqrt(np.power(local_x,2)+np.power(local_y,2))*np.cos(np.arctan2(local_x,local_y)-self.deg2rad(global_teta))
	    return global_x, global_y

	def create_dataframe(self, pos_x=0, pos_y=0, angle=0):
		frame=cv2.imread("Total3.jpg",0)
		(rows,cols)=np.shape(frame)
		#angle need to be in degree
		M=cv2.getRotationMatrix2D((pos_x+self.frame_x/2,pos_y+self.frame_y/2), self.rad2deg(angle), self.scale)
		frame=cv2.warpAffine(frame,M,(cols,rows))
		mask=frame[pos_y:pos_y+self.frame_y,pos_x:pos_x+self.frame_x]
		return mask

	def run_test(self, pos_x_tab, pos_y_tab ,orientation_tab):
		# orientation_tab is in Degree
		self.estimated_pos_x_tab.append(self.pos_x)
		self.estimated_pos_y_tab.append(self.pos_y)
		self.estimated_orientation_tab.append(self.angle)
		for i in range(np.shape(pos_x_tab)[0]-1):
			print "ITERATION",i
			img_0=self.create_dataframe(pos_x_tab[i], pos_y_tab[i], orientation_tab[i])
			img_1=self.create_dataframe(pos_x_tab[i+1], pos_y_tab[i+1], orientation_tab[i+1])

			kp_0,des_0,feats_0=self.ExtractingFeaturesORB(img_0)
			kp_1,des_1,feats_1=self.ExtractingFeaturesORB(img_1)

			matched_0,matched_1,good=self.MatchingFeatures(des_0, des_1, feats_0, feats_1)

			drawMatches=cv2.drawMatchesKnn(img_0, kp_0, img_1, kp_1, good, None, flags=2)
			cv2.imwrite("matchedFeatures/match_"+str(i)+".png",drawMatches)

			if np.shape(matched_0)[0]<=2:
				print "ESTIMATION TO BE DONE; WORK IN PROGRESS"
				self.estimating(i)
			else:
				R,t=self.AffineTransform(matched_0, matched_1)
				new_t=self.local2global(t[0],t[1],self.angle,0,0)
				self.actualize_pos_orientation(R,new_t)
			print "---------------"
			print "self.pos_x (TOTAL DISPLACEMENT) = ",self.pos_x
			print "self.pos_y (TOTAL DISPLACEMENT) = ",self.pos_y
			print "self.angle (TOTAL DISPLACEMENT) = ",self.rad2deg(self.angle)
			print "---------------"

			self.estimated_pos_x_tab.append(self.pos_x)
			self.estimated_pos_y_tab.append(self.pos_y)
			self.estimated_orientation_tab.append(self.angle)

		print "estimated_pos_x_tab = ",self.estimated_pos_x_tab
		print "estimated_pos_y_tab = ",self.estimated_pos_y_tab
		print "estimated_orientation_tab = ",self.estimated_orientation_tab

		self.plot_visualization(pos_x_tab,pos_y_tab,orientation_tab)

	def estimating(self,i):
		# DO SOME FILTERING TO ESTIMATED VALUE OF UNFOUND MATHCING FEATURES DISPLACEMENT'S
		if np.shape(self.estimated_pos_x_tab)[0]>2:
			self.pos_x+=(0.7*(self.estimated_pos_x_tab[i]-self.estimated_pos_x_tab[i-1])+0.3*(self.estimated_pos_x_tab[i-1]-self.estimated_pos_x_tab[i-2]))
			self.pos_y+=(0.7*(self.estimated_pos_y_tab[i]-self.estimated_pos_y_tab[i-1])+0.3*(self.estimated_pos_y_tab[i-1]-self.estimated_pos_y_tab[i-2]))
			self.angle+=(0.7*(self.estimated_orientation_tab[i]-self.estimated_orientation_tab[i-1])+0.3*(self.estimated_orientation_tab[i-1]-self.estimated_orientation_tab[i-2]))
		elif np.shape(self.estimated_pos_x_tab)[0]==2:
			self.pos_x+=(self.estimated_pos_x_tab[i]-self.estimated_pos_x_tab[i-1])
			self.pos_y+=(self.estimated_pos_y_tab[i]-self.estimated_pos_y_tab[i-1])
			self.angle+=(self.estimated_orientation_tab[i]-self.estimated_orientation_tab[i-1])
		elif np.shape(self.estimated_pos_x_tab)[0]==0 or np.shape(self.estimated_pos_x_tab)[0]==1:
			self.pos_x+=0
			self.pos_y+=0
			self.angle+=0
		return 

	def plot_visualization(self, pos_x_tab,pos_y_tab,orientation_tab):
		img=cv2.imread("Total3.jpg",1)
		fontFace=cv2.FONT_HERSHEY_SIMPLEX
		fontScale=1.5

		color_real=(0,255,0)
		color_estimation=(0,0,255)
			
		### PLOT REAL DATA
		cv2.circle(img,(int(pos_x_tab[0]),int(pos_y_tab[0])),35,color_real,5)
		text_pos=tuple(map(operator.add,(int(pos_x_tab[0]),int(pos_y_tab[0])),(40,-10)))
		cv2.putText(img, str(0), text_pos, fontFace, fontScale, color_real,thickness=3)
		orientation_line=tuple(map(operator.add,(pos_x_tab[0],pos_y_tab[0]),(30*np.cos(orientation_tab[0]),30*np.sin(orientation_tab[0]))))
		cv2.line(img,(int(pos_x_tab[0]),int(pos_y_tab[0])),(int(orientation_line[0]),int(orientation_line[1])),color_real, 6)
		for i in range(np.shape(pos_x_tab)[0]-1):
			cv2.circle(img,(int(pos_x_tab[i+1]),int(pos_y_tab[i+1])),35,color_real,5)
			text_pos=tuple(map(operator.add,(int(pos_x_tab[i+1]),int(pos_y_tab[i+1])),(40,-10)))
			cv2.putText(img, str(i+1), text_pos, fontFace, fontScale, color_real,thickness=3)
			#cv2.line(img, (int(pos_x_tab[i]),int(pos_y_tab[i])), (int(pos_x_tab[i+1]),int(pos_y_tab[i+1])), color_real, 3) 
			orientation_line=tuple(map(operator.add,(pos_x_tab[i+1],pos_y_tab[i+1]),(30*np.cos(orientation_tab[i+1]),30*np.sin(orientation_tab[i+1]))))

			cv2.line(img,(int(pos_x_tab[i+1]),int(pos_y_tab[i+1])),(int(orientation_line[0]),int(orientation_line[1])),color_real, 6)

		### PLOT ESTIMATED DATA
		cv2.circle(img,(int(self.estimated_pos_x_tab[0]),int(self.estimated_pos_y_tab[0])),30,color_estimation,5)
		text_pos=tuple(map(operator.add,(int(self.estimated_pos_x_tab[0]),int(self.estimated_pos_y_tab[0])),(40,40)))
		cv2.putText(img, str(0), text_pos, fontFace, fontScale, color_estimation,thickness=3)
		orientation_line=tuple(map(operator.add,(self.estimated_pos_x_tab[0],self.estimated_pos_y_tab[0]),(30*np.cos(self.estimated_orientation_tab[0]),30*np.sin(self.estimated_orientation_tab[0]))))
		cv2.line(img,(int(self.estimated_pos_x_tab[0]),int(self.estimated_pos_y_tab[0])),(int(orientation_line[0]),int(orientation_line[1])),color_estimation, 3)
		for i in range(np.shape(self.estimated_pos_x_tab)[0]-1):
			cv2.circle(img,(int(self.estimated_pos_x_tab[i+1]),int(self.estimated_pos_y_tab[i+1])),30,color_estimation,5)
			text_pos=tuple(map(operator.add,(int(self.estimated_pos_x_tab[i+1]),int(self.estimated_pos_y_tab[i+1])),(40,40)))
			cv2.putText(img, str(i+1), text_pos, fontFace, fontScale, color_estimation,thickness=3)
			#cv2.line(img, (int(estimated_pos_x_tab[i]),int(estimated_pos_y_tab[i])), (int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])), color_estimation, 3) 
			orientation_line=tuple(map(operator.add,(self.estimated_pos_x_tab[i+1],self.estimated_pos_y_tab[i+1]),(30*np.cos(self.estimated_orientation_tab[i+1]),30*np.sin(self.estimated_orientation_tab[i+1]))))
			cv2.line(img,(int(self.estimated_pos_x_tab[i+1]),int(self.estimated_pos_y_tab[i+1])),(int(orientation_line[0]),int(orientation_line[1])),color_estimation, 3)

		cv2.imwrite("result2.png",img)
		cv2.imshow("img",img)
		cv2.waitKey(0)

# INIT VARIABLES
frame_x=600
frame_y=600
margin_ROI=0
angle=0
pos_x=2000
pos_y=2000
scale=1

#Creation object of Class OpticalFlow
optflow=OpticalFlow(margin_ROI, frame_x, frame_y, angle, pos_x, pos_y, scale) 

#Data
#pos_x_tab=np.array([100,200,300,400,500,600,700,800,900,1000])
#pos_y_tab=np.array([100,200,300,400,500,600,700,800,900,1000])
#pos_x_tab=np.array([800,900,1000,1100,1200,1300,1400,1500,1600,1700])
#pos_y_tab=np.array([800,800,800,800,800,800,800,800,800,800])
#pos_y_tab=np.array([800,900,1000,1100,1200,1300,1400,1500,1600,1700])
#orientation_tab=np.array([30,20,10,0,10,20,30,20,10,0],np.float64)
#orientation_tab=np.array([0,0,0,0,0,0,0,0,0,0],np.float64)

pos_x_tab=np.array([2000,2050,2100,2150])
pos_y_tab=np.array([2000,2050,2100,2150])
orientation_tab=np.array([0,30,30,20],np.float64)

# Convertion to rad!
orientation_tab*=np.tile(3.14/180,np.shape(orientation_tab)[0])

#Run Test
optflow.run_test(pos_x_tab, pos_y_tab, orientation_tab)


