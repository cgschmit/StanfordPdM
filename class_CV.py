from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator
import math


class OpticalFlow:

	def __init__(self, margin, frame_x, frame_y, angle, pos_x, pos_y, scale):
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
                            edgeThreshold=20, 
                            firstLevel=0, 
                            WTA_K=4, 
                            patchSize=60
                            )

		self.orb = cv2.ORB_create(**feature_params_orb)

	def createFrame(self, x, y, teta):
		frame=cv2.imread("Total5.jpg",0)
		(rows,cols)=np.shape(frame)
		#Rotation of image
		M=cv2.getRotationMatrix2D((x+self.frame_x/2,y+self.frame_y/2), self.rad2deg(teta), self.scale)
		frame_rotated=cv2.warpAffine(frame,M,(cols,rows))
		#Select Frame
		mask_rotated=frame_rotated[int(np.rint(y)):int(np.rint(y+self.frame_y)),int(np.rint(x)):int(np.rint(x+self.frame_x))]    
		return mask_rotated

	def extractingFeatures_ORB(self, img):
		kp, des = self.orb.detectAndCompute(img,mask=None)
		feats = self.keypoints2feature(kp)
		return kp, des, feats

	def keypoints2feature(self, kp):
		totalTab=[]
		for i in range(np.shape(kp)[0]):
			tab = [[np.float32(kp[i].pt[0]), np.float32(kp[i].pt[1])]]
			totalTab.append(tab)
		myarray = np.asarray(totalTab)
		return myarray

	def featureMatching(self, des_0, des_1, feats_0, feats_1, featureMatchingSensitivity = 0.55):
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
		matches = bf.knnMatch(des_0, des_1, k = 2)
		good = []
		for m,n in matches:
			if m.distance < featureMatchingSensitivity*n.distance:
				good.append([m])
		coords_reordered_0 = np.zeros((np.shape(good)[0],2))
		coords_reordered_1 = np.zeros((np.shape(good)[0],2))
		for i in range(np.shape(good)[0]):
			coords_reordered_0[i]=feats_0[good[i][0].queryIdx]
			coords_reordered_1[i]=feats_1[good[i][0].trainIdx]
		return coords_reordered_0, coords_reordered_1, good

	def filterAffineTransform(self, A, B, method="RANSAC"):
		### Select the training features (to calculate R and t)
		outlier_ratio=0.2 #experimental value
		prob_success=0.99 #probability that one 
		number_sample=3
		number_trials=int(np.ceil(np.log(1-prob_success)/np.log(1-np.power((1-outlier_ratio),number_sample)))) #formula RANSAC
		init=True

		for i in range(number_trials):
			idx=np.asarray(list(range(np.shape(A)[0])))
			np.random.shuffle(idx)
			idx=idx[:3]
			A_rand=A[idx]
			B_rand=B[idx]

			### Select all the other features, to test
			mask=np.ones(np.shape(A)[0],np.bool)
			mask[idx]=0
			idx_test=np.asarray(list(range(np.shape(A)[0])))
			idx_test=idx_test[mask]
			A_test=A[idx_test]
			B_test=B[idx_test]

			### Call affineTransform
			R,t=self.affineTransform(A_rand,B_rand)
			
			# B TO A
			vecto_r=B_test-np.tile([self.frame_x/2,self.frame_y/2],(np.shape(B_test)[0],1))
			A_estimated=np.dot(np.linalg.inv(R),vecto_r.T)
			A_estimated=A_estimated.T+np.tile(t,(np.shape(B_test)[0],1))+np.tile([self.frame_x/2,self.frame_y/2],(np.shape(A_test)[0],1))


			distances=np.sqrt(np.power(A_test[:,0]-A_estimated[:,0],2)+np.power(A_test[:,1]-A_estimated[:,1],2))
			
			# METHOD 1
			# Choose the R, t minimizing total distance after projection
			if method == "distance":
				total_distance=np.sum(distances)			
				if init:
					best_total_distance=total_distance
					R_best=R
					t_best=t
					init=False
					A_test_best=A_test
					A_estimated_best=A_estimated

				if total_distance<best_total_distance:
					R_best=R
					t_best=t
					best_total_distance=total_distance
					A_test_best=A_test
					A_estimated_best=A_estimated

			# METHOD 2
			# Choose the R, t that maximize the total number of inlier.
			elif method == "RANSAC":
				counter=0
				for j in range(np.shape(A_test)[0]):
					if distances[j]<2: #Test value, can be changed to anything. In px ! The smaller, the more accurate.
						counter+=1
				if init:
					best_counter=counter
					R_best=R
					t_best=t
					init=False
					A_test_best=A_test
					A_estimated_best=A_estimated
				if counter>best_counter:
					best_counter=counter
					R_best=R
					t_best=t
					A_test_best=A_test
					A_estimated_best=A_estimated

		return R_best, t_best

	def affineTransform(self, A, B):
		assert len(A) == len(B)
		N = A.shape[0]
		centroid_A = np.mean(A, axis=0, dtype=np.float64)
		centroid_B = np.mean(B, axis=0, dtype=np.float64)
		AA = A - np.tile(centroid_A, (N, 1))
		BB = B - np.tile(centroid_B, (N, 1))
		H = np.dot(np.transpose(AA),BB)
		U, S, Vt = np.linalg.svd(H)
		R = np.dot(Vt.T,U.T)
		recenter=centroid_B-[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
		#recenter=np.dot(np.linalg.inv(R),recenter)+[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
		recenter=np.dot(np.dot(U.T,Vt.T),recenter)+[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
		t=-recenter+centroid_A.T
		return R, t

	def weightedAffineTransform(self, A, B, distance, norm_type="L2", method="inverse"):
		# source: ETHZ: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
		sum_=0
		if norm_type=="L2":
			if method=="euler":
				weights=np.exp(-distance/100)
				for i in range(np.shape(weights)[0]):
					sum_+=np.power(weights[i],2)
				sum_=np.sqrt(sum_)
				weights_norm=weights/sum_
				weights=weights_norm
			elif method =="inverse":
				# TO AVOID SINGULARITY WITH DISTANCE==0, WE ADDED +0.05
				weights=1/(distance+0.05)
				for i in range(np.shape(weights)[0]):
					sum_+=np.power(weights[i],2)
				sum_=np.sqrt(sum_)
				weights_norm=weights/sum_
				weights=weights_norm
		elif norm_type=="L1":
			if method=="euler":
				weights=np.exp(-distance/100)
				sum_=np.sum(weights)
				weights_norm=weights/sum_
				weights=weights_norm
			elif method =="inverse":
				weights=1/(distance+0.05)
				sum_=np.sum(weights)
				weights_norm=weights/sum_
				weights=weights_norm

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
		#id_[M-1,M-1]=np.linalg.det(np.dot(Vt.T,U.T))
		R=np.dot(np.dot(Vt.T,id_),U.T)
		recenter=centroid_B-[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
		recenter=np.dot(np.linalg.inv(R),recenter)+[(1-2*margin)/2*frame_x,(1-2*margin)/2*frame_y]
		t=-recenter+centroid_A.T

		return R,t

	def affineTransform3d(self,A, B):
		#Perform getAffineTransform of two vertices to get scale, orientation, translation with the openCV function
		M=cv2.getAffineTransform(A, B)
		t=np.array([M[0][2],M[1][2]])
		s_x=np.sqrt(np.power(M[0][0],2)+np.power(M[1][0],2))
		s_y=np.sqrt(np.power(M[0][1],2)+np.power(M[1][1],2))
		s=np.array([[s_x,0],[0,s_y]])
		R=np.array([[M[0][0]/s_x,M[0][1]/s_y],[M[1][0]/s_x,M[1][1]/s_y]])
		return R,s,t

	def local2global(self, t, pos_frame_global_x=0, pos_frame_global_y=0):
		global_x_est=pos_frame_global_x+np.sqrt(np.power(t[0],2,dtype=np.float64)+np.power(t[1],2,dtype=np.float64),dtype=np.float64)*np.sin(np.arctan2(t[0],t[1])-self.angle,dtype=np.float64)
		global_y_est=pos_frame_global_y+np.sqrt(np.power(t[0],2,dtype=np.float64)+np.power(t[1],2,dtype=np.float64),dtype=np.float64)*np.cos(np.arctan2(t[0],t[1])-self.angle,dtype=np.float64)
		return global_x_est, global_y_est

	def deg2rad(self,deg):
		rad=deg*math.pi/180
		return rad

	def rad2deg(self,rad):
		deg=rad*180/math.pi
		return deg

	def update(self, R, global_x_est, global_y_est):
		local_angle=np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64)
		print "########################################"
		print "INCREMENT ANGLE = ",self.rad2deg(local_angle)
		print "INCREMENT X = ",global_x_est
		print "INCREMENT Y = ",global_y_est
		self.pos_x+=global_x_est
		self.pos_y+=global_y_est
		self.angle+=local_angle
		return

	def run_test(self,global_x_tab,global_y_tab,global_orientation_tab):
		estimated_pos_x_tab=[]
		estimated_pos_y_tab=[]
		estimated_orientation_tab=[]
		for i in range(np.shape(global_x_tab)[0]-1):
			print "ITERATION NUMBER: ",(i+1)
			img_0=self.createFrame(global_x_tab[i],global_y_tab[i],global_orientation_tab[i])
			img_1=self.createFrame(global_x_tab[i+1],global_y_tab[i+1],global_orientation_tab[i+1])

			kp_0,des_0,feats_0 = self.extractingFeatures_ORB(img_0)
			kp_1,des_1,feats_1 = self.extractingFeatures_ORB(img_1)

			feats_ordered_0, feats_ordered_1, good = self.featureMatching(des_0, des_1, feats_0, feats_1, featureMatchingSensitivity=0.35)

			matches_orb=cv2.drawMatchesKnn(img_0,kp_0,img_1,kp_1,good,None,flags=2)
			cv2.imwrite("ORB_matching"+str(i)+".png",matches_orb)

			#R, t = self.affineTransform(feats_ordered_0, feats_ordered_1)
			R, t = self.filterAffineTransform(feats_ordered_0, feats_ordered_1)

			global_x_est, global_y_est=self.local2global(t)

			self.update(R, global_x_est, global_y_est)

			
			print "GLOBAL ANGLE = ",self.rad2deg(self.angle)
			print "GLOBAL POS X = ",self.pos_x
			print "GLOBAL POS Y = ",self.pos_y
			print "########################################"

			estimated_pos_x_tab.append(self.pos_x)
			estimated_pos_y_tab.append(self.pos_y)
			estimated_orientation_tab.append(self.angle)

		self.plot_visualization(global_x_tab,global_y_tab,global_orientation_tab, estimated_pos_x_tab, estimated_pos_y_tab, estimated_orientation_tab)

	def pixel2meters(self, len_sensor_x, len_sensor_y, focal_length, scale_variation):
		#TO DO !!!!
		D_D=scale_variation
		tan_alpha=D_X/(2*D_D)

		return meters_x, meters_y

	def run(self, camera_name):
		estimated_pos_x_tab=[]
		estimated_pos_y_tab=[]
		estimated_orientation_tab=[]

		cap=cv2.VideoCapture(camera_name)
		ret, img_0=cap.read()
		while(cap.isOpened()):
			ret, img_1=cap.read()
			if ret:
				kp_0,des_0,feats_0 = self.extractingFeatures_ORB(img_0)
				kp_1,des_1,feats_1 = self.extractingFeatures_ORB(img_1)

				feats_ordered_0, feats_ordered_1, good = self.featureMatching(des_0, des_1, feats_0, feats_1, featureMatchingSensitivity=0.3)

				matches_orb=cv2.drawMatchesKnn(img_0,kp_0,img_1,kp_1,good,None,flags=2)
				cv2.imwrite("ORB_matching"+str(i)+".png",matches_orb)

				#R, t = self.affineTransform(feats_ordered_0, feats_ordered_1)
				R, t = self.filterAffineTransform(feats_ordered_0, feats_ordered_1)

				global_x_est, global_y_est=self.local2global(t)

				self.update(R, global_x_est, global_y_est)

				
				print "GLOBAL ANGLE = ",self.rad2deg(self.angle)
				print "GLOBAL POS X = ",self.pos_x
				print "GLOBAL POS Y = ",self.pos_y
				print "########################################"

				estimated_pos_x_tab.append(self.pos_x)
				estimated_pos_y_tab.append(self.pos_y)
				estimated_orientation_tab.append(self.angle)

				#Reattribution of img_0 <- img_1
				img_0=img_1
			else:
				print "End of file OR error in reading video's frame"

	def plot_visualization(self, pos_x_tab,pos_y_tab,orientation_tab,estimated_pos_x_tab,estimated_pos_y_tab,estimated_orientation_tab):
		img=cv2.imread("Total5.jpg",1)
		fontFace=cv2.FONT_HERSHEY_SIMPLEX
		fontScale=1.5

		color_real=(0,255,0)
		color_estimation=(0,0,255)
			
		### PLOT REAL DATA
		cv2.circle(img,(int(np.rint(pos_x_tab[0])),int(np.rint(pos_y_tab[0]))),30,color_real,5)
		text_pos=tuple(map(operator.add,(int(np.rint(pos_x_tab[0])),int(np.rint(pos_y_tab[0]))),(40,-10)))
		cv2.putText(img, str(0), text_pos, fontFace, fontScale, color_real,thickness=3)
		orientation_line=tuple(map(operator.add,(pos_x_tab[0],pos_y_tab[0]),(30*np.cos(orientation_tab[0]),30*np.sin(orientation_tab[0]))))
		cv2.line(img,(int(np.rint(pos_x_tab[0])),int(np.rint(pos_y_tab[0]))),(int(np.rint(orientation_line[0])),int(np.rint(orientation_line[1]))),color_real, 6)
		for i in range(np.shape(pos_x_tab)[0]-1):
			cv2.circle(img,(int(np.rint(pos_x_tab[i+1])),int(np.rint(pos_y_tab[i+1]))),30,color_real,5)
			text_pos=tuple(map(operator.add,(int(np.rint(pos_x_tab[i+1])),int(np.rint(pos_y_tab[i+1]))),(40,-10)))
			cv2.putText(img, str(i+1), text_pos, fontFace, fontScale, color_real,thickness=3)
			#cv2.line(img, (int(pos_x_tab[i]),int(pos_y_tab[i])), (int(pos_x_tab[i+1]),int(pos_y_tab[i+1])), color_real, 3) 
			orientation_line=tuple(map(operator.add,(pos_x_tab[i+1],pos_y_tab[i+1]),(30*np.cos(orientation_tab[i+1]),30*np.sin(orientation_tab[i+1]))))

			cv2.line(img,(int(np.rint(pos_x_tab[i+1])),int(np.rint(pos_y_tab[i+1]))),(int(np.rint(orientation_line[0])),int(np.rint(orientation_line[1]))),color_real, 6)

		### PLOT ESTIMATED DATA
		cv2.circle(img,(int(np.rint(estimated_pos_x_tab[0])),int(np.rint(estimated_pos_y_tab[0]))),30,color_estimation,5)
		text_pos=tuple(map(operator.add,(int(np.rint(estimated_pos_x_tab[0])),int(np.rint(estimated_pos_y_tab[0]))),(40,40)))
		cv2.putText(img, str(1), text_pos, fontFace, fontScale, color_estimation,thickness=3)
		orientation_line=tuple(map(operator.add,(estimated_pos_x_tab[0],estimated_pos_y_tab[0]),(30*np.cos(estimated_orientation_tab[0]),30*np.sin(estimated_orientation_tab[0]))))
		cv2.line(img,(int(np.rint(estimated_pos_x_tab[0])),int(np.rint(estimated_pos_y_tab[0]))),(int(np.rint(orientation_line[0])),int(np.rint(orientation_line[1]))),color_estimation, 3)
		for i in range(np.shape(estimated_pos_x_tab)[0]-1):
			cv2.circle(img,(int(np.rint(estimated_pos_x_tab[i+1])),int(np.rint(estimated_pos_y_tab[i+1]))),30,color_estimation,5)
			text_pos=tuple(map(operator.add,(int(np.rint(estimated_pos_x_tab[i+1])),int(np.rint(estimated_pos_y_tab[i+1]))),(40,40)))
			cv2.putText(img, str(i+2), text_pos, fontFace, fontScale, color_estimation,thickness=3)
			#cv2.line(img, (int(estimated_pos_x_tab[i]),int(estimated_pos_y_tab[i])), (int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])), color_estimation, 3) 
			orientation_line=tuple(map(operator.add,(estimated_pos_x_tab[i+1],estimated_pos_y_tab[i+1]),(30*np.cos(estimated_orientation_tab[i+1]),30*np.sin(estimated_orientation_tab[i+1]))))
			cv2.line(img,(int(np.rint(estimated_pos_x_tab[i+1])),int(np.rint(estimated_pos_y_tab[i+1]))),(int(np.rint(orientation_line[0])),int(np.rint(orientation_line[1]))),color_estimation, 3)

		cv2.imwrite("result5.png",img)
		cv2.imshow("img",img)
		cv2.waitKey(0)

	###################
	# TESTING FUNCTION
	###################

	# def test_feature_extractor_descriptor(self):


	# 	return

	# def test_feature_matching(self):


	# 	return

	def test_affine_transform(self, which_tag="regular"):
		#test regular, weighted, openCV one

		if which_tag=="regular":
			A=np.array([(10,10),(15,11),(13,13)],np.float32)
			B=self.applyRotationTo(A,0,0,0,0,10) #Apply rotation of 10
			B=B+np.array([(10,10),(10,10),(10,10)],np.float32) #Apply translation of x=10 and y=10

			R,t=self.affineTransform(self,A,B)

			print "==================================================="
			print "Result should be: angle=10, x=10, y=10"
			print "angle = ", self.rad2deg(np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64))
			print "x = ", t[0]
			print "y = ". t[1]
			print "==================================================="

		elif which_tag=="weighted":
			A=np.array([(10,10),(15,11),(13,13)],np.float32)
			B=self.applyRotationTo(A,0,0,0,0,10) #Apply rotation of 10
			B=B+np.array([(10,10),(10,10),(10,10)],np.float32) #Apply translation of x=10 and y=10
			distance=np.array([[10],[4],[1]],np.float32) #distance computed when the feature matching is done. Correspond of the "error" between two descriptor.

			R,t=self.weightedAffineTransform(A, B, distance, norm_type="L2", method="inverse")

			print "==================================================="
			print "Result should be: angle=10, x=10, y=10"
			print "angle = ", self.rad2deg(np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64))
			print "x = ", t[0]
			print "y = ". t[1]
			print "==================================================="

		elif which_tag=="with_scale":
			A=np.array([(10,10),(15,11),(13,13)],np.float32)
			B=A*1.2 #Apply scaling of +20%
			B=self.applyRotationTo(A,0,0,0,0,10) #Apply rotation of 10
			B=B+np.array([(10,10),(10,10),(10,10)],np.float32) #Apply translation of x=10 and y=10

			R,t,s=self.affineTransform3d(A, B)

			print "Result should be: angle=10, x=10, y=10, scaling=1.2"
			print "==================================================="
			print "angle = ", self.rad2deg(np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64))
			print "x = ", t[0]
			print "y = ", t[1]
			print "scaling = ",s[0][0]
			print "==================================================="

		return





# INIT VARIABLES FOR "OpticalFlow" CLASS
frame_x_size=1200.0
frame_y_size=1200.0
margin_ROI=0.0
global_orientation=0.0
global_x=4000.0
global_y=2400.0
scale=1.0
global_x_tab=np.array([4000,3900,3800,3200,2800,3000],np.float64)
global_y_tab=np.array([2400,2200,2000,1800,1600,1400],np.float64)
global_orientation_tab=np.array([0,40,20,50,90,10],np.float64)
# global_x_tab=np.array([3900,4000],np.float64)
# global_y_tab=np.array([2300,2400],np.float64)
# global_orientation_tab=np.array([0,10],np.float64)

global_orientation_tab*=np.tile(math.pi/180,np.shape(global_orientation_tab)[0])

# Creation object of Class OpticalFlow
optflow=OpticalFlow(margin_ROI, frame_x_size, frame_y_size, global_orientation, global_x, global_y, scale) 

# Run test 
optflow.run_test(global_x_tab, global_y_tab, global_orientation_tab)

# Run test Affine Transform
#optflow.test_affine_transform(which_tag="regular")
#optflow.test_affine_transform(which_tag="weighted")
#optflow.test_affine_transform(which_tag="with_scale")



# A=np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]],np.float64)#,[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],[19,19],[20,20],[21,21],[22,22],[23,23],[24,24],[25,25],[26,26],[27,27],[28,28],[29,29]],np.float64)
# B=np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]],np.float64)#,[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],[19,19],[20,20],[21,21],[22,22],[23,23],[24,24],[25,25],[26,26],[27,27],[28,28],[29,29],[30,30]])

# optflow.filterRigidTransform(A, B)

