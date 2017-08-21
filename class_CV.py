from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import cv2
import operator
import math
import pandas as pd
import time


class OpticalFlow:

	### INITIALIZATION OF CLASS
	def __init__(self, margin, frame_x, frame_y, angle, pos_x, pos_y, pos_z, scale, algo_name="ORB"):
		self.margin=margin
		self.frame_x=frame_x
		self.frame_y=frame_y
		self.angle=angle
		self.pos_x=pos_x
		self.pos_y=pos_y
		self.pos_z=pos_z
		self.scale=scale
		self.estimated_pos_x_tab=[]
		self.estimated_pos_y_tab=[]
		self.estimated_orientation_tab=[]

		if algo_name=="ORB":
			feature_params_orb = dict ( nfeatures=100,
	                            scaleFactor=2,
	                            nlevels=5,
	                            edgeThreshold=20, 
	                            firstLevel=0, 
	                            WTA_K=4, 
	                            patchSize=60
	                            )
			self.orb = cv2.ORB_create(**feature_params_orb)
		elif algo_name=="SIFT" or algo_name=="ROOTSIFT":
			feature_param_sift = dict( nfeatures = 50,
								nOctaveLayers = 3,
								contrastThreshold = 0.04,
								edgeThreshold = 10,
								sigma = 1.6 
								)
			self.sift = cv2.xfeatures2d.SIFT_create(**feature_param_sift)
		elif algo_name=="SURF":
			feature_param_surf = dict( hessianThreshold = 50,
								nOctaves = 4,
								nOctaveLayers = 3,
								extended = False,
								upright = False 
								)
			self.surf = cv2.xfeatures2d.SURF_create(**feature_param_surf)
		elif algo_name=="BRIEF":
			feature_param_star = dict( maxSize=50,
								responseThreshold=30,
	        					lineThresholdProjected=10,
            					lineThresholdBinarized=8,
            					suppressNonmaxSize=5)
			feature_param_brief = dict( bytes = 32, use_orientation = False)
			self.star = cv2.xfeatures2d.StarDetector_create(**feature_param_star)
			self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		elif algo_name=="ALL":
			feature_params_orb = dict (nfeatures=100,
	                            scaleFactor=2,
	                            nlevels=5,
	                            edgeThreshold=20, 
	                            firstLevel=0, 
	                            WTA_K=4, 
	                            patchSize=60
	                            )
			self.orb = cv2.ORB_create(**feature_params_orb)
			feature_param_sift = dict( nfeatures = 100,
								nOctaveLayers = 3,
								contrastThreshold = 0.04,
								edgeThreshold = 10,
								sigma = 1.6 
								)
			self.sift = cv2.xfeatures2d.SIFT_create(**feature_param_sift)
			feature_param_surf = dict( hessianThreshold = 100,
								nOctaves = 4,
								nOctaveLayers = 3,
								extended = False,
								upright = False 
								)
			self.surf = cv2.xfeatures2d.SURF_create(**feature_param_surf)
			feature_param_star = dict( maxSize=100,
								responseThreshold=30,
	        					lineThresholdProjected=20,
            					lineThresholdBinarized=16,
            					suppressNonmaxSize=5)
			feature_param_brief = dict( bytes = 32, use_orientation = True)
			self.star = cv2.xfeatures2d.StarDetector_create(**feature_param_star)
			self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(**feature_param_brief)

	### CREATING FRAME FOR TESTING ALGORITHMS
	def createFrame(self, x, y, teta):
		frame=cv2.imread("Total2.jpg",0)
		(rows,cols)=np.shape(frame)
		#Rotation of image
		M=cv2.getRotationMatrix2D((x+self.frame_x/2,y+self.frame_y/2), self.rad2deg(teta), self.scale)
		frame_rotated=cv2.warpAffine(frame,M,(cols,rows))
		#Select Frame
		mask_rotated=frame_rotated[int(np.rint(y)):int(np.rint(y+self.frame_y)),int(np.rint(x)):int(np.rint(x+self.frame_x))]    
		return mask_rotated

	### FEATURE EXTRACTORS AND DESCRIPTORS
	def extractingFeatures_ORB(self, img):
		kp, des = self.orb.detectAndCompute(img,mask=None)
		feats = self.keypoints2feature(kp)
		return kp, des, feats
	def extractingFeatures_SIFT(self, img):
		kp, des = self.sift.detectAndCompute(img, mask=None)
		feats = self.keypoints2feature(kp)
		return kp, des, feats
	def extractingFeatures_SURF(self, img):
		kp, des = self.surf.detectAndCompute(img, mask=None)
		feats = self.keypoints2feature(kp)
		return kp, des, feats
	def extractingFeatures_BRIEF(self, img):
		kp = self.star.detect(img,None)
		kp, des = self.brief.compute(img, kp)
		feats = self.keypoints2feature(kp)
		return kp, des, feats
	def extractingFeatures_OPP_SIFT(self, img):
		#NOT YET WORKING. APPARENTLY NO FUNCTION OPPONENT IN OpenCV3
		kp, des = self.orb.detectAndCompute(img,mask=None)
		feats = self.keypoints2feature(kp)
		return kp, des, feats
	def extractingFeatures_ROOT_SIFT(self, img):
		eps=1e-7
		kp, des = self.sift.detectAndCompute(img, mask=None)
		if len(kp) == 0:
			return ([], None)
		des /= (des.sum(axis=1, keepdims=True) + eps)
		des = np.sqrt(des)

		feats = self.keypoints2feature(kp)
		return kp, des, feats

	### FEATURE MATCHING
	def featureMatching(self, des_0, des_1, feats_0, feats_1, featureMatchingSensitivity = 0.75):
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

	### AFFINE TRANSFORM
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

	### REFERENTIAL
	def local2global(self, t, pos_frame_global_x=0, pos_frame_global_y=0):
		global_x_est=pos_frame_global_x+np.sqrt(np.power(t[0],2,dtype=np.float64)+np.power(t[1],2,dtype=np.float64),dtype=np.float64)*np.sin(np.arctan2(t[0],t[1])-self.angle,dtype=np.float64)
		global_y_est=pos_frame_global_y+np.sqrt(np.power(t[0],2,dtype=np.float64)+np.power(t[1],2,dtype=np.float64),dtype=np.float64)*np.cos(np.arctan2(t[0],t[1])-self.angle,dtype=np.float64)
		return global_x_est, global_y_est
	
	### UPDATE POSITION ROBOT
	def update(self, R, global_x_est, global_y_est):
		local_angle=np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64)
		# print "########################################"
		# print "INCREMENT ANGLE = ",self.rad2deg(local_angle)
		# print "INCREMENT X = ",global_x_est
		# print "INCREMENT Y = ",global_y_est
		self.pos_x+=global_x_est
		self.pos_y+=global_y_est
		self.angle+=local_angle
		return

	### RUN TESTS / RUN CLASS
	def run_test(self,global_x_tab,global_y_tab,global_orientation_tab, return_perf=False):
		estimated_pos_x_tab=[]
		estimated_pos_y_tab=[]
		estimated_orientation_tab=[]
		timer=[]

		for i in range(np.shape(global_x_tab)[0]-1):
			start=time.time()
			print "ITERATION NUMBER: ",(i+1)
			img_0=self.createFrame(global_x_tab[i],global_y_tab[i],global_orientation_tab[i])
			img_1=self.createFrame(global_x_tab[i+1],global_y_tab[i+1],global_orientation_tab[i+1])
			
			kp_0,des_0,feats_0 = self.extractingFeatures_ROOT_SIFT(img_0)
			kp_1,des_1,feats_1 = self.extractingFeatures_ROOT_SIFT(img_1)

			feats_ordered_0, feats_ordered_1, good = self.featureMatching(des_0, des_1, feats_0, feats_1, featureMatchingSensitivity=0.75)

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
			stop=time.time()
			timer.append(stop-start)
		self.plot_visualization(global_x_tab,global_y_tab,global_orientation_tab, estimated_pos_x_tab, estimated_pos_y_tab, estimated_orientation_tab)
		
		if return_perf==True:
			error_x=global_x_tab[1:]-estimated_pos_x_tab
			error_y=global_y_tab[1:]-estimated_pos_y_tab
			error_orientation=global_orientation_tab[1:]-estimated_orientation_tab
			error_total_x=0
			error_total_y=0
			error_total_orientation=0
			cummulative_error_x=np.zeros(np.shape(error_x)[0])
			cummulative_error_y=np.zeros(np.shape(error_y)[0])
			cummulative_error_orientation=np.zeros(np.shape(error_orientation)[0])
			displacement_x=np.zeros(np.shape(error_orientation)[0])
			displacement_y=np.zeros(np.shape(error_orientation)[0])
			displacement_angle=np.zeros(np.shape(error_orientation)[0])
			for i in range(np.shape(error_x)[0]):
				error_total_x+=error_x[i]
				error_total_y+=error_y[i]
				error_total_orientation+=error_orientation[i]
				displacement_x[i]=global_x_tab[i+1]-global_x_tab[i]
				displacement_y[i]=global_y_tab[i+1]-global_y_tab[i]
				displacement_angle[i]=global_orientation_tab[i+1]-global_orientation_tab[i]
			dist_err=np.sqrt(np.power(error_x,2)+np.power(error_y,2))
			plt.plot(dist_err,'y--')
			plt.xlabel('Frame Number')
			plt.ylabel('Distance Error (pixel)')
			plt.savefig("thesis/general_perf/ROOT_SIFT/cummulative_error_dist.png")
			plt.close()
			plt.plot(error_x,'b')
			plt.plot(error_y,'r')
			plt.xlabel('Frame Number')
			plt.ylabel('Cummulative Error (pixel)')
			plt.savefig("thesis/general_perf/ROOT_SIFT/cummulative_error_xy.png")
			plt.close()
			plt.plot(error_orientation,'g')
			plt.xlabel('Frame Number')
			plt.ylabel('Cummulative Error (radian)')
			plt.savefig("thesis/general_perf/ROOT_SIFT/cummulative_error_angle.png")
			plt.close()
			plt.plot(timer,'g')
			plt.xlabel('Frame Number')
			plt.ylabel('Time of processing (second)')
			plt.savefig("thesis/general_perf/ROOT_SIFT/processing-time.png")
			plt.close()
			#In Percentage of Displacement! /Divide by i+1
			plt.plot(dist_err/np.sqrt(np.power(displacement_x,2)+np.power(displacement_y ,2)),'y--')
			plt.xlabel('Frame Number')
			plt.ylabel('Distance Error (%displacement)')
			plt.savefig("thesis/general_perf/ROOT_SIFT/cummulative_error_dist_percentage.png")
			plt.close()


	def run(self, camera_name):
		estimated_pos_x_tab=[]
		estimated_pos_y_tab=[]
		estimated_orientation_tab=[]
		iterator=-1
		t_tab=[]
		R_tab=[]
		cap=cv2.VideoCapture(camera_name)
		ret, img_0=cap.read()
		while(cap.isOpened()):
			iterator+=1
			ret, img_1=cap.read()
			if ret:
				kp_0,des_0,feats_0 = self.extractingFeatures_ORB(img_0)
				kp_1,des_1,feats_1 = self.extractingFeatures_ORB(img_1)

				feats_ordered_0, feats_ordered_1, good = self.featureMatching(des_0, des_1, feats_0, feats_1, featureMatchingSensitivity=0.3)

				#R, t = self.affineTransform(feats_ordered_0, feats_ordered_1)
				if np.shape(feats_ordered_0)[0]>3:
					R, t = self.filterAffineTransform(feats_ordered_0, feats_ordered_1)
					t_tab.append(t)
					R_tab.append(R)
				elif np.shape(feats_ordered_0)[0]<=3 and iterator>2:
					# t=0.7*t_tab[iterator-1]+0.3*t_tab[iterator-2]
					# R=0.7*R_tab[iterator-1]+0.3*R_tab[iterator-2]
					t=np.array([0,0])
					R=np.array([[1,0],[0,1]])
					# Don't want to push predicted of predicted value in the table for prediction!
					# t_tab.append(t)
					# R_tab.append(R)
				else:
					t=np.array([0,0])
					R=np.array([[1,0],[0,1]])
					# Don't want to push predicted of predicted value in the table for prediction!
					# t_tab.append(t)
					# R_tab.append(R)

				global_x_est, global_y_est=self.local2global(t)

				self.update(R, global_x_est, global_y_est)
				
				# print "GLOBAL ANGLE = ",self.rad2deg(self.angle)
				# print "GLOBAL POS X = ",self.pos_x
				# print "GLOBAL POS Y = ",self.pos_y
				# print "########################################"

				estimated_pos_x_tab.append(self.pos_x)
				estimated_pos_y_tab.append(self.pos_y)
				estimated_orientation_tab.append(self.angle)

				#Reattribution of img_0 <- img_1
				img_0=img_1
			else:
				print "End of file OR error in reading video's frame. BLABLA"
				break

		npz_calibration = np.load("video_data/calib_v_1.MOV.npz") #has been found with the video calibration_v.MOV and the file calibration.py
		camera_matrix = npz_calibration["camera_matrix"]
		dist_coefs = npz_calibration["dist_coefs"]

		h,  w = img_0.shape[:2]
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

		outpu_t=self.pixel2meters(estimated_pos_x_tab,estimated_pos_y_tab,newcameramtx)
		# plt.plot(meters_x)
		# plt.plot(meters_y)
		#Plot GROUNDTRUTH
		Excel=pd.read_excel("video_data_2/2017-8-11_test2_v.xlsx")
		#Excel['X'].head()
		X_Data=np.array([])
		Y_Data=np.array([])
		Z_Data=np.array([])
		for i in range(np.size(Excel['t_X_'])-1):
		    if i%4==0:
		        X_Data=np.append(X_Data,Excel['t_X_'][i+1])
		        Y_Data=np.append(Y_Data,Excel['t_Y_'][i+1])
		plt.plot(X_Data,Y_Data,'g')

		plt.plot(outpu_t[:,0],outpu_t[:,1],'g--')
		plt.show()

	### DATA PROCESSING FOR TESTING CAMERA
	def import_curvefit_data(self): # TO WORK ON STILL
		Excel=pd.read_excel("video_data/same_height.xlsx")
		index=np.array([])
		X_Data=np.array([])
		Y_Data=np.array([])
		Z_Data=np.array([])
		j=0
		for i in range(np.size(Excel['X'])):
		    if i%4==0:
		    	j+=1
		    	index=np.append(index,j)
		        X_Data=np.append(X_Data,Excel['X'][i])
		        Y_Data=np.append(Y_Data,Excel['Y'][i])
		        Z_Data=np.append(Z_Data,Excel['Z'][i])
		        W_Data=np.append(W_Data,Excel['W'][i])

		plt.plot(X_Data,'r--',label='X True')
		plt.plot(Y_Data,'b--',label='Y True')
		plt.plot(Z_Data,'g--',label='Z True')
		plt.plot(W_Data,'y--',label='W True (Rotation around Z)')
		plt.show()

		deg=200
		coeff=np.polyfit(index, Z_Data, deg)
		p = np.poly1d(coeff)
		plt.plot(index,p(index),'b')
		plt.plot(index,Z_Data,'r')
		plt.show()
		return 

	def pixel2meters(self,dst_px_x, dst_px_y,camera_matrix):
		#value iphone 7 camera
		# focal_length=0.028
		# size_sensor_px_x=320
		# size_sensor_px_y=568
		# size_sensor_m_x=0.0036
		# size_sensor_m_y=0.0048
		outpu_t=np.zeros((np.shape(dst_px_x)[0],3))
		for i in range(np.shape(dst_px_x)[0]):
			inpu_t=np.array([dst_px_x[i],dst_px_y[i],1.08-0.68])
			outpu_t[i]=np.dot(np.linalg.inv(camera_matrix),inpu_t)
			outpu_t[i][0]=outpu_t[i][0]/outpu_t[i][2]
			outpu_t[i][1]=outpu_t[i][1]/outpu_t[i][2]
			outpu_t[i][2]=outpu_t[i][2]/outpu_t[i][2]
		return outpu_t


	### UTILS
	def keypoints2feature(self, kp):
		totalTab=[]
		for i in range(np.shape(kp)[0]):
			tab = [[np.float32(kp[i].pt[0]), np.float32(kp[i].pt[1])]]
			totalTab.append(tab)
		myarray = np.asarray(totalTab)
		return myarray
	def deg2rad(self,deg):
		rad=deg*math.pi/180
		return rad
	def rad2deg(self,rad):
		deg=rad*180/math.pi
		return deg

	### VISUALIZATION
	def plot_visualization(self, pos_x_tab,pos_y_tab,orientation_tab,estimated_pos_x_tab,estimated_pos_y_tab,estimated_orientation_tab):
		img=cv2.imread("Total2.jpg",1)
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

		cv2.imwrite("result2.png",img)
		cv2.imshow("img",img)
		cv2.waitKey(0)

	### TESTING FUNCTION
	def test_feature_extractor_descriptor(self, global_x_tab, global_y_tab, global_orientation_tab, write_img=True):
		estimated_pos_x_tab=[]
		estimated_pos_y_tab=[]
		estimated_orientation_tab=[]
		#extractor_descriptor_tab=['ORB','SIFT','SURF','BRIEF','ROOT_SIFT']

		start=time.time()
		for i in range(np.shape(global_x_tab)[0]):
			img_0=self.createFrame(global_x_tab[i],global_y_tab[i],global_orientation_tab[i])
			kp_0,des_0,feats_0 = self.extractingFeatures_SIFT(img_0)
			if write_img==True:
				for j in range(np.shape(feats_0)[0]):
					cv2.circle(img_0, (feats_0[j][0][0],feats_0[j][0][1]), 10, (255,0,0), 1)
				cv2.imwrite('test_feature_extractor_descriptor/img_SIFT_'+str(i)+'.png',img_0)
		stop=time.time()
		SIFT_timer_average=(stop-start)/(np.shape(global_x_tab)[0])

		start=time.time()
		for i in range(np.shape(global_x_tab)[0]):
			img_0=self.createFrame(global_x_tab[i],global_y_tab[i],global_orientation_tab[i])
			kp_0,des_0,feats_0 = self.extractingFeatures_SURF(img_0)
			if write_img==True:
				for j in range(np.shape(feats_0)[0]):
					cv2.circle(img_0, (feats_0[j][0][0],feats_0[j][0][1]), 10, (255,0,0), 1)
				cv2.imwrite('test_feature_extractor_descriptor/img_SURF_'+str(i)+'.png',img_0)
		stop=time.time()
		SURF_timer_average=(stop-start)/(np.shape(global_x_tab)[0])

		start=time.time()
		for i in range(np.shape(global_x_tab)[0]):
			img_0=self.createFrame(global_x_tab[i],global_y_tab[i],global_orientation_tab[i])
			kp_0,des_0,feats_0 = self.extractingFeatures_BRIEF(img_0)
			if write_img==True:
				for j in range(np.shape(feats_0)[0]):
					cv2.circle(img_0, (feats_0[j][0][0],feats_0[j][0][1]), 10, (255,0,0), 1)
				cv2.imwrite('test_feature_extractor_descriptor/img_BRIEF_'+str(i)+'.png',img_0)
		stop=time.time()
		BRIEF_timer_average=(stop-start)/(np.shape(global_x_tab)[0])

		start=time.time()
		for i in range(np.shape(global_x_tab)[0]):
			img_0=self.createFrame(global_x_tab[i],global_y_tab[i],global_orientation_tab[i])
			kp_0,des_0,feats_0 = self.extractingFeatures_ROOT_SIFT(img_0)
			if write_img==True:
				for j in range(np.shape(feats_0)[0]):
					cv2.circle(img_0, (feats_0[j][0][0],feats_0[j][0][1]), 10, (255,0,0), 1)
				cv2.imwrite('test_feature_extractor_descriptor/img_ROOT_SIFT_'+str(i)+'.png',img_0)
		stop=time.time()
		ROOT_SIFT_timer_average=(stop-start)/(np.shape(global_x_tab)[0])

		start=time.time()
		for i in range(np.shape(global_x_tab)[0]):
			img_0=self.createFrame(global_x_tab[i],global_y_tab[i],global_orientation_tab[i])
			kp_0,des_0,feats_0 = self.extractingFeatures_ORB(img_0)
			#DRAW ON IMAGE FEATURE DETECTED
			if write_img==True:
				for j in range(np.shape(feats_0)[0]):
					cv2.circle(img_0, (feats_0[j][0][0],feats_0[j][0][1]), 10, (255,0,0), 1)
				cv2.imwrite('test_feature_extractor_descriptor/img_ORB_'+str(i)+'.png',img_0)
		stop=time.time()
		ORB_timer_average=(stop-start)/(np.shape(global_x_tab)[0])

		
		timers=[ORB_timer_average,SIFT_timer_average,SURF_timer_average,BRIEF_timer_average,ROOT_SIFT_timer_average]
		return timers

	# def test_feature_matching(self):



	# 	return 

	# def test_affine_transform(self, which_tag="all"):
	# 	#test regular, weighted, openCV one
	# 	if which_tag=="regular" or which_tag=="all":
	# 		print "Rigid Transform:"
	# 		print ""
	# 		A=np.array([(10,10),(15,11),(13,13)],np.float32)
	# 		B=self.applyRotationTo(A,0,0,0,0,10) #Apply rotation of 10
	# 		B=B+np.array([(10,10),(10,10),(10,10)],np.float32) #Apply translation of x=10 and y=10

	# 		R,t=self.affineTransform(self,A,B)

	# 		print "==================================================="
	# 		print "Result should be: angle=10, x=10, y=10"
	# 		print "angle = ", self.rad2deg(np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64))
	# 		print "x = ", t[0]
	# 		print "y = ". t[1]
	# 		print "==================================================="
	# 	if which_tag=="weighted" or which_tag=="all":
	# 		print "Weighted Rigid Transform:"
	# 		print ""
	# 		A=np.array([(10,10),(15,11),(13,13)],np.float32)
	# 		B=self.applyRotationTo(A,0,0,0,0,10) #Apply rotation of 10
	# 		B=B+np.array([(10,10),(10,10),(10,10)],np.float32) #Apply translation of x=10 and y=10
	# 		distance=np.array([[10],[4],[1]],np.float32) #distance computed when the feature matching is done. Correspond of the "error" between two descriptor.

	# 		R,t=self.weightedAffineTransform(A, B, distance, norm_type="L2", method="inverse")

	# 		print "==================================================="
	# 		print "Result should be: angle=10, x=10, y=10"
	# 		print "angle = ", self.rad2deg(np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64))
	# 		print "x = ", t[0]
	# 		print "y = ". t[1]
	# 		print "==================================================="
	# 	if which_tag=="with_scale" or which_tag=="all":
	# 		print "Affine Transform:"
	# 		print ""
	# 		A=np.array([(10,10),(15,11),(13,13)],np.float32)
	# 		B=A*1.2 #Apply scaling of +20%
	# 		B=self.applyRotationTo(A,0,0,0,0,10) #Apply rotation of 10
	# 		B=B+np.array([(10,10),(10,10),(10,10)],np.float32) #Apply translation of x=10 and y=10

	# 		R,t,s=self.affineTransform3d(A, B)

	# 		print "Result should be: angle=10, x=10, y=10, scaling=1.2"
	# 		print "==================================================="
	# 		print "angle = ", self.rad2deg(np.arctan2(R.item([1][0]),R.item([0][0]),dtype=np.float64))
	# 		print "x = ", t[0]
	# 		print "y = ", t[1]
	# 		print "scaling = ",s[0][0]
	# 		print "==================================================="
	# 	return





# INIT VARIABLES FOR "OpticalFlow" CLASS
frame_x_size=1200.0
frame_y_size=800.0
# frame_x_size=320
# frame_y_size=568
margin_ROI=0.0
global_orientation=0 #-0.54
# global_x=4000.0
# global_y=2400.0
global_x=4000
global_y=2400
global_z=(1.10-0.68) #in meters!
scale=1.0
# global_x_tab=np.array([4000,3900,3800,3200,2800,3000],np.float64)
# global_y_tab=np.array([2400,2200,2000,1800,1600,1400],np.float64)
# global_orientation_tab=np.array([0,40,20,50,90,10],np.float64)
# global_x_tab=np.array([4000,3900,3800,3200,2800,3000,3250,3500,3800,4000,3900,3800,3200,2800,3000,3250,3500,3800,4000,3900,3800,3200,2800,3000,3250,3500,3800,4000,3900,3800,3200,2800,3000,3250,3500,3800],np.float64)
# global_y_tab=np.array([2400,2200,2000,1800,1600,1400,1700,2000,2100,2400,2200,2000,1800,1600,1400,1700,2000,2100,2400,2200,2000,1800,1600,1400,1700,2000,2100,2400,2200,2000,1800,1600,1400,1700,2000,2100],np.float64)
# global_orientation_tab=np.array([0,40,20,50,90,10,10,15,20,0,40,20,50,90,10,10,15,20,0,40,20,50,90,10,10,15,20,0,40,20,50,90,10,10,15,20],np.float64)

global_x_tab=np.array(          [4000,3900,3800,3200,2800,3000,3250,3500,3750,4000,3800,4000,3700,3500,3200,3100,2800,2500,2200,2000,1700,1500,1200,1000, 800],np.float64)
global_y_tab=np.array(          [2400,2200,2000,1800,1600,1400,1200,1200,1000, 800,1100, 900,1100,1200,1000,1100,1200,1300,1200,1100,1300,1300,1100,1300,1400],np.float64)
global_orientation_tab=np.array([   0,  10,  20,  30,  20,  15,  20,  20,  10,  20,  10,  20,  20,  20,  10,  12,  15,  18,  20,  25,  30,  25,  22,  24,  25],np.float64)
#global_orientation_tab=np.array([   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],np.float64)
#									0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24


global_orientation_tab*=np.tile(math.pi/180,np.shape(global_orientation_tab)[0])

# Creation object of Class OpticalFlow
optflow=OpticalFlow(margin_ROI, frame_x_size, frame_y_size, global_orientation, global_x, global_y, global_z, scale, algo_name="ALL") 

# Run test on consecutive predefined frame
optflow.run_test(global_x_tab, global_y_tab, global_orientation_tab, return_perf=True)
#timers=optflow.test_feature_extractor_descriptor(global_x_tab, global_y_tab, global_orientation_tab, write_img=False)
#print timers
# Run test on vieo
#optflow.run("video_data_2/video_2_12sec.MOV")

# Test import_curvefit__data(self)
#optflow.import_curvefit_data()

