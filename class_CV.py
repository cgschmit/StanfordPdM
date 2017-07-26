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
		frame=cv2.imread("Total1.jpg",0)
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

	def filterAffineTransform(self, A, B):
		### Select the training features (to calculate R and t)
		outlier_ratio=0.2 #experimental value
		prob_success=0.99 #probability that one 
		number_sample=3
		number_trials=int(np.rint(np.log(1-prob_success)/np.log(1-np.power((1-outlier_ratio),number_sample)))) #formula RANSAC

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
			#B_estimated=np.dot(np.linalg.inv(R),A_test.T)-np.tile(t.T,((np.shape(A_test)[0]),1)).T
			B_estimated=np.dot(R,A_test.T)-np.tile(t.T,((np.shape(A_test)[0]),1)).T
			B_estimated=B_estimated.T

			### Calculate distance from point B_test to B_estimated, if within a predefined distance, include as inlier,if not outlier. Count inlier.
			distances=np.sqrt(np.power(B_test[:,0]-B_estimated[:,0],2)+np.power(B_test[:,1]-B_estimated[:,1],2))
			total_distance=np.sum(distances)
			print total_distance

			if init:
				best_total_distance=total_distance
				R_best=R
				t_best=t
				init=False
		
			if total_distance<best_total_distance:
				R_best=R
				t_best=t
				best_total_distance=total_distance

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
		print "INCREMENT X = ",global_x_est*np.cos(self.angle,dtype=np.float64)
		print "INCREMENT Y = ",global_y_est*np.cos(self.angle,dtype=np.float64)
		self.pos_x+=global_x_est*np.cos(self.angle,dtype=np.float64)
		self.pos_y+=global_y_est*np.cos(self.angle,dtype=np.float64)
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

			feats_ordered_0, feats_ordered_1, good = self.featureMatching(des_0, des_1, feats_0, feats_1, featureMatchingSensitivity=0.55)

			matches_orb=cv2.drawMatchesKnn(img_0,kp_0,img_1,kp_1,good,None,flags=2)
			cv2.imwrite("ORB_matching"+str(i)+".png",matches_orb)

			matches_orb=cv2.drawMatchesKnn(img_0,kp_0,img_1,kp_1,good,None,flags=2)
			cv2.imwrite("debug/ORB_test"+str(i)+".png",matches_orb)

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

	def plot_visualization(self, pos_x_tab,pos_y_tab,orientation_tab,estimated_pos_x_tab,estimated_pos_y_tab,estimated_orientation_tab):
		img=cv2.imread("Total1.jpg",1)
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

		cv2.imwrite("result1.png",img)
		cv2.imshow("img",img)
		cv2.waitKey(0)

# INIT VARIABLES FOR "OpticalFlow" CLASS
frame_x_size=1200.0
frame_y_size=1200.0
margin_ROI=0.0
global_orientation=0.0
global_x=4000.0
global_y=2400.0
scale=1.0

# Creation object of Class OpticalFlow
optflow=OpticalFlow(margin_ROI, frame_x_size, frame_y_size, global_orientation, global_x, global_y, scale) 

# Test with size = 600
# global_x_tab=np.array([2000,2040,2080,2120,2160,2200],np.float64)
# global_y_tab=np.array([2000,2040,2080,2120,2160,2200],np.float64)
# global_orientation_tab=np.array([0,10,20,30,20,10],np.float64)

# Test start top Right corner
global_x_tab=np.array([4000,3800,3600,3400,3200,3000],np.float64)
global_y_tab=np.array([2400,2200,2000,1800,1600,1400],np.float64)
global_orientation_tab=np.array([0,10,20,30,20,10],np.float64)

# Test with size = 1200
# global_x_tab=np.array([ 500, 700,1000,1300,1500,1700,1800,2000,2000,2500,2700,2800,2900,3100,2900,2500,2000,1800,1500,1200],np.float64)
# global_y_tab=np.array([1000,1000,1000,1500,2000,2200,2200,2100,2000,1800,1500,1200,1200, 800, 500, 500, 600, 700, 800,1000],np.float64)
# global_orientation_tab=np.array([0,5,10,7,14,17,20,25,35,30,25,30,20,25,18,10,5,0,5,7],np.float64)

# Convertion to rad!
global_orientation_tab*=np.tile(3.14/180,np.shape(global_orientation_tab)[0])

# Run Test
optflow.run_test(global_x_tab, global_y_tab, global_orientation_tab)

print "Increment Angle:   5 /   2 /   3 /   5 /  -5"
print "Increment Pos X: 200 / 300 / 300 / 200 / 200"
print "Increment Pos Y:   0 /   0 / 500 / 500 / 200"



# A=np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]],np.float64)#,[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],[19,19],[20,20],[21,21],[22,22],[23,23],[24,24],[25,25],[26,26],[27,27],[28,28],[29,29]],np.float64)
# B=np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]],np.float64)#,[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],[19,19],[20,20],[21,21],[22,22],[23,23],[24,24],[25,25],[26,26],[27,27],[28,28],[29,29],[30,30]])

# optflow.filterRigidTransform(A, B)



