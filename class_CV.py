from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import cv2
import operator

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

		feature_params_orb = dict ( nfeatures=70,
                            scaleFactor=2,
                            nlevels=5,
                            edgeThreshold=40, 
                            firstLevel=0, 
                            WTA_K=4, 
                            patchSize=60
                            )

		self.orb = cv2.ORB_create(**feature_params_orb)

	def createFrame(self, x, y, teta):
		frame=cv2.imread("Total3.jpg",0)
		(rows,cols)=np.shape(frame)
		#Rotation of image
		M=cv2.getRotationMatrix2D((x+self.frame_x/2,y+self.frame_y/2), self.rad2deg(teta), self.scale)
		frame_rotated=cv2.warpAffine(frame,M,(cols,rows))
		#Select Frame
		mask_rotated=frame_rotated[y:y+self.frame_y,x:x+self.frame_x]    
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

	def affineTransform(self, A, B):
		assert len(A) == len(B)
		N = A.shape[0]
		centroid_A = np.mean(A, axis=0)
		centroid_B = np.mean(B, axis=0)
		AA = A - np.tile(centroid_A, (N, 1))
		BB = B - np.tile(centroid_B, (N, 1))
		H = np.dot(np.transpose(AA),BB)
		U, S, Vt = np.linalg.svd(H)
		R = np.dot(Vt.T,U.T)
		recenter=centroid_B-[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
		recenter=np.dot(np.linalg.inv(R),recenter)+[(1-2*self.margin)/2*self.frame_x,(1-2*self.margin)/2*self.frame_y]
		t=-recenter+centroid_A.T
		return R, t

	def local2global(self, t, pos_frame_global_x=0, pos_frame_global_y=0):
		global_x_est=pos_frame_global_x+np.sqrt(np.power(t[0],2)+np.power(t[1],2))*np.sin(np.arctan2(t[0],t[1])-self.angle)
		global_y_est=pos_frame_global_y+np.sqrt(np.power(t[0],2)+np.power(t[1],2))*np.cos(np.arctan2(t[0],t[1])-self.angle)
		return global_x_est, global_y_est

	def deg2rad(self,deg):
		rad=deg*0.0174
		return rad

	def rad2deg(self,rad):
		deg=rad*57.32
		return deg

	def update(self, R, global_x_est, global_y_est):
		local_angle=np.arctan2(R.item([1][0]),R.item([0][0]))
		self.pos_x+=global_x_est*np.cos(self.angle)
		self.pos_y+=global_y_est*np.cos(self.angle)
		self.angle+=local_angle
		return

	def run_test(self,global_x_tab,global_y_tab,global_orientation_tab):
		estimated_pos_x_tab=[]
		estimated_pos_y_tab=[]
		estimated_orientation_tab=[]
		for i in range(np.shape(global_x_tab)[0]-1):

			img_0=self.createFrame(global_x_tab[i],global_y_tab[i],global_orientation_tab[i])
			img_1=self.createFrame(global_x_tab[i+1],global_y_tab[i+1],global_orientation_tab[i+1])

			kp_0,des_0,feats_0 = self.extractingFeatures_ORB(img_0)
			kp_1,des_1,feats_1 = self.extractingFeatures_ORB(img_1)

			feats_ordered_0, feats_ordered_1, good = self.featureMatching(des_0, des_1, feats_0, feats_1, featureMatchingSensitivity=0.55)

			matches_orb=cv2.drawMatchesKnn(img_0,kp_0,img_1,kp_1,good,None,flags=2)
			cv2.imwrite("debug/ORB_test"+str(i)+".png",matches_orb)

			R, t = self.affineTransform(feats_ordered_0, feats_ordered_1)

			global_x_est, global_y_est=self.local2global(t)

			self.update(R, global_x_est, global_y_est)

			print "########################################"
			print "global_angle = ",self.rad2deg(self.angle)
			print "global_pos_x_robot = ",self.pos_x
			print "global_pos_y_robot = ",self.pos_y
			print "########################################"

			estimated_pos_x_tab.append(self.pos_x)
			estimated_pos_y_tab.append(self.pos_y)
			estimated_orientation_tab.append(self.angle)

		self.plot_visualization(global_x_tab,global_y_tab,global_orientation_tab, estimated_pos_x_tab, estimated_pos_y_tab, estimated_orientation_tab)

	def plot_visualization(self, pos_x_tab,pos_y_tab,orientation_tab,estimated_pos_x_tab,estimated_pos_y_tab,estimated_orientation_tab):
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
		cv2.circle(img,(int(estimated_pos_x_tab[0]),int(estimated_pos_y_tab[0])),30,color_estimation,5)
		text_pos=tuple(map(operator.add,(int(estimated_pos_x_tab[0]),int(estimated_pos_y_tab[0])),(40,40)))
		cv2.putText(img, str(0), text_pos, fontFace, fontScale, color_estimation,thickness=3)
		orientation_line=tuple(map(operator.add,(estimated_pos_x_tab[0],estimated_pos_y_tab[0]),(30*np.cos(estimated_orientation_tab[0]),30*np.sin(estimated_orientation_tab[0]))))
		cv2.line(img,(int(estimated_pos_x_tab[0]),int(estimated_pos_y_tab[0])),(int(orientation_line[0]),int(orientation_line[1])),color_estimation, 3)
		for i in range(np.shape(estimated_pos_x_tab)[0]-1):
			cv2.circle(img,(int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])),30,color_estimation,5)
			text_pos=tuple(map(operator.add,(int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])),(40,40)))
			cv2.putText(img, str(i+1), text_pos, fontFace, fontScale, color_estimation,thickness=3)
			#cv2.line(img, (int(estimated_pos_x_tab[i]),int(estimated_pos_y_tab[i])), (int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])), color_estimation, 3) 
			orientation_line=tuple(map(operator.add,(estimated_pos_x_tab[i+1],estimated_pos_y_tab[i+1]),(30*np.cos(estimated_orientation_tab[i+1]),30*np.sin(estimated_orientation_tab[i+1]))))
			cv2.line(img,(int(estimated_pos_x_tab[i+1]),int(estimated_pos_y_tab[i+1])),(int(orientation_line[0]),int(orientation_line[1])),color_estimation, 3)

		cv2.imwrite("result2.png",img)
		cv2.imshow("img",img)
		cv2.waitKey(0)

# INIT VARIABLES FOR "OpticalFlow" CLASS
frame_x_size=600
frame_y_size=600
margin_ROI=0
global_orientation=0
global_x=2000
global_y=2000
scale=1

# Creation object of Class OpticalFlow
optflow=OpticalFlow(margin_ROI, frame_x_size, frame_y_size, global_orientation, global_x, global_y, scale) 

# Data
# pos_x_tab=np.array([100,200,300,400,500,600,700,800,900,1000])
# pos_y_tab=np.array([100,200,300,400,500,600,700,800,900,1000])
# pos_x_tab=np.array([800,900,1000,1100,1200,1300,1400,1500,1600,1700])
# pos_y_tab=np.array([800,800,800,800,800,800,800,800,800,800])
# pos_y_tab=np.array([800,900,1000,1100,1200,1300,1400,1500,1600,1700])
# orientation_tab=np.array([30,20,10,0,10,20,30,20,10,0],np.float64)
# orientation_tab=np.array([0,0,0,0,0,0,0,0,0,0],np.float64)

global_x_tab=np.array([2000,2100,2150,2200])
global_y_tab=np.array([2000,2100,2150,2250])
global_orientation_tab=np.array([0,10,20,30],np.float64)

# Convertion to rad!
global_orientation_tab*=np.tile(3.14/180,np.shape(global_orientation_tab)[0])

# Run Test
optflow.run_test(global_x_tab, global_y_tab, global_orientation_tab)


