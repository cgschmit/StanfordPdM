import numpy as np
import cv2

# ---------------------------
# Create training pattern
# ---------------------------

def createTrainingPattern_simple(height=1600, width=1600, x=400, y=400, thick=30, radius=180,teta=0):

   img_pattern = np.zeros((height,width), np.uint8)*255

   teta_rad=teta*2*3.14/360
   center=(x,y)
   color = 255
   cv2.circle(img_pattern, center, radius, color, thick, cv2.LINE_AA)
   # cv2.circle(img_pattern, center, radius, color, thick)
   img_circle = img_pattern.copy()
   cv2.line(img_pattern, center, (int(center[0]+radius*np.sin(teta_rad)), int(center[1]-radius*np.cos(teta_rad))), color, thick, cv2.LINE_AA)

   #cv2.imshow('img_pattern',img_pattern)
   #cv2.waitKey(0)
   cv2.destroyAllWindows()

   return img_pattern, img_circle



def createTrainingPattern_img(height=1000, width=1000, x=0, y=0, thick=30, radius=100,teta=0,scale=1):

   img_pattern = np.zeros((height,width), np.uint8)*255
   frame=cv2.imread("pechecopynoir.png",0)
   (rows,cols)=np.shape(frame)

   img_pattern[width/2-cols/2+x:width/2+cols/2+x,height/2-rows/2+y:height/2+rows/2+y]=frame

   #cv2.imshow("img_pattern",img_pattern)
   #cv2.waitKey(2500)

   M=cv2.getRotationMatrix2D(((width/2+x),height/2+y), teta, scale)
   frame=cv2.warpAffine(img_pattern,M,(height,width))
   #cv2.imshow("IMG ROTATED",frame)
   #cv2.waitKey(2500)

   cv2.destroyAllWindows()

   return frame

# Beware that the rotation is not working for frame that are not squared. So keep the frame squared.
def createMovingMask(x=0, y=0,teta=0,scale=1, frame_x=300, frame_y=300):
   frame=cv2.imread("Test-FeaturesMatching/Total3.jpg",0)
   (rows,cols)=np.shape(frame)
   M=cv2.getRotationMatrix2D((x+frame_x/2,y+frame_y/2), teta, scale)
   frame=cv2.warpAffine(frame,M,(cols,rows))
   mask=frame[x:x+frame_x,y:y+frame_y]
   return mask
   
   
# Test ROTATION
################
i=0
img=createMovingMask(x=2000, y=2000,teta=0,scale=1, frame_x=600, frame_y=600)
string_name="Test-FeaturesMatching/img-rot/image_"+str(i)+".png"
cv2.imwrite(string_name,img)
i=1
img=createMovingMask(x=2000, y=2000,teta=10,scale=1, frame_x=600, frame_y=600)
string_name="Test-FeaturesMatching/img-rot/image_"+str(i)+".png"
cv2.imwrite(string_name,img)
i=2
img=createMovingMask(x=2000, y=2000,teta=12,scale=1, frame_x=600, frame_y=600)
string_name="Test-FeaturesMatching/img-rot/image_"+str(i)+".png"
cv2.imwrite(string_name,img)
i=3
img=createMovingMask(x=2000, y=2000,teta=27,scale=1, frame_x=600, frame_y=600)
string_name="Test-FeaturesMatching/img-rot/image_"+str(i)+".png"
cv2.imwrite(string_name,img)
i=4
img=createMovingMask(x=2000, y=2000,teta=36,scale=1, frame_x=600, frame_y=600)
string_name="Test-FeaturesMatching/img-rot/image_"+str(i)+".png"
cv2.imwrite(string_name,img)
################



# Test ROTATION + TRANSLATION
################
# i=0
# img=createMovingMask(x=200, y=200,teta=0,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=1
# img=createMovingMask(x=300, y=300,teta=10,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=2
# img=createMovingMask(x=400, y=400,teta=0,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=3
# img=createMovingMask(x=500, y=500,teta=-10,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=4
# img=createMovingMask(x=600, y=600,teta=5,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=5
# img=createMovingMask(x=700, y=700,teta=15,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=6
# img=createMovingMask(x=800, y=800,teta=20,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=7
# img=createMovingMask(x=900, y=900,teta=0,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=8
# img=createMovingMask(x=1000, y=1000,teta=20,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=9
# img=createMovingMask(x=1100, y=1100,teta=10,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=10
# img=createMovingMask(x=1200, y=1200,teta=-10,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=11
# img=createMovingMask(x=1300, y=1300,teta=-5,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=12
# img=createMovingMask(x=1400, y=1400,teta=-2,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=13
# img=createMovingMask(x=1500, y=1500,teta=0,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=14
# img=createMovingMask(x=1600, y=1600,teta=10,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=15
# img=createMovingMask(x=1700, y=1700,teta=20,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=16
# img=createMovingMask(x=1800, y=1800,teta=40,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=17
# img=createMovingMask(x=1900, y=1900,teta=30,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=18
# img=createMovingMask(x=2000, y=2000,teta=10,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=19
# img=createMovingMask(x=2100, y=2100,teta=0,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=20
# img=createMovingMask(x=2200, y=2200,teta=-5,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=21
# img=createMovingMask(x=2300, y=2300,teta=-10,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
# i=22
# img=createMovingMask(x=2400, y=2400,teta=0,scale=1, frame_x=600, frame_y=600)
# string_name="Test-FeaturesMatching/img-2D-rot/image_"+str(i)+".png"
# cv2.imwrite(string_name,img)
################


#Testing translation of moving pattern
################
# for i in range(30):
#    frame=createTrainingPattern_img(height=1600, width=1600, x=200+50*i, y=200+50*i, thick=30, radius=180,teta=0)
#    string_name="IMAGE-TranslationPattern/test1-50x-50y-0angle/image_"+str(i)+".png"
#    cv2.imwrite(string_name,frame)
################

# Testing moving mask to recreate condition of moving camera
################
# angle=[10,20,10,20,30,10,5,10,-5,-10,-20,-5,0,10,5,15,0,-10,-20,-15,
#        10,20,10,20,30,10,5,10,-5,-10,-20,-5,0,10,5,15,0,-10,-20,-15,
#        10,20,10,20,30,10,5,10,-5,-10,-20,-5,0,10,5,15,0,-10,-20,-15,
#        10,20,10,20,30,10,5,10,-5,-10,-20,-5,0,10,5,15,0,-10,-20,-15,
#        10,20,10,20,30,10,5,10,-5,-10,-20,-5,0,10,5,15,0,-10,-20,-15]
# for i in range(100):
#    frame=createMovingMask(x=200+43*i, y=200+30*i,teta=0,scale=1,frame_x=600, frame_y=400)
#    #cv2.imshow('Frame',frame)
#    #cv2.waitKey(0)
#    string_name="IMAGE_MovingFrameOnBiggerPicture/test4-translation/image_"+str(i)+".png"
#    cv2.imwrite(string_name,frame)
################



#SCRIP TO WRITE ALL THE IMAGE IN FOLDER! Testing rotation robustness
################
#for i in range(360):
#   img_pattern=createTrainingPattern_img(height=400, width=400,teta=i)
#   string_name="Image/image_angle"+str(i)+".png"
#   cv2.imwrite(string_name,img_pattern)
################

# SCRIP TO WRITE specified displacement!
################
# i=0
# print "written"
# img_pattern=createTrainingPattern_img(x=-250, y=0,teta=0)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=1
# img_pattern=createTrainingPattern_img(x=-220, y=0,teta=10)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=2
# img_pattern=createTrainingPattern_img(x=-200, y=10,teta=0)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=3
# img_pattern=createTrainingPattern_img(x=-150, y=50,teta=10)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=4
# img_pattern=createTrainingPattern_img(x=-100, y=10,teta=0)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=5
# img_pattern=createTrainingPattern_img(x=-60, y=0,teta=-10)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=6
# img_pattern=createTrainingPattern_img(x=-50, y=-50,teta=-25)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=7
# img_pattern=createTrainingPattern_img(x=-45, y=0,teta=-30)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=8
# img_pattern=createTrainingPattern_img(x=-30, y=-20,teta=-12)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=9
# img_pattern=createTrainingPattern_img(x=0, y=-100,teta=0)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=10
# img_pattern=createTrainingPattern_img(x=40, y=-50,teta=11)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=11
# img_pattern=createTrainingPattern_img(x=50, y=-20,teta=15)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=12
# img_pattern=createTrainingPattern_img(x=100, y=0,teta=0)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=13
# img_pattern=createTrainingPattern_img(x=150, y=50,teta=-10)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# i=14
# img_pattern=createTrainingPattern_img(x=250, y=0,teta=-10)
# string_name="IMAGE_MovementPatternCircle/image_"+str(i)+".png"
# cv2.imwrite(string_name,img_pattern)
# Corresponding Movements
# x=[-250,-220,-200,-150,-100,-60,-50,-45, -30,   0, 40, 50,100,150,250]
# y=[   0,   0,  10,  50,  10,  0,-50,  0, -20,-100,-50,-20,  0, 50,  0]
# ang=[ 0,  10,   0,  10,   0,-10,-25,-30, -12,   0, 11, 15,  0,-10,-10]
################
