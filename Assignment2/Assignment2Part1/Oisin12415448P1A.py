import numpy as np
import cv2
import glob

# These two imports are for the signal handler
import signal
import sys


### Some helper functions #####
def reallyDestroyWindow(windowName) :
    ''' Bug in OpenCV's destroyWindow method, so... '''
    ''' This fix is from http://stackoverflow.com/questions/6116564/ '''
    cv2.destroyWindow(windowName)
    for i in range (1,5):
        cv2.waitKey(1) 
##################


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WIDTH = 6
HEIGHT = 9

def calibrateCamera():
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
	objp[:,:2] = np.mgrid[0:HEIGHT,0:WIDTH].T.reshape(-1,2)
	
	#objpoints 3D, imgpoints2D use to get camera calib mtx 
	objpoints = []
	imgpoints = []
	
	while (True):
		#Test case for picture with camera rather than video 
		img = cv2.imread("testPic.jpg")
	
		# Our operations on the frame come here
	       	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	        # Find the chess board corners	
	        ret, corners = cv2.findChessboardCorners(gray, (HEIGHT,WIDTH),None)


        	# If found, add object points, image points (after refining them)
        	if ret == True:
	
    			#get 3D points
    			objpoints.append(objp)
    			cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    			#get 2d points
    			imgpoints.append(corners)
       		break
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

	h,  w = img.shape[:2]
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	cv2.imwrite('calibresult.png', (cv2.undistort(img, mtx, dist, None, newcameramtx)))

calibrateCamera() 
