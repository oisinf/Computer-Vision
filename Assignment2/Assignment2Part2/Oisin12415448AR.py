import numpy as np
import cv2
import glob
import Image
import ImageDraw
# These two imports are for the signal handler
import signal
import sys

#Globals
#capture video
cap = cv2.VideoCapture(0)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WIDTH = 6
HEIGHT = 9

### Some helper functions #####
def reallyDestroyWindow(windowName) :
    ''' Bug in OpenCV's destroyWindow method, so... '''
    ''' This fix is from http://stackoverflow.com/questions/6116564/ '''
    cv2.destroyWindow(windowName)
    for i in range (1,5):
        cv2.waitKey(1) 

def shutdown():
        ''' Call to shutdown camera and windows '''
        global cap
        cap.release()
        reallyDestroyWindow('img')

def signal_handler(signal, frame):
        ''' Signal handler for handling ctrl-c '''
        shutdown()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
##################

############## calibration of plane to plane 3x3 projection matrix 

def compute_homography(fp,tp):
    ''' Compute homography that takes fp to tp. 
    fp and tp should be (N,3) '''

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # create matrix for linear method, 2 rows for each correspondence pair
    num_corners = fp.shape[0]

    # construct constraint matrix
    A = np.zeros((num_corners*2,9)); 
    A[0::2,0:3] = fp
    A[1::2,3:6] = fp
    A[0::2,6:9] = fp * -np.repeat(np.expand_dims(tp[:,0],axis=1),3,axis=1)
    A[1::2,6:9] = fp * -np.repeat(np.expand_dims(tp[:,1],axis=1),3,axis=1)

    # solve using *naive* eigenvalue approach
    D,V = np.linalg.eig(A.transpose().dot(A))

    H = V[:,np.argmin(D)].reshape((3,3))
    
    # normalise and return
    return H

##############



def planarAR():
	
	#read in image
	pic = cv2.imread("bearCropped.jpg")
	#just to intialize bearpic before use, as white so can be added to img without affecting it
	bearpic = np.zeros([480,640,3],dtype=np.uint8)
	bearpic.fill(255)
	#get shape of image (height, width)
	h, w, n=  np.shape(pic)
	#Intialize mask to image size, mask is white
	mask = np.zeros([h,w,3],dtype=np.uint8)
	mask.fill(255)

	#Get x and y points in the image to match checkboard points
	xpts = np.linspace(0, h, WIDTH)
	ypts = np.linspace(0, w, HEIGHT)
	
	##meshgrid and then stack to get list of xy points that represent samples from image
	y2, x2 = np.meshgrid(ypts, xpts)
	imgpts = np.column_stack((y2.ravel(), x2.ravel()))
	
	#Ones to stack on impgpts to make homogenous
	ones = np.ones((54, 1))	
	imgpts = np.hstack((imgpts, ones))
	
	while(True):
		#capture a frame
        	ret, img = cap.read()
        
        	# Our operations on the frame come here
        	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        	# Find the chess board corners
        	ret, corners = cv2.findChessboardCorners(gray, (HEIGHT,WIDTH),None)

        	# If found, add object points, image points (after refining them)
        	if ret == True:
        	 	cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

		   	#Corner coordinates for each frame just equal corners, shape of corners is 54, 1,2 so squeeze to get rid of 1 d
			corners = np.squeeze(corners) 
			#Stack ones to make homogenous
			corners = np.hstack((corners, ones))
			
			#Compute homography 
			H = compute_homography(imgpts, corners)
			
			#get warp perspective of picture and mask
			bearpic = cv2.warpPerspective(pic, H,(640, 480))
			m = cv2.warpPerspective(mask, H,(640, 480))
			#Invert mask
			m = cv2.bitwise_not(m)
			#Multiply img by mask to get rid of checkerboard center
			img = img*m

			#This inverts image so invert back
			img = cv2.bitwise_not(img)

			#Lot of noise present due to addition and multiplication of image, denoising to get rid of noise
			#this does slow the program down a lot so left uncommented
			#img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
		#Show image with picture overlay
        	cv2.imshow('img',img+bearpic)      
        	if cv2.waitKey(1) & 0xFF == ord('q'):
        	    	break

		# release everything
	shutdown()


planarAR()
