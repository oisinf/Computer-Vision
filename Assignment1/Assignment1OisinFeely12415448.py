import numpy as np
import numpy.matlib 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import spatial


def calibrateCamera3D( d ) :
	#Load data
	data = np.loadtxt(d)

	#In-homogenous points xyz
	IP = data[:, 0:3]
	
	#Matrix of ones 
	ones = np.ones((491, 1))
	
	#Add ones to IP to make homogenous 
	P = np.hstack((IP,ones))
	print np.shape(IP)
	#-xP and -yP
	xpts = data[:, 3]
	ypts = data[:, 4]
	
	#Xpts is horizontal 1 D matrix
	xpts = np.array([xpts])	
	#Xmtx is vertical 1 D matrix 
	xmtx = xpts.transpose()
	#Repmat xmtx so that it is is x1, x1, x1, x1 to xn xn xn xn vertically
	#for multiplication with PT
	xmtx = np.matlib.repmat(xmtx,1, 4)
	
	#Same process for ypts
	ypts = np.array([ypts])	
	ymtx = ypts.transpose()
	ymtx = np.matlib.repmat(ymtx,1, 4)

	#-xPn and -xPy
	PX = P*(-xmtx)
	PY = P*(-ymtx)
	
	#A matrix 
	A = np.zeros(((491*2), 12))
	A[0::2, 8::1] = PX[0::1, 0::1]
	A[1::2, 8::1] = PY[0::1, 0::1]	
	A[0::2, 0:4] = P[0::1, 0::1]
	A[1::2, 4:8] = P[0::1, 0::1]
	
	D,V = np.linalg.eig(A.transpose().dot(A))

	M = V[:,np.argmin(D)]

	return M

def visualiseCameraCalibration3D (d, p):
	
	data = np.loadtxt(d)
	
	xpts = data[:, 3]
	ypts = data[:, 4]
	
	plt.ion() 
    	plt.cla()
   	plt.show()
    	plt.plot(xpts,ypts,'g.')

	IP = data[:, 0:3]
	ones = np.ones((491, 1))	
	P = np.hstack((IP,ones))
	
	p = np.reshape(p, (3, 4))	

	P = P.transpose()
	
	#Multiply 3D homogenous points by calibration matrix
	pts = np.dot(p, P[:, 0::1])
	
	#Plot x/w and y/w
	plt.plot((pts[0:1,]/pts[2::1,]), (pts[1::2,]/pts[2::1,]), 'r.')
	#Keep graph showing after script has ran
	plt.show(block="true")
	
def evaluateCameraCalibration3D (d, p):
		
	data = np.loadtxt(d)
	
	xpts = data[:, 3]
	ypts = data[:, 4]	
	cpoints = np.hstack((xpts, ypts))
	cpoints = np.reshape(cpoints, (2, 491))

	
	IP = data[:, 0:3]
	ones = np.ones((491, 1))	
	P = np.hstack((IP,ones))
	
	p = np.reshape(p, (3, 4))	

	P = P.transpose()

	pts = np.dot(p, P[:, 0::1])
	ptsx = pts[0:1,]/pts[2::1,]
	ptsy = pts[1::2,]/pts[2::1,]
	pts = np.hstack((ptsx, ptsy))
	pts = np.reshape(pts, (2, 491))
	

	mpm = (((np.mean(xpts))+(np.mean(ypts)))/2)
	print "Measured pixels mean: " + str(mpm)
	cpm = (((np.mean(ptsx))+(np.mean(ptsy)))/2)
	print "Computed pixels mean: " + str(cpm)
	
	mpv = (np.var(xpts+ypts))
	print "Measured pixels variance: " + str(mpv)
	cpv = (np.var(ptsx + ptsy))
	print "Computed pixels variance: " + str(cpv)

	minPixelDist = np.min(spatial.distance.cdist(pts[::1,], cpoints[::1,]))
	print 'Minimium Pixel Distance: ' + str(minPixelDist)
	maxPixelDist = np.max(spatial.distance.cdist(pts[::1,], cpoints[::1,]))
	print 'Maxiumium Pixel Distance: ' + str(maxPixelDist)

D = 'data.txt'
P = calibrateCamera3D(D)
evaluateCameraCalibration3D(D, P)
visualiseCameraCalibration3D(D, P)
