######################################
#
#	Topology Optimization
#	
#	V 01 0 
#	14 11 2016
#
#
# Adapted from TopOpt165
#
######################################

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# Libraries for plotting
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  as cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

# Start Main 
def main(nelx,nely,volfrac,penal,rmin): # ft = filter Method

	deb_mode = 0 # Debugging mode 1-on 0-off
	mesh_plot_b = 0 # Plot Mesh
	fem_plot_b = 0 # Plot FEM Results

	print('Start Topology Optimization\n')
	print('nodes x: ' + str(nelx) + '; nodes y: ' + str(nely))
	print('volfrac: ' + str(volfrac) + '; penal: ' + str(penal) + '; rmin: ' + str(rmin) + '\n')

	# Material properties
	# Max and min stiffness
	Emin = 1e-9	# min. Young's modulus - void regions 
	E0 = 1 		# Young's modulus of the material
	# Poisson's ratio
	nu  = 0.3
	Ke = lk() # elemt Stiffness Matrix

	# Degrees of freedom
	ndof = 2*(nely+1)*(nelx+1)
	if deb_mode==1: print('ndof: ' + str(ndof)) #debug

	# Initialize Design Variables
	x = volfrac * np.ones(nely*nelx,dtype=float)
	xold = x.copy()
	xPhys = x.copy()

	# Initialize Sensitivity
	dc = np.zeros(nely*nelx,dtype=float)

	# Matrix to create the Stiffness matrix -  ith row=element nodes
	edofMat = np.zeros((nely*nelx,8),dtype=int)

	for ely in range(nely):
		for elx in range(nelx):
			el = elx + nelx * (ely)
			P1 = 2*el + ely*2 
			P2 = 2*el + ely*2 +1
			P3 = 2*el + ely*2 +2
			P4 = 2*el + ely*2 +3
			P5 = 2* el + (nelx+1)*2 + 2*ely +2
			P6 = 2* el + (nelx+1)*2 + 2*ely +3 
			P7 = 2* el + (nelx+1)*2 + 2*ely 
			P8 = 2* el + (nelx+1)*2 + 2*ely + 1
			edofMat[el,:] = np.array([P1,P2,P3,P4,P5,P6,P7,P8])
			
			#edofMat[el,:] = [1,2,3,4,5,6,7,8]

	#print(edofMat)
	
	iK = np.kron(edofMat,np.ones((1,8))).flatten()
	jK = np.kron(edofMat,np.ones((8,1))).flatten()
	
	# Filter


	# BC and support
	dofs = np.arange((nely+1)*(nelx+1)*2)
	fixed = np.union1d(dofs[0:2*(nelx+1)*(nely+1):2*(nelx+1)],np.array([(nelx*2+1)]))
	free = np.setdiff1d(dofs,fixed)

	# Solution and rhs-vector
	f = np.zeros((ndof,1))
	u = np.zeros((ndof,1))

	# set load
	f[2*(nelx+1)*nely+1,0] = 1
	#fixed=np.union1d(dofs[1:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))

	if deb_mode==1: print('dofs: ' + str(dofs)) #debug
	if deb_mode==1: print('fixed: ' + str(fixed)) #debug
	if deb_mode==1: print('free: ' + str(free)) #debug
	#if deb_mode==1: print('f: ' + str(f)) #debug
	# LOOP -  stup
	ce = np.ones(nelx*nely)
	c = np.ones(nelx*nely)
	loop = 1
	change = 0.015 
	while change>0.01 and loop<1000:
		loop += 1
		# FE-solver
		sK = ((Ke.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(E0-Emin))).flatten(order='F')
		K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
		# Remove constrained dofs from matrix
		K = K[free,:][:,free]
		# Solve System
		u[free,0] = spsolve(K,f[free,0])
		# get ojective C
		ce[:] = (np.dot(u[edofMat].reshape(nelx*nely,8),Ke)*u[edofMat].reshape(nelx*nely,8) ).sum(1)
		obj = ((Emin + xPhys ** (penal-1) * (E0-Emin))*ce).sum()

		print(obj)



		change -= 0.005 # debug

		if fem_plot_b==1: plot_FEM(u,nelx,nely) #debug


	#print(K)

	if mesh_plot_b == 1: plot_mesh(nelx,nely)



	print('---DONE--- \n \n')
# create Elemet Stiffness Matrix
def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	Ke = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
	[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
	[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
	[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
	[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
	[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
	[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
	[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
	#print(KE)
	return (Ke)


def plot_mesh(nelx,nely):
	print('... plot mesh')
	fig = plt.figure()
	#ax = fig.gca(projection='3d')
	#X = np.arange(0,nelx,1)
	#Y = np.arange(0,nely,1)
	X = np.linspace(0, nelx, nelx) 
	Y = np.linspace(0, nely, nely)
	X,Y = np.meshgrid(X,Y)
	#Z = X * np.sinc(X ** 2 + Y ** 2)
	Z=np.ones(nelx)
	C = np.ones((nelx,nely))


	#plt.pcolormesh(X,Y,Z,cmap = cm.gray)
	plt.pcolormesh(C,edgecolors='red')
	plt.show()

def plot_FEM(u,nelx,nely):
	plt.title('X-Direction')
	ndof = 2*(nely+1)*(nelx+1)
	print('... plot FEM')
	grid = u[0:ndof:2].reshape((nely+1,nelx+1))
	plt.imshow(grid , extent=(0,nelx,0,nely), interpolation='nearest')
	plt.show()
	#plt.title('Y-Direction')
	#ndof = 2*(nely+1)*(nelx+1)
	#print('... plot FEM')
	#grid = u[1:ndof:2].reshape((nely+1,nelx+1))
	#plt.imshow(grid , extent=(0,nelx,0,nely), interpolation='nearest')
	#plt.show()

# Optimality criterion

######### START ######

main(4,3,1,1,1)
#main(4,3,1,1,1)
