#!/usr/bin/env python
# encoding: utf-8
"""
Foundation classes for the `difforma` library.

Documentation guidelines are available `here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Created by Klas Modin on 2014-11-03 and documented in M. Bauer, S. Joshi, 
K. Modin. Diffeomorphic density matching by optimal information transport, 
SIAM J. Imaging Sci., 8(3):1718-1751, 2015

Modified by Carl-Joar Karlsson as part of his master's thesis, spring 2019.

"""

import numpy as np 
import numba

def generate_optimized_image_composition(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
	def image_compose_2d(I,xphi,yphi,Iout):
		for i in range(s):
			for j in range(s):
				xind = int(xphi[i,j])
				yind = int(yphi[i,j])
				xindp1 = xind+1
				yindp1 = yind+1
				deltax = xphi[i,j]-float(xind)
				deltay = yphi[i,j]-float(yind)
				
				# Id xdelta is negative it means that xphi is negative, so xind
				# is larger than xphi. We then reduce xind and xindp1 by 1 and
				# after that impose the periodic boundary conditions.
				if (deltax < 0 or xind < 0):
					deltax += 1.0
					xind -= 1
					xind %= s
					xindp1 -= 1
					xindp1 %= s
				elif (xind >= s):
					xind %= s
					xindp1 %= s
				elif (xindp1 >= s):
					xindp1 %= s

				if (deltay < 0 or xind < 0):
					deltay += 1.0
					yind -= 1
					yind %= s
					yindp1 -= 1
					yindp1 %= s
				elif (yind >= s):
					yind %= s
					yindp1 %= s
				elif (yindp1 >= s):
					yindp1 %= s
					
				onemdeltax = 1.-deltax
				onemdeltay = 1.-deltay

				Iout[i,j] = I[yind,xind]*onemdeltax*onemdeltay+\
					I[yind,xindp1]*deltax*onemdeltay+\
					I[yindp1,xind]*deltay*onemdeltax+\
					I[yindp1,xindp1]*deltay*deltax

	return image_compose_2d

def generate_optimized_diffeo_evaluation(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:],f8[:],f8[:],f8[:])')
	def eval_diffeo_2d(xpsi,ypsi,xvect,yvect,xout,yout):
	# Evaluate diffeo psi(x,y) for each pair in xvect, yvect. 
	# Assuming psi is periodic.

		d = xvect.shape[0]

		for i in range(d):
			xind = int(xvect[i])
			yind = int(yvect[i])
			xindp1 = xind+1
			yindp1 = yind+1
			deltax = xvect[i]-float(xind)
			deltay = yvect[i]-float(yind)
			xshift = 0.0
			xshiftp1 = 0.0
			yshift = 0.0
			yshiftp1 = 0.0
			
			# Id xdelta is negative it means that xphi is negative, so xind
			# is larger than xphi. We then reduce xind and xindp1 by 1 and
			# after that impose the periodic boundary conditions.
			if (deltax < 0 or xind < 0):
				deltax += 1.0
				xind -= 1
				xindp1 -= 1
				xind %= s
				xshift = -float(s) # Should use floor_divide here instead.
				if (xindp1 < 0):
					xindp1 %= s
					xshiftp1 = -float(s)
			elif (xind >= s):
				xind %= s
				xindp1 %= s
				xshift = float(s)
				xshiftp1 = float(s)
			elif (xindp1 >= s):
				xindp1 %= s
				xshiftp1 = float(s)

			if (deltay < 0 or yind < 0):
				deltay += 1.0
				yind -= 1
				yindp1 -= 1
				yind %= s
				yshift = -float(s) # Should use floor_divide here instead.
				if (yindp1 < 0):
					yindp1 %= s
					yshiftp1 = -float(s)
			elif (yind >= s):
				yind %= s
				yindp1 %= s
				yshift = float(s)
				yshiftp1 = float(s)
			elif (yindp1 >= s):
				yindp1 %= s
				yshiftp1 = float(s)
				
			xout[i] = (xpsi[yind,xind]+xshift)*(1.-deltax)*(1.-deltay)+\
				(xpsi[yind,xindp1]+xshiftp1)*deltax*(1.-deltay)+\
				(xpsi[yindp1,xind]+xshift)*deltay*(1.-deltax)+\
				(xpsi[yindp1,xindp1]+xshiftp1)*deltay*deltax
			
			yout[i] = (ypsi[yind,xind]+yshift)*(1.-deltax)*(1.-deltay)+\
				(ypsi[yind,xindp1]+yshift)*deltax*(1.-deltay)+\
				(ypsi[yindp1,xind]+yshiftp1)*deltay*(1.-deltax)+\
				(ypsi[yindp1,xindp1]+yshiftp1)*deltay*deltax

	return eval_diffeo_2d

def generate_optimized_diffeo_composition(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
	def diffeo_compose_2d(xpsi,ypsi,xphi,yphi,xout,yout):
	# Compute composition psi o phi. 
	# Assuming psi and phi are periodic.

		for i in range(s):
			for j in range(s):
				xind = int(xphi[i,j])
				yind = int(yphi[i,j])
				xindp1 = xind+1
				yindp1 = yind+1
				deltax = xphi[i,j]-float(xind)
				deltay = yphi[i,j]-float(yind)
				xshift = 0.0
				xshiftp1 = 0.0
				yshift = 0.0
				yshiftp1 = 0.0
				
				# Id xdelta is negative it means that xphi is negative, so xind
				# is larger than xphi. We then reduce xind and xindp1 by 1 and
				# after that impose the periodic boundary conditions.
				if (deltax < 0 or xind < 0):
					deltax += 1.0
					xind -= 1
					xindp1 -= 1
					xind %= s
					xshift = -float(s) # Should use floor_divide here instead.
					if (xindp1 < 0):
						xindp1 %= s
						xshiftp1 = -float(s)
				elif (xind >= s):
					xind %= s
					xindp1 %= s
					xshift = float(s)
					xshiftp1 = float(s)
				elif (xindp1 >= s):
					xindp1 %= s
					xshiftp1 = float(s)

				if (deltay < 0 or yind < 0):
					deltay += 1.0
					yind -= 1
					yindp1 -= 1
					yind %= s
					yshift = -float(s) # Should use floor_divide here instead.
					if (yindp1 < 0):
						yindp1 %= s
						yshiftp1 = -float(s)
				elif (yind >= s):
					yind %= s
					yindp1 %= s
					yshift = float(s)
					yshiftp1 = float(s)
				elif (yindp1 >= s):
					yindp1 %= s
					yshiftp1 = float(s)

				xout[i,j] = (xpsi[yind,xind]+xshift)*(1.-deltax)*(1.-deltay)+\
					(xpsi[yind,xindp1]+xshiftp1)*deltax*(1.-deltay)+\
					(xpsi[yindp1,xind]+xshift)*deltay*(1.-deltax)+\
					(xpsi[yindp1,xindp1]+xshiftp1)*deltay*deltax
				
				yout[i,j] = (ypsi[yind,xind]+yshift)*(1.-deltax)*(1.-deltay)+\
					(ypsi[yind,xindp1]+yshift)*deltax*(1.-deltay)+\
					(ypsi[yindp1,xind]+yshiftp1)*deltay*(1.-deltax)+\
					(ypsi[yindp1,xindp1]+yshiftp1)*deltay*deltax

	return diffeo_compose_2d

def generate_optimized_diffeo_gradient_y(image):
	# Handles warping over indices>shape
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:])')
	def diffeo_gradient_y_2d(I,dIdx,dIdy):
		i = 0
		j = 0
		ip1 = i+1
		jp1 = j+1
		im1 = s-1
		jm1 = s-1
		im2 = s-2
		jm2 = s-2
		for j in range(s):
			dIdy[0,j] = (I[ip1,j]-I[im1,j]+s)/2.0
			dIdy[im1,j] = (I[i,j]-I[im2,j]+s)/2.0
		j = 0
		for i in range(s):
			dIdx[i,0] = (I[i,jp1]-I[i,jm1])/2.0
			dIdx[i,jm1] = (I[i,j]-I[i,jm2])/2.0
		im1 = 0
		jm1 = 0
		for i in range(1,s-1):
			ip1 = i+1
			for j in range(s):
				dIdy[i,j] = (I[ip1,j]-I[im1,j])/2.0
			im1 = i
		for j in range(1,s-1):
			jp1 = j+1
			for i in range(s):
				dIdx[i,j] = (I[i,jp1]-I[i,jm1])/2.0
			jm1 = j

	return diffeo_gradient_y_2d

def generate_optimized_diffeo_gradient_x(image):
	# Handles warping over indices>shape
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:])')
	def diffeo_gradient_x_2d(I,dIdx,dIdy):
		i = 0
		j = 0
		ip1 = i+1
		jp1 = j+1
		im1 = s-1
		jm1 = s-1
		im2 = s-2
		jm2 = s-2
		for j in range(s):
			dIdy[0,j] = (I[ip1,j]-I[im1,j])/2.0
			dIdy[im1,j] = (I[i,j]-I[im2,j])/2.0
		j = 0
		for i in range(s):
			dIdx[i,0] = (I[i,jp1]-I[i,jm1]+s)/2.0
			dIdx[i,jm1] = (I[i,j]-I[i,jm2]+s)/2.0
		im1 = 0
		jm1 = 0
		for i in range(1,s-1):
			ip1 = i+1
			for j in range(s):
				dIdy[i,j] = (I[ip1,j]-I[im1,j])/2.0
			im1 = i
		for j in range(1,s-1):
			jp1 = j+1
			for i in range(s):
				dIdx[i,j] = (I[i,jp1]-I[i,jm1])/2.0
			jm1 = j

	return diffeo_gradient_x_2d

def generate_optimized_image_gradient(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:])')
	def image_gradient_2d(I,dIdx,dIdy):
		im1 = s-1
		jm1 = s-1
		for i in range(s-1):
			ip1 = i+1
			for j in range(s-1):
				jp1 = j+1
				dIdy[i,j] = (I[ip1,j]-I[im1,j])/2.0
				dIdx[i,j] = (I[i,jp1]-I[i,jm1])/2.0
				jm1 = j
			dIdy[i,s-1] = (I[ip1,s-1]-I[im1,s-1])/2.0
			dIdx[i,s-1] = (I[i,0]-I[i,s-2])/2.0
			jm1 = s-1
			im1 = i
		for j in range(s-1):
			jp1 = j+1
			dIdy[s-1,j] = (I[0,j]-I[im1,j])/2.0
			dIdx[s-1,j] = (I[s-1,jp1]-I[s-1,jm1])/2.0
			jm1 = j
		dIdy[s-1,s-1] = (I[0,s-1]-I[s-2,s-1])/2.0
		dIdx[s-1,s-1] = (I[s-1,0]-I[s-1,s-2])/2.0

	return image_gradient_2d

def generate_optimized_image_gradient_forward(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:])')
	def image_gradient_2d_forward(I,dIdx,dIdy):
		im1 = s-1
		jm1 = s-1
		for i in range(s-1):
			ip1 = i+1
			for j in range(s-1):
				jp1 = j+1
				dIdy[i,j] = (I[ip1,j]-I[im1,j])/2.0
				dIdx[i,j] = (I[i,jp1]-I[i,jm1])/2.0
				jm1 = j
			dIdy[i,s-1] = (I[ip1,s-1]-I[im1,s-1])/2.0
			dIdx[i,s-1] = (I[i,0]-I[i,s-2])/2.0
			jm1 = s-1
			im1 = i
		for j in range(s-1):
			jp1 = j+1
			dIdy[s-1,j] = (I[0,j]-I[im1,j])/2.0
			dIdx[s-1,j] = (I[s-1,jp1]-I[s-1,jm1])/2.0
			jm1 = j
		dIdy[s-1,s-1] = (I[0,s-1]-I[s-2,s-1])/2.0
		dIdx[s-1,s-1] = (I[s-1,0]-I[s-1,s-2])/2.0

	return image_gradient_2d_forward


def generate_optimized_divergence(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:])')
	def divergence_2d(vx,vy,divv):
		im1 = s-1
		jm1 = s-1
		for i in range(s-1):
			ip1 = i+1
			for j in range(s-1):
				jp1 = j+1
				divv[i,j] = (vy[ip1,j]-vy[im1,j] + vx[i,jp1]-vx[i,jm1])/2.0
				jm1 = j
			divv[i,s-1] = (vy[ip1,s-1]-vy[im1,s-1] + vx[i,0]-vx[i,s-2])/2.0
			jm1 = s-1
			im1 = i
		for j in range(s-1):
			jp1 = j+1
			divv[s-1,j] = (vy[0,j]-vy[im1,j] + vx[s-1,jp1]-vx[s-1,jm1])/2.0
			jm1 = j
		divv[s-1,s-1] = (vy[0,s-1]-vy[s-2,s-1] + vx[s-1,0]-vx[s-1,s-2])/2.0

	return divergence_2d

def generate_optimized_jacobian_forward(image):
	s = image.shape[0]
	if (len(image.shape) != 2):
		raise(NotImplementedError('Only 2d images are allowed so far.'))
	if (image.shape[1] != s):
		raise(NotImplementedError('Only square images are allowed so far.'))
	if (image.dtype != np.float64):
		raise(NotImplementedError('Only float64 images are allowed so far.'))

	@numba.njit('f8(f8,f8,f8,f8)')
	def det_2d(a11,a21,a12,a22):
		return a11*a22 - a12*a21

	@numba.njit('void(f8[:,:],f8[:,:],f8[:,:])')
	def jacobian_2d_forward(xphi,yphi,jac):
		for i in range(s-1):
			for j in range(s-1):
				dxphi_dx = xphi[i,j+1]-xphi[i,j]
				dxphi_dy = xphi[i+1,j]-xphi[i,j]
				dyphi_dx = yphi[i,j+1]-yphi[i,j]
				dyphi_dy = yphi[i+1,j]-yphi[i,j]
				jac[i,j] = det_2d(dxphi_dx,dyphi_dx,\
								  dxphi_dy,dyphi_dy)

			dxphi_dx = xphi[i,0]+s-xphi[i,s-1]
			dxphi_dy = xphi[i+1,s-1]-xphi[i,s-1]
			dyphi_dx = yphi[i,0]-yphi[i,s-1]
			dyphi_dy = yphi[i+1,s-1]-yphi[i,s-1]
			jac[i,s-1] = det_2d(dxphi_dx,dyphi_dx,\
							  dxphi_dy,dyphi_dy)
		for j in range(s-1):
			dxphi_dx = xphi[s-1,j+1]-xphi[s-1,j]
			dxphi_dy = xphi[0,j]-xphi[s-1,j]
			dyphi_dx = yphi[s-1,j+1]-yphi[s-1,j]
			dyphi_dy = yphi[0,j]+s-yphi[s-1,j]
			jac[s-1,j] = det_2d(dxphi_dx,dyphi_dx,\
							  dxphi_dy,dyphi_dy)

		dxphi_dx = xphi[s-1,0]+s-xphi[s-1,s-1]
		dxphi_dy = xphi[0,s-1]-xphi[s-1,s-1]
		dyphi_dx = yphi[s-1,0]-yphi[s-1,s-1]
		dyphi_dy = yphi[0,s-1]+s-yphi[s-1,s-1]
		jac[s-1,s-1] = det_2d(dxphi_dx,dyphi_dx,\
						  dxphi_dy,dyphi_dy)

	return jacobian_2d_forward



class DiffeoFunctionMatching(object):
	"""
	Implementation of the two component function matching algorithm.

	The computations are accelerated using the `numba` library.
	"""

	def __init__(self, source, target, alpha=0.001, beta=0.03, sigma=0.5, compute_phi=True):
		"""
		Initialize the matching process.

		Implements to algorithm in the paper by Modin and Karlsson (to be published).

		Parameters
		----------
		source : array_like
			Numpy array (float64) for the source image.
		target : array_like
			Numpy array (float64) for the target image.
			Must be of the same shape as `source`.
		sigma : float
			Parameter for penalizing change of volume (divergence).
		compute_phi : bool
			Whether to compute the forward phi mapping or not.

		Returns
		-------
		None or Python generator

		See also
		--------
		N/A

		Examples
		--------
		N/A
		"""	
		
		self.source = source
		self.target = target
		self.compute_phi = compute_phi
		I0 = target
		I1 = source

		# Check input
		if (I0.shape != I1.shape):
			raise(TypeError('Source and target images must have the same shape.'))
		if (I0.dtype != I1.dtype):
			raise(TypeError('Source and target images must have the same dtype.'))
		if (sigma < 0):
			raise(ValueError('Paramter sigma must be positive.'))
		for d in I1.shape:
			if (d != I1.shape[0]):
				raise(NotImplementedError('Only square images allowed so far.'))
		if (len(I1.shape) != 2):
			raise(NotImplementedError('Only 2d images allowed so far.'))

		# Create optimized algorithm functions
		self.image_compose = generate_optimized_image_composition(I1)
		self.diffeo_compose = generate_optimized_diffeo_composition(I1)
		self.image_gradient = generate_optimized_image_gradient(I1)
		self.diffeo_gradient_y = generate_optimized_diffeo_gradient_y(I1)
		self.diffeo_gradient_x = generate_optimized_diffeo_gradient_x(I1)
		self.evaluate = generate_optimized_diffeo_evaluation(I1)

		# Allocate and initialize variables
		self.alpha = alpha
		self.beta = beta
		self.sigma = sigma
		self.s = I1.shape[0]
		self.E = []
		self.I0 = np.zeros_like(I0)
		np.copyto(self.I0,I0)
		self.I1 = np.zeros_like(I1)
		np.copyto(self.I1,I1)
		self.I = np.zeros_like(I1)
		np.copyto(self.I,I1)
		self.dIdx = np.zeros_like(I1)
		self.dIdy = np.zeros_like(I1)
		self.vx = np.zeros_like(I1)
		self.vy = np.zeros_like(I1)
		self.divv = np.zeros_like(I1)
				
		# Allocate and initialize the diffeos
		x = np.linspace(0, self.s, self.s, endpoint=False)
		[self.idx, self.idy] = np.meshgrid(x, x)
		self.phiinvx = self.idx.copy()
		self.phiinvy = self.idy.copy()
		self.psiinvx = self.idx.copy()
		self.psiinvy = self.idy.copy()
		if self.compute_phi:
			self.phix = self.idx.copy()
			self.phiy = self.idy.copy()	
			self.psix = self.idx.copy()
			self.psiy = self.idy.copy()	
		self.tmpx = self.idx.copy()
		self.tmpy = self.idy.copy()

		# test case
		#self.phiinvy += 5.e-8*self.phiinvy**2*(self.s-1-self.phiinvy)**2 + 5.e-8*self.phiinvx**2*(self.s-1-self.phiinvx)**2# compare with += 3.e-7*(...)
		#self.phiinvx += 1.e-7*self.phiinvx**2*(self.s-1-self.phiinvx)**2


		# Allocate and initialize the metrics
		self.g = np.array([[np.ones_like(I1),np.zeros_like(I1)],[np.zeros_like(I1),np.ones_like(I1)]])
		self.h = np.array([[np.ones_like(I1),np.zeros_like(I1)],[np.zeros_like(I1),np.ones_like(I1)]])
		self.hdet = np.zeros_like(I1)
		self.dhaadx = np.zeros_like(I1)
		self.dhbadx = np.zeros_like(I1)
		self.dhabdx = np.zeros_like(I1)
		self.dhbbdx = np.zeros_like(I1)
		self.dhaady = np.zeros_like(I1)
		self.dhbady = np.zeros_like(I1)
		self.dhabdy = np.zeros_like(I1)
		self.dhbbdy = np.zeros_like(I1)
		self.yddy = np.zeros_like(I1)
		self.yddx = np.zeros_like(I1)
		self.xddy = np.zeros_like(I1)
		self.xddx = np.zeros_like(I1)
		self.G = np.zeros_like(np.array([self.g,self.g]))
		self.Jmap = np.zeros_like(np.array([I1,I1]))

		# Create wavenumber vectors
		k = [np.hstack((np.arange(n//2),np.arange(-n//2,0))) for n in self.I0.shape]

		# Create wavenumber tensors
		K = np.meshgrid(*k, sparse=False, indexing='ij')

		# Create Fourier multiplicator
		self.multipliers = np.ones_like(K[0])
		self.multipliers = self.multipliers*self.alpha
		for Ki in K:
			Ki = Ki*self.beta
			self.multipliers = self.multipliers+Ki**2
		if self.alpha == 0:
			self.multipliers[0,0]=1.0#self.multipliers[(0 for _ in self.s)] = 1. # Avoid division by zero
			self.Linv = 1./self.multipliers
			self.multipliers[0,0]=0.
		else:
			self.Linv = 1./self.multipliers

		
	def run(self, niter=300, epsilon=0.1):
		"""
		Carry out the matching process.

		Implements to algorithm in the paper by Modin and Karlsson (to appear).

		Parameters
		----------
		niter : int
			Number of iterations to take.
		epsilon : float
			The stepsize in the gradient descent method.
		yielditer : bool
			If `True`, then a yield statement is executed at the start of
			each iterations. This is useful for example when animating 
			the warp in real-time.

		Returns
		-------
		None or Python generator

		See also
		--------
		N/A

		Examples
		--------
		N/A
		"""		

		kE = len(self.E)
		self.E = np.hstack((self.E,np.zeros(niter)))

		for k in range(niter):
			
			# OUTPUT
			np.copyto(self.tmpx, self.I)
			np.copyto(self.tmpy, self.I0)
			self.tmpx = self.tmpx-self.tmpy
			self.tmpx **= 2
			self.E[k+kE] = self.tmpx.sum()
			np.copyto(self.tmpx, (self.h[0,0]-self.g[0,0])**2+(self.h[1,0]-self.g[1,0])**2+\
				(self.h[0,1]-self.g[0,1])**2+(self.h[1,1]-self.g[1,1])**2)
			self.E[k+kE] += self.sigma*self.tmpx.sum()

			self.image_compose(self.I1, self.phiinvx, self.phiinvy, self.I)
			
			self.diffeo_gradient_y(self.phiinvy, self.yddx, self.yddy)
			self.diffeo_gradient_x(self.phiinvx, self.xddx, self.xddy)
			np.copyto(self.h[0,0], self.yddy*self.yddy+self.xddy*self.xddy)
			np.copyto(self.h[1,0], self.yddx*self.yddy+self.xddx*self.xddy)
			np.copyto(self.h[0,1], self.yddy*self.yddx+self.xddy*self.xddx)
			np.copyto(self.h[1,1], self.yddx*self.yddx+self.xddx*self.xddx)

			self.image_gradient(self.h[0,0], self.dhaadx, self.dhaady)
			self.image_gradient(self.h[0,1], self.dhabdx, self.dhabdy)
			self.image_gradient(self.h[1,0], self.dhbadx, self.dhbady)
			self.image_gradient(self.h[1,1], self.dhbbdx, self.dhbbdy)

			np.copyto(self.Jmap[0], -(self.h[0,0]-self.g[0,0])*self.dhaady -(self.h[0,1]-self.g[0,1])*self.dhabdy -(self.h[1,0]-self.g[1,0])*self.dhbady -(self.h[1,1]-self.g[1,1])*self.dhbbdy +\
				2*self.dhaady*self.h[0,0] + 2*self.dhabdx*self.h[0,0] + 2*self.dhbady*self.h[1,0] + 2*self.dhbbdx*self.h[1,0] +\
				2*(self.h[0,0]-self.g[0,0])*self.dhaady + 2*(self.h[1,0]-self.g[1,0])*self.dhbady + 2*(self.h[0,1]-self.g[0,1])*self.dhaadx + 2*(self.h[1,1]-self.g[1,1])*self.dhbadx)
			
			np.copyto(self.Jmap[1], -(self.h[0,0]-self.g[0,0])*self.dhaadx -(self.h[0,1]-self.g[0,1])*self.dhabdx -(self.h[1,0]-self.g[1,0])*self.dhbadx -(self.h[1,1]-self.g[1,1])*self.dhbbdx +\
				2*self.dhaady*self.h[0,1] + 2*self.dhabdx*self.h[0,1] + 2*self.dhbady*self.h[1,1] + 2*self.dhbbdx*self.h[1,1] +\
				2*(self.h[0,0]-self.g[0,0])*self.dhabdy + 2*(self.h[1,0]-self.g[1,0])*self.dhbbdy + 2*(self.h[0,1]-self.g[0,1])*self.dhabdx + 2*(self.h[1,1]-self.g[1,1])*self.dhbbdx)

			self.image_gradient(self.I, self.dIdx, self.dIdy)
			self.vx = -(self.I-self.I0)*self.dIdx + 2*self.sigma*self.Jmap[1]# axis: [1]
			self.vy = -(self.I-self.I0)*self.dIdy + 2*self.sigma*self.Jmap[0]# axis: [0]
			fftx = np.fft.fftn(self.vx)
			ffty = np.fft.fftn(self.vy)
			fftx *= self.Linv
			ffty *= self.Linv
			self.vx[:] = -np.fft.ifftn(fftx).real # vx[:]=smth will copy while vx=smth directs a pointer
			self.vy[:] = -np.fft.ifftn(ffty).real

			# STEP 4 (v = -grad E, so to compute the inverse we solve \psiinv' = -epsilon*v o \psiinv)
			np.copyto(self.tmpx, self.vx)
			self.tmpx *= epsilon
			np.copyto(self.psiinvx, self.idx)
			self.psiinvx -= self.tmpx
			if self.compute_phi: # Compute forward phi also (only for output purposes)
				np.copyto(self.psix, self.idx)
				self.psix += self.tmpx

			np.copyto(self.tmpy, self.vy)
			self.tmpy *= epsilon
			np.copyto(self.psiinvy, self.idy)
			self.psiinvy -= self.tmpy
			if self.compute_phi: # Compute forward phi also (only for output purposes)
				np.copyto(self.psiy, self.idy)
				self.psiy += self.tmpy

			self.diffeo_compose(self.phiinvx, self.phiinvy, self.psiinvx, self.psiinvy, \
								self.tmpx, self.tmpy) # Compute composition phi o psi = phi o (1-eps*v)
			np.copyto(self.phiinvx, self.tmpx)
			np.copyto(self.phiinvy, self.tmpy)
			if self.compute_phi: # Compute forward phi also (only for output purposes)
				self.diffeo_compose(self.psix, self.psiy, \
									self.phix, self.phiy, \
									self.tmpx, self.tmpy)
				np.copyto(self.phix, self.tmpx)
				np.copyto(self.phiy, self.tmpy)

if __name__ == '__main__':
	pass


# ON COPYTO
# https://stackoverflow.com/questions/6431973/how-to-copy-data-from-a-numpy-array-to-another
