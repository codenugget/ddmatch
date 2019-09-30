# Encoding: UTF-8
"""Usage: python example2.py
Should work from any directory. Will create lots of folders and
subfolders in the working directory. Running from inside "Example2" is
recommended.

At the end of this file, 5 functions (testX()) are called. This can
take a long time (about 35min for the author of this file). Uncomment
to run specific tests faster."""

# For Python2 compatibility:
from __future__ import print_function, division

# Relative import in python:
# https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py/27876800#27876800
# Explanation: When invoked as "python dirs/example2.py" the variable
# '__file__' is the string "dirs/example2.py". Then path.abspath
# interprets the string as a path (relative to the current directory of
# the executing shell) and returns an absolute path. Next path.dirname
# extracts the parent directory of this file (i.e. it removes
# "example2.py" from the string). Calling it again removes "Example2".
# Finally the parent directory is added to the search path for modules.
# And then the module is imported as usual.
import sys
from os import path
#print(__file__)
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import difforma_base

# Other imports
import os
import matplotlib
matplotlib.use('AGG') # non-interactive raster graphics backend
from matplotlib import pyplot as plt
import numpy as np
from random import randint
from scipy import ndimage
import time

# Function definitions
def plot_warp(xphi, yphi, downsample='auto', **kwarg):
  """Borrowed from ../difforma_base_example.ipynb."""
  if (downsample == 'auto'):
    skip = np.max([xphi.shape[0]/32,1])
  elif (downsample == 'no'):
    skip = 1
  else:
    skip = downsample
  plt.plot(xphi[:,skip::skip],yphi[:,skip::skip],'black',\
           xphi[skip::skip,::1].T,yphi[skip::skip,::1].T,'black', **kwarg)


def run_and_save_example(I0, I1, subpath, description):
  """Utility function to run and export results for a test case."""
  print('"%s": Initializing' % subpath)
  dm = difforma_base.DiffeoFunctionMatching(
    source=I0, target=I1,
    alpha=0.001, beta=0.03, sigma=0.05
  )
  print('"%s": Running' % subpath)
  dm.run(1000, epsilon=0.1)
  
  print('"%s": Creating plots' % subpath)
  if not path.exists(subpath):
    os.makedirs(subpath)
  
  # 2x2 overview plot
  plt1 = plt.figure(1, figsize=(11.7,9))
  plt.clf()

  plt.subplot(2,2,1)
  plt.imshow(dm.target, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
  plt.colorbar()
  plt.title('Target image')

  plt.subplot(2,2,2)
  plt.imshow(dm.source, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
  plt.colorbar()
  plt.title('Template image')

  plt.subplot(2,2,3)
  plt.imshow(dm.I, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
  plt.colorbar()
  plt.title('Warped image')

  plt.subplot(2,2,4)
  # Forward warp    
  phix = dm.phix
  phiy = dm.phiy
  # Uncomment for backward warp
  #phix = dm.phiinvx
  #phiy = dm.phiinvy
  plot_warp(phix, phiy, downsample=4)
  plt.axis('equal')
  warplim = [phix.min(), phix.max(), phiy.min(), phiy.max()]
  warplim[0] = min(warplim[0], warplim[2])
  warplim[2] = warplim[0]
  warplim[1] = max(warplim[1], warplim[3])
  warplim[3] = warplim[1]

  plt.axis(warplim)
  plt.gca().invert_yaxis()
  plt.gca().set_aspect('equal')
  plt.title('Warp')
  plt.grid()
  plt1.savefig(path.join(subpath, 'overview.png'), dpi=300, bbox_inches='tight')

  # Energy convergence plot
  plt2 = plt.figure(2, figsize=(8,4.5))
  plt.clf()
  plt.plot(dm.E)
  plt.grid()
  plt.ylabel('Energy')
  plt2.savefig(os.path.join(subpath, 'convergence.png'), dpi=150, bbox_inches='tight')

  # Dedicated warp plot (forward only)
  plt3 = plt.figure(3, figsize=(10,10))
  plt.clf()
  plot_warp(phix, phiy, downsample=4, )
  plt.axis('equal')
  warplim = [phix.min(), phix.max(), phiy.min(), phiy.max()]
  warplim[0] = min(warplim[0], warplim[2])
  warplim[2] = warplim[0]
  warplim[1] = max(warplim[1], warplim[3])
  warplim[3] = warplim[1]

  plt.axis(warplim)
  plt.gca().invert_yaxis()
  plt.gca().set_aspect('equal')
  plt.title('Warp')
  plt.axis('off')
  #plt.grid(color='black')
  plt3.savefig(path.join(subpath, 'warp.png'), dpi=150, bbox_inches='tight')
  
  # Output description
  with open(path.join(subpath, 'description.txt'), 'w') as f:
    f.write(subpath)
    f.write('\n')
    f.write(description)
    
  print('Done at ' + time.asctime() + '\n')

def test1():
  description = '''
Translations ought to be exactly achievable even with periodic
boundary conditions. This test verifies that presumption.

It seems that images that are non-smooth on the pixel level causes
divergence. Some binary "smoothing" method may be employed. Instead
of single pixels, small squares or circles could be used.
'''
  nPoints = 30
  delta = 20
  I0 = np.zeros((64,64))
  I1 = I0 + 0
  for i in range(nPoints):
    px = randint(5,25)
    py = randint(5,25)
    I0[px,py] = 1
    I1[px+delta,py+delta] = 1

  subpath = path.join('translation', 'low_density')
  run_and_save_example(I0, I1, subpath, description)
  
  subpath = path.join('translation', 'low_density_smoothed')
  I2 = ndimage.gaussian_filter(I0, sigma=1)
  I3 = ndimage.gaussian_filter(I1, sigma=1)
  run_and_save_example(I2, I3, subpath, description)

  subpath = path.join('translation', 'medium_density')
  nPoints = 200
  for i in range(nPoints):
    px = randint(5,25)
    py = randint(5,25)
    I0[px,py] = 1
    I1[px+delta,py+delta] = 1
  run_and_save_example(I0, I1, subpath, description)

  subpath = path.join('translation', 'medium_density_smoothed')
  I2 = ndimage.gaussian_filter(I0, sigma=1)
  I3 = ndimage.gaussian_filter(I1, sigma=1)
  run_and_save_example(I2, I3, subpath, description)
  
  subpath = path.join('translation', 'high_density')
  nPoints = 400
  for i in range(nPoints):
    px = randint(5,25)
    py = randint(5,25)
    I0[px,py] = 1
    I1[px+delta,py+delta] = 1
  run_and_save_example(I0, I1, subpath, description)
  
  subpath = path.join('translation', 'high_density_smoothed')
  I2 = ndimage.gaussian_filter(I0, sigma=1)
  I3 = ndimage.gaussian_filter(I1, sigma=1)
  run_and_save_example(I2, I3, subpath, description)

  subpath = path.join('translation', 'full_density')
  I0[5:26,5:26] = 1
  I1[(5+delta):(26+delta),(5+delta):(26+delta)] = 1
  run_and_save_example(I0, I1, subpath, description)
  
  subpath = path.join('translation', 'full_density_smoothed')
  I2 = ndimage.gaussian_filter(I0, sigma=1)
  I3 = ndimage.gaussian_filter(I1, sigma=1)
  run_and_save_example(I2, I3, subpath, description)

def test2():
  description = '''
How smooth do things need to be?
'''
  nPoints = 10
  # Diagonal translation in pixels
  delta = 30
  # Use the same random points for all block sizes
  points = [randint(50,150) for _ in range(2*nPoints)]
  for p in [1, 2, 4, 8, 16, 32, 64]:
    I0 = np.zeros((256,256))
    I1 = I0 + 0
    for i in range(nPoints):
      px = points[2*i]
      py = points[2*i+1]
      I0[px:(px+p),py:(py+p)] = 1
      I1[(px+delta):(px+delta+p),(py+delta):(py+delta+p)] = 1
    
    subpath = path.join('blocks', '%dpx' % p)
    run_and_save_example(I0, I1, subpath, description)
    
    subpath = path.join('blocks', '%dpx_smoothed' % p)
    I2 = ndimage.gaussian_filter(I0, sigma=p/4)
    I3 = ndimage.gaussian_filter(I1, sigma=p/4)
    run_and_save_example(I2, I3, subpath, description)

def test3():
  description = '''
Stretching is not a symmetry, neither locally nor globally. So we
here we expect that the algotithm doesn't quite converge. Because the
energy acts as a P-regulator which means there will be a residual
error. More precicely if there's a perfect match then the intensity
term will be zero, but the regularizer is non-zero and will pull the
diffeomorphism a bit towards the identity map.

Furthermore, the fact that the pullback is used means that the
intensity needs to be lower in the larger pattern for a perfect match.

Finally, the torus periodicity is a problem. Since total volume is
conserved a stretch in one location must be associated with a shrink in
another. In practice, the shrinking is spread out over as large an area
as possible because we have an L2-norm:
  E(a) = \sum_i=1^N |a_i|^b
under the constraint \sum_i a_i = constant is minimized when
|a_i|=const assuming b>1 (for b<1 the optimum is to have one a_i \neq 0
and all others zeros, whereas for b=1 it doesn't matter as long as all
a_i:s are parallell).
'''
  x0 = 50
  s1 = 50
  for s2 in [25, 75, 100, 150, 200]:
    I0 = np.zeros((256,256))
    I0[x0:(x0+s1),x0:(x0+s1)] = 1
    I1 = np.zeros((256,256))
    I1[x0:(x0+s2),x0:(x0+s2)] = 1
    subpath = path.join('stretch', 'x%.1f' % (s2/s1))
    run_and_save_example(I1, I0, subpath, description)
    
    subpath = path.join('stretch', 'x%.1f_smoothed' % (s2/s1))
    I2 = ndimage.gaussian_filter(I0, sigma=4)
    I3 = ndimage.gaussian_filter(I1, sigma=4)
    run_and_save_example(I3, I2, subpath, description)
    
def test4():
  description = '''
Rotation is a local symmetry, but not a global one. Therefore all
rotations are associated with a penalty spread out over the entire
domain. For the same reasoning as in the "stretch" tests it will be as
homogeneous as possible.

Local stretching seems to be favored over rotation for simple objects.
'''
  N = 256  # size in pixels
  a = 4    # "squarifier"
  b = 5    # cutoff exponent
  r = 30   # Pattern radius scale in pixels
  coords = range(-N//2, (N-1)//2 + 1)
  X, Y = np.meshgrid(coords, coords)
  R = (np.abs(X)**a + np.abs(Y)**a)**(1/a)
  I1 = np.exp(-(R/r)**b)
  
  for angle_deg in [5, 15, 30, 45]:
    th = angle_deg*np.pi/180
    # Anti-clockwise rotation by angle th
    Xp = X*np.cos(th) - Y*np.sin(th)
    Yp = X*np.sin(th) + Y*np.cos(th)
    Rp = (np.abs(Xp)**a + np.abs(Yp)**a)**(1/a)
    I0 = np.exp(-(Rp/r)**b)
    subpath = path.join('rotate', '%ddeg' % angle_deg)
    run_and_save_example(I0, I1, subpath, description)
    
def test5():
  description = '''
Most automatic matching tasks work best when the objects of study have
low symmetry and/or high entropy and/or significant (correlated) noise.
This is an attempt at an ideal situation for the matcher. Importantly,
it still needs to be smooth so that gradients contain "global"
information. We also need to be mindful of the finite torus volume.

To avoid conflating differences with effective difference in sigma, all
images are max-normalized before processing. Individual pixels should
have intensity of order unity.
'''
  N = 256  # size in pixels
  r = 40   # Pattern radius scale in pixels
  r0 = 1   # Smooth region scale in pixels
  coords = range(-N//2, (N-1)//2 + 1)
  X, Y = np.meshgrid(coords, coords)
  cutoff = np.exp(-(X*X+Y*Y)/(2*r*r))

  # Random affine transformation not too far away from identity
  dx = np.random.randn(2)*r0
  A = np.array([[1,0],[0,1]]) + 0.1*np.random.randn(2,2)
  
  def transform(pos):
    """Applies geometric transformation to tuple."""
    x = pos[0] - N//2
    y = pos[1] - N//2
    xp = N//2 + dx[0] + A[0,0]*x + A[0,1]*y
    yp = N//2 + dx[1] + A[1,0]*x + A[1,1]*y
    return (xp,yp)
  
  subpath = path.join('noise', 'low_pass')
  I1 = np.random.randn(N,N)
  I1 = cutoff*ndimage.gaussian_filter(I1, sigma=r0)
  I1 = I1/np.max(np.abs(I1))
  I0 = ndimage.geometric_transform(I1, transform)
  #run_and_save_example(I0, I1, subpath, description)
  
  # Create wave vectors
  coord = np.array(range(N), dtype=np.float64)
  kX, kY = np.meshgrid(coord, coord)
  kX[kX >= N//2] -= N
  kY[kY >= N//2] -= N
  # Normalize so the range is [-pi, pi)
  kX *= np.pi/N
  kY *= np.pi/N
  
  subpath = path.join('noise', 'brownian')
  I1 = np.random.randn(N,N) + 1j*np.random.randn(N,N)
  # Notice the large scale cutoff (small X and Y since we are in wave-
  # vector space here).
  I1 = I1/(0.1**2 + kX**2 + kY**2)
  I1 = np.real(np.fft.fftn(I1, axes=(0,1)))
  # Could include r0 in the frequency space factor instead, but getting
  # the units right are harder than an filtering step...
  I1 = cutoff*ndimage.gaussian_filter(I1, sigma=r0)
  I1 = I1/np.max(np.abs(I1))
  I0 = ndimage.geometric_transform(I1, transform)
  run_and_save_example(I0, I1, subpath, description)
  
  subpath = path.join('noise', 'blue')
  I1 = np.random.randn(N,N) + 1j*np.random.randn(N,N)
  I1 = I1*(kX**2 + kY**2)
  I1 = np.real(np.fft.fftn(I1, axes=(0,1)))
  I1 = cutoff*ndimage.gaussian_filter(I1, sigma=r0)
  I1 = I1/np.max(np.abs(I1))
  I0 = ndimage.geometric_transform(I1, transform)
  run_and_save_example(I0, I1, subpath, description)
    
test1()
test2()
test3()
test4()
test5()
