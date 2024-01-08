import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# % If linear interpolation is possible, use; otherwise use the nearest neighbour's value.
class LinearNDInterpolatorExt(object):
    
  def __init__(self, points,values):
    self.funcinterp  = LinearNDInterpolator(points,values)
    self.funcnearest = NearestNDInterpolator(points,values)
    
  def __call__(self,*args):
    t = self.funcinterp(*args)
    if not np.isnan(t).any():
      return t
    else:
      return self.funcnearest(*args)