import numpy as np

def getColor( v, vmin, vmax):

   c = np.array([1.0, 1.0, 1.0, 1.0])

   if v < vmin:
      v = vmin
   if v > vmax:
      v = vmax
   dv = vmax - vmin

   if v < (vmin + 0.25 * dv):
      c[0] = 0
      c[1] = 4 * (v - vmin) / dv
   elif v < (vmin + 0.5 * dv):
      c[0] = 0
      c[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv
   elif v < (vmin + 0.75 * dv):
      c[0] = 4 * (v - vmin - 0.5 * dv) / dv
      c[2] = 0
   else:
      c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
      c[2] = 0

   return c