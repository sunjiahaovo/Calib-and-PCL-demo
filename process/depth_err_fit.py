import numpy as np

dis = np.array([1,2,5,10,15,25,50,100])/150
err = np.array([11,12.5,17.0,24.5,32.0,47.0,84.6,159.9])/1000.0

xdata = dis
ydata = err


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 
def func(x, a1, b1, a2,b2):
   u = a1*x**2+b1
   std = a2*x**2+b2
   return 1/(np.sqrt(2*np.pi)*std) * np.exp(-(x-u)**2/(2*std**2) )
  
 
 
# Fit for the parameters a, b, c of the function func:
popt, pcov = curve_fit(func, xdata, ydata)
print(popt)
print(popt/150/150)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a1=%5.3f, b1=%5.3f, a2=%5.3f, b2=%5.3f' % tuple(popt))
 

 
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
