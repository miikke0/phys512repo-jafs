# -*- coding: utf-8 -*-
"""
@author: mjafs
"""

#necessary package imports and global function defs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splev, splrep, BarycentricInterpolator

"Good figure template example"
# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 15})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return

#%%
"""
QUESTION 1

(a) done in notes

b)
"""
#Using framework of Prof. Sievers "num_derivs_clean.py" file

#dx=np.linspace(0,1,1001) #we could do this, but we want a wider log-range of dx
logdx=np.linspace(-15,-1,1001)
logdx2 = np.linspace(-15,-0.1,1001)
dx=10**(logdx)
# dx = logdx 
dx2 = 10**logdx2
# dx2 = logdx2

fun=np.exp
x0=1
y0=fun(x0)
y1=fun(x0+dx)
y2=fun(x0-dx)
y3=fun(x0+2*dx)
y4=fun(x0-2*dx)

x0=1
y02=fun(0.01*x0)
y12=fun(0.01*(x0+dx2))
y22=fun(0.01*(x0-dx2))
y32=fun(0.01*(x0+2*dx2))
y42=fun(0.01*(x0-2*dx2))


#calculate the derivative to 4th order using 4 points
deriv1 = (-y3 + 8*y1 - 8*y2 + y4)/(12*dx) 
deriv2 = (-y32 + 8*y12 - 8*y22 + y42)/(12*dx2) 

plt.ion() #so we don't have to click away plots!
plt.clf()


#make a log plot of our errors in the derivatives
niceFigure(True)
plt.xlabel("$dx$")
plt.ylabel("Error ($\epsilon$)")
plt.loglog(dx,np.abs(deriv1-np.exp(x0)), label = '$e^x$')
plt.loglog(dx2,np.abs(deriv2-0.01*np.exp(0.01*x0)), label = '$e^{0.01x}$')
plt.legend()
plt.savefig('deriv_errors.png', format = 'png', dpi = 500, bbox_inches = 'tight')



#%%

"""
QUESTION 2
"""

def ndiff(fun, x, dxmin, dxmax, npts, full = False):
    """
    function which when True, returns the numerical derivative for a funciton given by 'fun', the estimated dx
    which corresponds to the lowest error, and the estimated order of the error. When False, only the numerical 
    derivative is returned.
    dxmin: minimum dx value to test
    dxmax: max dx value to test
    npts: sets the number of dx values to test between dxmin and dxmax
    """
    
    x0 = x
    dx = np.linspace(dxmin, dxmax, npts)
    diff = np.zeros(len(dx))  #empty list to store difference values in later
    
    #using 3 points to calculate our numerical derivative below,
    #we will calculate two slopes:
    y0 = fun(x0 - dx)
    y1 = fun(x0)
    y2 = fun(x0 + dx)
    x1 = x0 - dx
    x2 = x0
    x3 = x0 + dx
    
    #one calculated using the right two points
    dydxR = (y2 - y1)/(x3 - x2)
    
    #one using the left two points (note the middle point is shared)
    dydxL = (y1 - y0)/(x2 - x1)
    
    dydx = (dydxL + dydxR)/2 #average of the slopes
        
    #now find the centered (numerical) derivative
    f_prime = (y0 - y2)/(2*dx)
    

    #next we find the smallest difference value between the slopes and central difference 
    diff[:] = (np.abs(dydx - f_prime))  

    mindiff_ind = np.argmin(diff)  #retrieve the corresponding index
    dx_opt = dx[mindiff_ind]    #find the corresponding dx value
    yL = fun(x0 - dx_opt)
    yR = fun(x0 + dx_opt)
    deriv = (yR - yL)/(2*dx_opt)   #re-calculate the derivative with the 'optimized dx'
    
    #get the approximate order of the error from the analytical error expression
    error = dx_opt**3   #assuming f''' is of order ~O(1) and the fact that we are using the second derivative which is to order 2 in accuracy 
    
    if full is False:
        print("The derivative found numerically is", deriv)
    elif full is True:    
        print("The derivative found numerically is",deriv, "with dx=", dx_opt,". The estimated error on this calculation is", error,)
    
def f(x):
    return np.exp(x)
    
ndiff(f, 1, 1e-15, 1e-1, 1001, True)

#%%

" QUESTION 3"


def lakeshore_T_V(data):
    """
    function to take a data set with 2 columns and use the second column to interpolate the first
    data: dtype = string. file name corresponding to the data set
    """
    
    #load in the comma-seperated data
    T, V = np.loadtxt(data, delimiter = ',', usecols = (0,1), skiprows = 1, unpack = True)
    
    #scipy needs the 'x-values' to be strictly increasing
    if V[0]>V[-1]:
        V = V[::-1]   
        T = T[::-1]
    else:
        V = V
        T = T
    
    #generate the interpolated object using a cubic spline    
    cs = CubicSpline(V,T)
    
    error = np.std(np.abs(cs(V) - T))   #make note of the average error

    #plot both the original data and the interpolated data    
    niceFigure(True)
    plt.plot(V,T, '.', label = 'Lakeshore data')
    plt.plot(V,cs(V), label = 'Interpolated data')
    plt.xlabel("$V$")
    plt.ylabel("$T$")
    plt.legend()
    print("The estimated error on the interpolation is", error)
    plt.savefig('lakeshore_interp_data.png', format = 'png', dpi = 500, bbox_inches = 'tight')


lakeshore_T_V("lakeshore.csv")




#%%

"QUESTION 4"

n, m = 3, 3  #rational function numerator and denominator degree
#range of x values to do our interpolating over
x1cos = np.linspace(-np.pi/2, np.pi/2, n + m - 1)
x1lor = np.linspace(-1, 1, n + m - 1)
x2cos = np.linspace(-np.pi/2, np.pi/2, 1001)
x2lor = np.linspace(-1, 1, 1001)

#cosine and lorentzian function definitions
cos = np.cos
def lor(x):
    return 1/(1 + x**2)

y1=cos(x1cos)
y2=lor(x1lor)


"The following two function def's are taken from Prof. Siever's repo"

#create the rational function interpolation routine
def rat_eval(p,q,x):
    #create the general rational function, given an array of numerator (p) and denominator (q) coefficients
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i      #sets the numerator polynomial
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)   #sets the denominator polynomial
    return top/bot              #return the rational funciton

def rat_fit(x,y,n,m):
    #find the numerator and denominator coefficients p & q, given a set of x, and y data
    assert(len(x)==n+m-1)       #ensures the length of x and y arrays matches the number of free variables we need to solve for
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])   #empty matrix 
    
    #now populate the matrix with our x^n values 
    for i in range(n):         
        mat[:,i]=x**i   #first n rows and all the cols
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i  #next m rows and all the cols
    # print(mat)
    pars=np.linalg.pinv(mat)@y    #compute the matrix inverse times the y-values to generate the values of p and q
    
    #collect the p and q values from the larger 'pars' matrix
    p=pars[:n]
    q=pars[n:]
    # print(p)   #print these to compare if necessary
    # print(q)
    return p,q



"Cosine interpolation calculations"
y1_true_interpts=cos(x2cos)         #theoretical cos function vals (evaluated at 1001 pts)
p,q=rat_fit(x1cos,y1,n,m)   #retrieve p and q values from our rational fit

#now do the rational funciton interpolation
y1_rat_interp=rat_eval(p,q,x2cos)   

#use scipy polynomial interpolator
y1_poly_interp = BarycentricInterpolator(x1cos,y1)

#generate the fit using scipy's cubic spline
y1_cs = CubicSpline(x1cos, y1)

#check the accuracy of our interp methods
print("When interpolating the cos function,")
print("Error on rational interp=", np.std(y1_rat_interp - y1_true_interpts))
print("Error on polynomial interp=", np.std(y1_poly_interp(x2cos) - y1_true_interpts))
print("Error on cubic spline interp=", np.std(y1_cs(x2cos) - y1_true_interpts))




"Lorentzian interpolation calc's"
del p, q
y2_true_interpts=lor(x2lor)
p,q=rat_fit(x1lor,y2,n,m)

#now do the rational funciton interpolation
y2_rat_interp=rat_eval(p,q,x2lor)   

#use scipy polynomial interpolator
y2_poly_interp = BarycentricInterpolator(x1lor,y2)

#generate the fit using scipy's cubic spline
y2_cs = CubicSpline(x1lor, y2)

#check the accuracy of our interp methods
print("")
print("When interpolating the Lorentzian,")
print("Error on rational interp=", np.std(y2_rat_interp - y2_true_interpts))
print("Error on polynomial interp=", np.std(y2_poly_interp(x2lor) - y2_true_interpts))
print("Error on cubic spline interp=", np.std(y2_cs(x2lor) - y2_true_interpts))




"Graphics"
#example plot of interpolation pts and rational function interpolation
niceFigure(True)
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (6,8))
fig.tight_layout(pad = 3)
ax1.set_title("$cos(x)$")
ax1.plot(x1cos, cos(x1cos), 'o')
ax1.plot(x2cos, y1_rat_interp)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax2.set_title("$Lorentzian(x)$")
ax2.plot(x1lor, lor(x1lor), 'o')
ax2.plot(x2lor, y2_rat_interp)
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")
plt.savefig("question4plot.png", format = 'png', dpi = 500, bbox_inches = 'tight')













