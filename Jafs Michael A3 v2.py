# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:15:34 2022

@author: mjafs

Assignment 3: SDLF to solve the shrodinger equation. 

The user should begin by running the first cell below which contains module imports and function definitions 
that are used throughout the code. The user can then either sequentially step through each of the 3 question cells
or choose to run one question at a time. This has been automated so that there only includes one cell per
question, with graphics that close on their own, so the user only needs to press command+enter once. Question 1 
provides the user with the ability to choose a 'run method' (see the def for keyword_function_mapper) but 
defaults to matrix slicing as a derivative construction method. The user will also find an option within each 
cell to modify the grid size, and the number of timesteps/periods (question dependent) that the function uses 
to solve the system of equations. 

"""


#%%
import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.sparse as sparse
from IPython import get_ipython

"Good figure template example"
# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 40})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return




"Leap frog algorithm"
def leapfrog(diffeq, r0 , v0 , t, dt, V, x, h): # vectorized leapfrog
    r1 = r0 + (dt/2)*diffeq(0, r0, v0, t, V, x, h) # 1: r1 at h/2 using v0
    v1 = v0 + dt*diffeq(1, r1, v0, t, V, x, h) # 2: v1 using a(r) at h/2
    r1 = r1 + (dt/2)*diffeq(0, r0, v1, t, V, x, h) # 3: r1 at h using v1
    return r1, v1




"MAIN FUNCTION DEFINITIONS"

def wavepacket(sig, x0, k0, x):  #gaussian wave packet, o is sigma
    C = (sig*np.sqrt(np.pi))**(-1/2)  #constant out in front
    exp = np.exp(-(x - x0)**2/(2*sig**2))   #first piece in the exponent
    
    R = C*exp*np.cos(k0*x)    #real part of wavepacket
    I = C*exp*np.sin(k0*x)    #imaginary part 
    
    return R, I

"Derivatives that use matrix slicing"
def slice_derivs(id, R, I, t, V, x, h):
    
    b = 1/(2*h**2)
    if (id == 0):
        var1 = -b*I
        drdt = (2*b +  V(x))*I
        drdt[1:-1] += var1[:-2] + var1[2:]
        
        "Add periodic boundary conditions"
        drdt[0] += var1[1] + var1[-1]
        drdt[-1] += var1[-2] + var1[0]
        return drdt
    
    if (id == 1):
        var2 = b*R
        didt = -(2*b + V(x))*R
        didt[1:-1] += var2[:-2] + var2[2:]
        
        "Periodic BC's"
        didt[0] += var2[1] + var2[-1]
        didt[-1] += var2[-2] + var2[0]
        # didt[-0] = var2[-1]
        return didt
    
    
"Derivatives that use a sparse matrix in the ODE's"
def sparse_derivs(id, R, I, t, V, x, h):
    b = 1/(2*h**2)
    Ns = len(I)  #size of original N
    
    #define values for matrix 
    k = [-b*np.ones(Ns-1),(2*b + V(x))*np.ones(Ns),-b*np.ones(Ns-1), -b, -b]
    offset = [-1,0,1, -(Ns-1),Ns-1]   #offsets and periodic boundary conditions
    A = sparse.diags(k, offset)   #convert to sparse 

    #define the ode's in terms of matrix multiplication
    if (id == 0):   
        drdt = A@I
        return drdt
    else:
        didt = -A@R
        return didt
    
    
"Derivatives that leave A as a full matrix"
def full_derivs(id, R, I, t, V, x, h):
    b = 1/(2*h**2)
    Ns = len(I)  #size of original N
    
    #define values for matrix 
    k = [-b*np.ones(Ns-1),(2*b + V(x))*np.ones(Ns),-b*np.ones(Ns-1), -b, -b]
    offset = [-1,0,1, -(Ns-1),Ns-1]   #offsets and periodic boundary conditions
    A = sparse.diags(k, offset).toarray()   #leave as full matrix 
    
    #calculate the derivatives
    if (id == 0):   
        drdt = A@(I)
        return drdt
    else:
        didt = -A@(R)
        return didt

def initialcond(xmin, xmax, w0, m, N):
    """
    Function which returns the array of space values, the spacial step size,
    and the real and imaginary parts of psi respectively.
    Takes xmin: minimum value for the grid points, xmax: the maximum value in atomic units, w0: the 
    initial frequency, m: the mass if we need to include it somewhere, and N: the size of the spacial 
    grid.
    
    """
    x = np.linspace(xmin, xmax, N)
    R, I = wavepacket(sig, x0, k0, x)
    h = x[1] - x[0]
    return x, h, R, I

def keyword_function_mapper(choice):
    """
    Parameters
    ----------
    choice : Takes a string defining which derivative function leapfrog should call.
    Three options are available:
        'slice' uses the slicing approach when defining the matrix A
        'sparse' defines A as a sparse matrix
        'full' uses the derivatives that takes A as a full matrix 
    Returns
    -------
    slice_derivs, sparse_derivs, or full_derivs respectively
    """
    if (choice == 'slice'):
        return slice_derivs
    elif (choice == 'sparse'):
        return sparse_derivs
    elif (choice == 'full'):
        return full_derivs
    
    
def clear_variables(tf):
    """
    Simple command which clears memory of specified relevant initial condition variables. Argument 
    can take True or False which either does or doesn't clear memory of specified variables.
    """
    if tf == True:
        try:
            global x0, sig, k0, x1, x2, w0, m, xarr, hs, R0, I0, dt, tlist, npts, N, tstep, pd 
            del x0, sig, k0, x1, x2, w0, m, xarr, hs, R0, I0, dt, tlist, npts, N, tstep, pd 
        except NameError:
            pass
    elif tf == False:
        pass

def clean_ALL(tf):
    """
    Simple command which clears memory of all earlier variables (including modules and function definitions). Argument 
    can take True or False which either does or doesn't clear memory of variables.
    """
    if tf == True:
        get_ipython().magic('reset -sf')
    elif tf == False:
        pass 
    
#%%
"QUESTION 1"
"The animation here defaults to a grid size of 1000 and 15000 timesteps (this can be changed below)"


"begin by making sure all previous initial condition variables are cleared."
plt.close('all') #and close previous plots
clear_variables(True)




"Start with a run time comparison between matrix construction approaches"

"Potential for question 1"
def V1(x):   #simple free space potential
    return x*0




run_method = keyword_function_mapper('slice')  #users choice of run method


"Begin setting up the problem"
#Modify the number of grid points or timesteps"
N = 1000   #number of grid points
tstep = 15000 #number of timesteps

#Setup the initial conditions
x0, sig, k0 = -5, 0.5, 5 #wavepacket parameters: center of gaussian, width, init avg momentum
x1, x2 = -10, 10    #min and max values of grid in units of au
w0, m = 1, 1        #initial frequency, mass
xarr, hs, R0, I0 = initialcond(x1, x2, w0, m, N)   #retrieve the initial conditions

#setup the time spacing and number of points in time array
dt = 0.5*hs*hs   #timestep definition
tlist = np.arange(0,tstep*dt, dt)   
npts = len(tlist) #number of points in the time array

#prepare to store the full solution in arrays
Rarr, Iarr = np.zeros((npts,N)), np.zeros((npts,N)) 
R, I = R0, I0 
Rarr[0,:], Iarr[0,:] = R, I   #assign initial conditions to arrays to append to later
# plt.plot(xarr, R0)


"Check time to solve"
start = timeit.default_timer()
for i in range(0,npts):
    R, I = leapfrog(run_method, R, I, tlist[i], dt, V1, xarr, hs) # solve the matrix ode equation
    Rarr[i,:], Iarr[i,:] = R, I
stop = timeit.default_timer()
ttot = stop - start   #collect the total time for the solver
print("The time to solve the problem using the users choice of method is:",ttot, "seconds")




"Now use animation to check for periodic boundary conditions and see wavepacket behaviour in time"
#delete memory of R0 and I0 after building the list in the for loop above
del R0, I0

#retrieve initial R and I again
xarr, hs, R0, I0 = initialcond(x1, x2, w0, m, N)   #retrieve the initial conditions


def go(cycle):
    """
    Simple animation function which displays wavepacket behaviour. Note that the users
    choice of run method is decided above and the animation will automatically use 
    which ever method was called at that time.
    
    Parameters
    ----------
    cycle : The number of cycles per update
    """
    
    ic=0
    niceFigure(True)   #plot using Latex
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_axes([0.2, 0.2, .7, .7])
    ax.grid('on')
    ax.set_xlabel('$x (a.u.)$', fontsize = 50)     # add labels
    ax.set_ylabel('$|\psi(x,t)|^2$', fontsize = 50)
    ax.set_xticks([-10,-5,0,5,10])
    
    #prepare to animate psi
    R, I = R0, I0   #set values to send to leapfrog algorithm in animation
    prob = abs(R0**2) + abs(I0**2)  #compute probability density of psi
    line, = ax.plot( xarr, prob,linewidth = 2) # Fetch the line object
    t=0.   #start the animation at time t = 0
    x = 0      # update the position from our intitial x value
    tfin = tstep*dt  #final time (in units of the timestep dt)
    tpause = 0.01 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    #retrieve function mapper call as the run method
    # run_method = keyword_function_mapper(run_choice)

    while t<tfin:        
        
        #solve within loop for animation
        R, I = leapfrog(run_method, R, I, t, dt, V1, xarr, hs) # solve using leapfrog
        prob1 = abs(R**2) + abs(I**2)   

        if (ic % cycle == 0): # very simple animate (update data with pause)

            ax.set_title("frame time {}".format(ic)) # show current time on graph
            line.set_ydata(prob1)  #plot psi as a function of x and t
            line.set_xdata(xarr)
            
            #to save images at specified frames for each of the run methods
            if (run_method == slice_derivs):
                
                if (ic == 1500):
                    plt.savefig('Q1timestep_slice=1500.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
                if (ic == 4500):
                    plt.savefig('Q1timestep_slice=4500.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
                if (ic == 10000):
                    plt.savefig('Q1timestep_slice=10000.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
                if (ic == 15000):
                    plt.savefig('Q1timestep_slice=15000.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
            elif (run_method == sparse_derivs):
                
                if (ic == 1500):
                    plt.savefig('Q1timestep_sparse=1500.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
                if (ic == 4500):
                    plt.savefig('Q1timestep_sparse=4500.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
                if (ic == 10000):
                    plt.savefig('Q1timestep_sparse=10000.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
                if (ic == 15000):
                    plt.savefig('Q1timestep_sparse=15000.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
            elif (run_method == full_derivs):
                
                if (ic == 1500):
                    plt.savefig('Q1timestep_full=1500.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
                if (ic == 4500):
                    plt.savefig('Q1timestep_full=4500.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
                if (ic == 10000):
                    plt.savefig('Q1timestep_full=10000.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
                if (ic == 15000):
                    plt.savefig('Q1timestep_full=15000.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
            plt.draw() # may not be needed (depends on your set up)
            plt.pause(tpause) # pause to see animation as code v. fast
               
        t  = t + dt # loop time
        x = x + hs      #move forward in space by steps of hs
        ic = ic + 1 # count the number of cycles

go(100)
plt.close('all')   #close once the animation is done

#%%
"QUESTION 2"

"""
The fastest method for implimenting the derivatives seemed to be when the slicing techniques
were used. The following will then use exclusively array slicing as the method of choice
for the derivative function.
"""

"begin by closing any open plots and clearing old variable assignments"
plt.close('all')
clear_variables(True)




#harmonic oscillator potential (new to question 2)
def V2(x):   
    return 0.5*x**2

"Set up the problem again"
#we have a slight change to the wavepacket parameters for this question
x0, sig, k0 = -5, 0.5, 0 #center of gaussian, width, init avg momentum

#again, define the grid shape and intitialize the setup
N = 1000   #number of grid points
x1, x2 = -10, 10    #min and max values of grid in units of au
w0, m = 1, 1        #initial fequency and mass
xarr, hs, R0, I0 = initialcond(x1, x2, w0, m, N)   #retrieve the initial conditions
dt = 0.5*hs*hs   #timestep





"Simulate wavepacket dynamics for the harmonic oscillator potential"
def go(cycle, nT):
    """
    Simple animation function which displays wavepacket behaviour. Note that this function
    does not contain a choice for which derivative method to use since we choose to default 
    to slicing techniques
    
    Parameters
    ----------
    cycle : The number of cycles per update of the animation
    nT : The total number of periods for the simulation (so, T/dt = timestep. Where T = period
    and timestep = the number of timesteps for the simulation). We have set w0 = 1, so one 
    period equals 2pi. Or timestep = 2pi/dt.
    """
    
    ic=0
    niceFigure(True)   #plot using Latex
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_axes([0.2, 0.2, .7, .7])
    ax.grid('on')
    ax.set_xlabel('$x (a.u.)$', fontsize = 50)     # add labels
    ax.set_ylabel('$|\psi(x,t)|^2$', fontsize = 50)
    ax.set_xticks([-10,-5,0,5,10])
    
    
    #prepare to simulate the full solution
    R, I = R0, I0   #set values to send to leapfrog algorithm in animation
    prob = abs(R0**2) + abs(I0**2)  #compute initial probability density of psi
    line, = ax.plot( xarr, prob,linewidth = 2) # Fetch the line object
    
    #plot the potential in the background normalized to 1
    ax.plot(xarr, V2(xarr)/max(V2(xarr)))
    
    #time and animation parameters
    t=0. #start the animation at time t = 0
    x = 0   # update the position from our intitial x value
    tstep = nT*2*np.pi/dt   #calculate the number of time steps from the total desired periods
    tfin = tstep*dt  #final time (in units of the timestep)
    tpause = 0.0001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    pd = []  #empty list for probability density values 
    
    
    while t<tfin:        
        
        #solve within loop for animation
        R, I = leapfrog(slice_derivs, R, I, t, dt, V2, xarr, hs) # solve using leapfrog
        prob1 = abs(R**2) + abs(I**2)   
        pd.append(prob1)    #append probability density values 


        if (ic % cycle == 0): # very simple animate (update data with pause)

            ax.set_title("frame time {}".format(ic)) # show current time on graph
            
            #plot psi as a function of x and t
            line.set_ydata(prob1)  
            line.set_xdata(xarr)
            
            #to save images at specified frames 
            if (ic == 200):
                plt.savefig('Q2snapshot1.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
            if (ic == 10000):
                plt.savefig('Q2snapshot2.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
            if (ic == 16000):
                plt.savefig('Q2snapshot3.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
            if (ic == 30000):
                plt.savefig('Q2snapshot4.jpg', format='jpg', dpi=1200,bbox_inches = 'tight')
            plt.draw() # may not be needed (depends on your set up)
            plt.pause(tpause) # pause to see animation as code v. fast
               
        t  = t + dt # loop time
        x = x + hs  #move forward in space by steps of hs
        ic = ic + 1 #count the number of cycles 
        
    return tstep, dt, pd #return these values for use in contour plot later

tstep, dt, pd = go(200, 2)

plt.close('all') #close animation once it is finished





"Plot the probability density distribution"
#create time array, with length and stepsize equal to that of the animation previous
tlist = np.arange(0, tstep*dt, dt)

#create the contour plot
fig = plt.figure(figsize = (10,5))
niceFigure(True)
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
ax.contourf(tlist, xarr, np.transpose(pd), cmap = 'plasma')
# ax.set_title("Probability Distribution")
ax.set_xlabel("$t \ (T_0)$", fontsize = 50)
ax.set_ylabel("$x$ (a.u.)", fontsize = 50)
ax.set_yticks([-10,-5,0,5,10])
plt.savefig( "Q2density.jpg", format = 'jpg', dpi = 1200, bbox_inches = 'tight')
plt.pause(10)    #wait for 10 seconds before closing the density plot
plt.close('all')






"Solve again to create arrays for virial thm calculation"
# del R0, I0, R, I  #reset initial arrays for psi
# xarr, hs, R0, I0 = initialcond(x1, x2, w0, m, N)   #retrieve the initial conditions

#prepare to store the full solution in arrays
npts = len(tlist)
R, I = R0, I0  
Rarr, Iarr = np.zeros((npts,N)), np.zeros((npts,N))  
Rarr[0,:], Iarr[0,:] = R, I   #assign initial conditions to arrays to append to later


#solve using slicing method
start = timeit.default_timer()
for i in range(0,npts):
    R, I = leapfrog(slice_derivs, R, I, tlist[i], dt, V2, xarr, hs) # solve the matrix ode equation
    Rarr[i,:], Iarr[i,:] = R, I
stop = timeit.default_timer()
print("The time to solve is:", stop - start)


"Check that the virial theorem is satisfied"
psi = Rarr + Iarr #construct the full wavefunction solution 

#find 1/2<T> 
#psi is a 2D array with rows=time points and cols=space points
gradpsi = np.gradient(psi, hs, axis = 1)  #take the gradient of psi along each row
tavgpsi_grad = np.average(abs(gradpsi**2), axis = 0)  #take the time average (therefore along each column)

#calculate the expectation of the kinetic energy
Tavg = np.trapz(0.5*tavgpsi_grad, xarr) #integrate the norm of the gradient squared over all space
print("The value for the average kinetic energy multiplied by 1/2 is:",Tavg, "in arbitrary units")

#now <V>
psinorm = abs(psi**2)    #(psi)(psi*)
tavg_psi = np.average(psinorm, axis = 0)  #timeaverage
Vavg = np.trapz(tavg_psi*V2(xarr), xarr)   #integrate over all space points
print("The value of average potential is:", Vavg, "in arbitrary units")


#print confirmation for virial thm
def virialthm_check(T, V):
    """
    Check for percent that the viral theorem is satisfied. Parameters are, T: user must input 1/2<T>, 
    V: user must input <V> 
    """
    diff = abs(T - V)
    print("The virial theorem is satisfied to", diff/((T+V)/2)*100, "%")
        
virialthm_check(Tavg, Vavg) #print the percentage to which the virial thm is satisfied
    
#%%
"QUESTION 3"

"""
The following will again use exclusively array slicing as the method of choice
for the derivative function.
"""

"begin by closing all plots and clearing old variable assignments"
plt.close('all')
clear_variables(True)






#question 3 potential (default to a = 1, b = 4 as per the question)
a, b = 1, 4    #can vary these to vary the depth of the 'wells' in the double well potential
def V3(x):
    V = a*x**4 - b*x**2
    return V



"Set up the problem once again"
#new wavepacket parameters for this question
x0, sig, k0 = -np.sqrt(2), 0.5, 0 #center of gaussian, width, init avg momentum

#again, define the grid shape and intitialize the setup (these change from question 2)
N = 500   #number of grid points
x1, x2 = -5, 5    #min and max values of grid in units of au
w0, m = 1, 1        #initial fequency and mass
xarr, hs, R0, I0 = initialcond(x1, x2, w0, m, N)   #retrieve the initial conditions
dt = 0.5*hs*hs   #timestep




"Simulate wavepacket dynamics for the potential which leads to quantum tunneling"
def go(cycle, nT):
    """
    Simple animation function which displays wavepacket behaviour. Note that this function
    does not contain a choice for which derivative method to use since we choose to default 
    to slicing techniques
    
    Parameters
    ----------
    cycle : The number of cycles per update of the animation
    nT : The total number of periods for the simulation (so, T/dt = timestep. Where T = period
    and timestep = the number of timesteps for the simulation). We have set w0 = 1, so one 
    period equals 2pi. Or timestep = 2pi/dt.
    """
    
    ic=0
    niceFigure(True)   #plot using Latex
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_axes([0.2, 0.2, .65, .65])
    ax.grid('on')
    ax.set_xlabel('$x (a.u.)$', fontsize = 50)     # add labels
    ax.set_ylabel('$|\psi(x,t)|^2$', fontsize = 50)
    ax.set_ylim(-0.6, 1.4)
    ax.set_xlim(-3.5,3.5)
    ax.set_yticks([-0.4, 0.0, 0.4, 0.8, 1.2])
    #plot the potential in the background 
    ax.plot(xarr, V3(xarr)/10)
    
    #time and animation parameters
    t=0. #start the animation at time t = 0
    x = 0   # update the position from our intitial x value
    tstep = nT*2*np.pi/dt   #calculate the number of time steps from the total desired periods
    tfin = tstep*dt  #final time (in units of the timestep)
    tpause = 0.001 # delay within animation 
    plt.pause(2) # pause for 2 seconds before start
    
    #prepare to simulate the full solution
    R, I = R0, I0   #set values to send to leapfrog algorithm in animation
    prob = abs(R0**2) + abs(I0**2)  #compute initial probability density of psi
    line, = ax.plot( xarr, prob,linewidth = 2) # Fetch the line object
    
    
    pd = []    #empty list for probability density values 
    
    
    while t<tfin:        
        
        #solve within loop for animation
        R, I = leapfrog(slice_derivs, R, I, t, dt, V3, xarr, hs) # solve using leapfrog
        prob1 = abs(R**2) + abs(I**2)   
        pd.append(prob1)    #append probability density values 


        if (ic % cycle == 0): # very simple animate (update data with pause)

            ax.set_title("frame time {}".format(ic)) # show current time on graph
            
            #plot psi as a function of x and t
            line.set_ydata(prob1)  
            line.set_xdata(xarr)
            
            #to save images at specified frames 
            if (ic == 1000):
                plt.savefig('Q3snap1,a={},b={}.jpg'.format(a,b), format='jpg', dpi=1200,bbox_inches = 'tight')
            if (ic == 20000):
                plt.savefig('Q3snap2,a={},b={}.jpg'.format(a,b), format='jpg', dpi=1200,bbox_inches = 'tight')
            if (ic == 60000):
                plt.savefig('Q3snap3,a={},b={}.jpg'.format(a,b), format='jpg', dpi=1200,bbox_inches = 'tight')
            if (ic == 100000):
                plt.savefig('Q3snap4,a={},b={}.jpg'.format(a,b), format='jpg', dpi=1200,bbox_inches = 'tight')
            plt.draw() # may not be needed (depends on your set up)
            plt.pause(tpause) # pause to see animation as code v. fast
               
        t  = t + dt # loop time
        x = x + hs  #move forward in space by steps of hs
        ic = ic + 1 #count the number of cycles 
        
    return tstep, dt, pd #return these values for use in contour plot later

tstep, dt, pd = go(500, 4)

plt.close('all')  #close the animation once it's done





"Plot the probability density distribution"


"Solve again to create arrays for expectation of position - use these arrays to plot the density as well"
del R0, I0  #reset initial arrays for psi
xarr, hs, R0, I0 = initialcond(x1, x2, w0, m, N)   #retrieve the initial conditions

#prepare to store the full solution in arrays
tlist = np.arange(0, tstep*dt, dt)
npts = len(tlist)
R, I = R0, I0
Rarr, Iarr = np.zeros((npts,N)), np.zeros((npts,N))  
Rarr[0,:], Iarr[0,:] = R, I   #assign initial conditions to arrays to append to later


#solve using slicing method
start = timeit.default_timer()
for i in range(0,npts):
    R, I = leapfrog(slice_derivs, R, I, tlist[i], dt, V3, xarr, hs) # solve the matrix ode equation
    Rarr[i,:], Iarr[i,:] = R, I
stop = timeit.default_timer()
print("The time to solve is:", stop - start)
# plt.plot(xarr,Rarr[1000,:])



#calculate expectation of x
psinorm = Rarr**2 + Iarr**2    #(psi)(psi*)
pos = xarr*psinorm           #integrand for <x>
posAvg = np.trapz(pos, xarr)   #calculate the avg over dx




#create the contour plot
fig = plt.figure(figsize = (10,5))
niceFigure(True)
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
ax.plot(tlist, posAvg, linewidth = 6, label = "$<x>$")
#use the arrays from the 'for' loop to plot the density
ax.contourf(tlist, xarr, np.transpose(Rarr**2 + Iarr**2), cmap = 'plasma') 
# ax.set_title("Probability Distribution")
ax.set_xlabel("$t \ (T_0)$",fontsize = 50)
ax.set_ylabel("$x$ (a.u.)", fontsize = 50)
ax.set_yticks([-5,-2.5,0,2.5,5])
ax.legend(loc = 'best')
plt.savefig( "Q3densitya={}b={}.jpg".format(a,b), format = 'jpg', dpi = 1200, bbox_inches = 'tight')
plt.pause(10)    #wait for 10 seconds before closing the density plot
plt.close('all')


