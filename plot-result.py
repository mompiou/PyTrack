import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.ticker as ticker  

def meanaveraging(x, N):
	return np.convolve(x, np.ones((N,))/N,'valid')

def reject_outliers(data, e, w):										#remove bad points based on the deviation from mean average in a range w

	for i in range(w,np.shape(data)[0]-w):
		mean=np.mean(data[i-w:i+w,0])
		std=np.std(data[i-w:i+w,0])
		
		if np.abs(data[i,0]-mean) > e * std :
			data[i,0]=data[i-1,0]
			
	return data

plt.rcParams['font.sans-serif']="DejaVu Sans"
f=plt.figure(1,figsize=(10,7.5),dpi=100)
ax1 = plt.subplot()

t0=np.genfromtxt('results.txt', delimiter='\t') 						#Read the results from the file
t0=reject_outliers(t0, 2,5)

g=50																	#averaging number


s0=(np.abs(t0[g:,0] - t0[:-g,0]))*1200e-9*0.8/616/(g*0.04)				#compute the average speed at every points
s1=np.diff(np.abs(t0[1::g,0]))*1200e-9*0.8/616/(g*0.04)  				#compute the average speed between points separated by g frames
s2=np.diff(meanaveraging(np.abs(t0[:,0]),g))*1200e-9*0.8/616/0.04		#calculate the running image average according a sliding window of width sm 
s=[s0,s1,s2]
t=[t0[g::1,1],t0[g::g,1],t0[g-1:-1,1]]										#time
print s2.shape, t[2].shape
i=1
A=np.vstack((t[i],s[i]))

Aa=A[:,A[0,:]>6]														#select the data range for fitting
Ab=Aa[:,Aa[0,:]<30]

Aa1=A[:,A[0,:]>6]														#select the data range for fitting
Ab1=Aa[:,Aa[0,:]<15]

def func(x, a,b):														#fitting function
    return a/(x+b)

popt, pcov = curve_fit(func, Ab[0,:], Ab[1,:])							#get the fitting parameters
popt1, pcov1 = curve_fit(func, Ab1[0,:], Ab1[1,:])		
print(popt, popt1)
#a=ax1.plot(t0[:,1], t0[:,0], label=r'Displacement vs time')	

a=ax1.plot(t[i], s[i]*1e9, label=r'exp.')				#plot
b=ax1.plot(t[i], func(t[i],popt[0]*1e9,popt[1]),label='fit')
c=ax1.plot(t[i][1:9], func(t[i][1:9],popt1[0]*1e9,popt1[1]),label=' ')
plt.setp(b,color='#4169E1',linewidth=2,alpha=1,ms=10,mew=0.1,ls='-',markevery=1)
plt.setp(c,color='#4169E1',linewidth=2,alpha=1,ms=10,mew=0.1,ls='--',markevery=1)
plt.setp(a,color='#FF4136',linewidth=0,alpha=1,marker='o',ms=10,mew=0.1,markevery=1)
ax1.set_xlabel(r't (s)', fontsize=16)
ax1.set_ylabel(r'v (nm/s)',fontsize=16)
shift = 0                                                                                                                                                                                                                                                              
ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x+shift))                                                                                                                                                                                                           
ax1.xaxis.set_major_formatter(ticks) 
ax1.axis([0,40,0,15])
plt.tick_params(labelsize=12)
ax1.axhline(linewidth=0.5)   
plt.legend(loc='upper right', frameon=False, numpoints=1)
plt.show()

