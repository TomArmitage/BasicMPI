# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI # module required to use MPI
from time import time

from MPIFuncs import *

# Set up MPI environment
comm=MPI.COMM_WORLD # Communication module
NProcs=comm.Get_size() # Number of processors available
MyRank=comm.Get_rank() # The rank of the process


####################################################################
#
# INSTRUCTIONS FOR USE - 
# enter: mpirun -n 4 python ExampleCalls.py
# in the terminal to start running this script on 4 threads
####################################################################


# Select which function to use
d={
    '1':'Hello',
    '2':'Timing',
    '3':'commune',
    '4':'coresum',
    '5':'varcom',
    '6':'trap'
}

if MyRank==0:
    print 'Please enter a number to run the corresponding function \n'
    for i in np.arange(1,len(d.keys())+1):
        i=str(i)
        print i+' : '+d[i]+'\n'
    funcToRun=np.zeros(1,dtype='i')+int(raw_input('Choose value: '))
else:
    funcToRun=np.ones(1,dtype='i')*20

comm.Barrier()

funcToRun=comm.bcast(funcToRun,root=0)

funcToRun=str(funcToRun[0])


if d[funcToRun]=='Hello':
    Hello()

elif d[funcToRun]=='Timing':
    if MyRank==0: print 'This function generates an array of 10^6 floats on core 0 which are then sent to core 1 using both the Uppercase and Lowercase methods, the time to complete in each case is shown'

    UpVLow(MyRank=MyRank,comm=comm,NProcs=NProcs)

elif d[funcToRun]=='coresum':
    if MyRank==0: print 'This function generates an array of length 10 for each core and then sums up the elements together'
    data=np.arange(10)
    print 'MyRank: '+str(MyRank)+' Data array: '+str(data)
    ans=coresum(data)
    if MyRank==0: print 'The summed array: ',ans

elif d[funcToRun]=='commune':
    if MyRank==0: print 'This function combines arrays of len 10 together from each core going from 0-9 and each array is multiplied by the core number + 1'
    data=np.arange(10)*(MyRank+1)
    print 'MyRank: '+str(MyRank)+' Data array: '+str(data)
    comm.Barrier()
    ans=commune(data)
    if MyRank==0: print 'Combined array: ', ans

elif d[funcToRun]=='varcom':
    if MyRank==0: print 'This function generates a gaussian distribution drawing from a distribution with varience 1 for each core. The varience for the varience for the whole sample is calculated and sent to core 0'
    data=np.random.normal(size=int(1e4*(MyRank+1)))
    length=np.array([len(data)])
    var=np.array([np.var(data)])
    mean=np.array([np.mean(data)])
    ans=varcom(length,var,mean)

    if MyRank!=0:
        print str(MyRank)+' var: '+str(var)
    else:
        print str(MyRank)+' var: '+str(var)+' varComb: '+str(ans)

elif d[funcToRun]=='trap':
    if MyRank==0: print 'This function calculates the integral x^2 between 0 and 100 using the trapezium rule. It is impllimented using peer to peer and collective communication as well as serial for reference \n'
    a=0
    b=100
    n=10000
    if MyRank==0: trapezoidSerial(a,b,n)
    comm.Barrier()
    trapezoidP2P(a,b,n)
    comm.Barrier()
    trapezoidCollective(a,b,n)

else:
    if MyRank==0:
        print 'Sorry no correct strings selected'