# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI # module required to use MPI
from time import time


# Set up MPI environment
comm=MPI.COMM_WORLD # Communication module
NProcs=comm.Get_size() # Number of processors available
MyRank=comm.Get_rank() # The rank of the process

def Hello(MyRank=MyRank,comm=comm):

    print 'Thread ',MyRank, ' says Hello World!'

    return

def UpVsmall(MyRank=MyRank,comm=comm,NProcs=NProcs):

    ArraySize=int(1e8)
    
    NumPerProc=ArraySize/NProcs
    Remain=ArraySize%NProcs

    if MyRank-Remain<0:
        Additional=1
    else:
        Additional=0

    LocArray=np.ones(NumPerProc+Additional)

    LocSum=np.sum(LocArray,dtype=np.int64)

    time0=time()
    Tot=np.zeros(1,dtype=np.int64)

    #print str(MyRank)+' '+str(LocSum)

    comm.Reduce([LocSum,MPI.__TypeDict__[LocSum.dtype.char]],[Tot,MPI.__TypeDict__[Tot.dtype.char]],op=MPI.SUM,root=0)

    time1=time()
    #if MyRank==0: print Tot,time1-time0

    Tot=np.zeros(1,dtype=np.int64)
    comm.reduce(LocSum,Tot,op=MPI.SUM,root=0)

    time2=time()
    #if MyRank==0: print Tot,time2-time1

    time3=time()
    Tot=np.zeros(1,dtype=np.int64)
    dump=np.sum(Tot)
    time4=time()

    if MyRank==0:
        print 'Serial ',time4-time3
        print 'Lower  ',time2-time1
        print 'Upper  ',time1-time0

def coresum(data,MyRank=MyRank,comm=comm,NProcs=NProcs):
    # Adds together all data in a array cell by cell and sends to all cores
    # source: D Barnes
    try:
        rslt=np.zeros(np.shape(data),dtype=data.dtype)
    except:
        rslt=np.zeros(len(data),dtype=data.dtype)
    comm.Allreduce([data,MPI.__TypeDict__[data.dtype.char]],[rslt,MPI.__TypeDict__[rslt.dtype.char]],op=MPI.SUM)
    del data
    return rslt
 
def commune(data,MyRank=MyRank,comm=comm,NProcs=NProcs):
    #Combines together arrays on different processes into one on all
    # Source D Barnes

    tmp=np.zeros(NProcs,dtype=np.int)
    tmp[MyRank]=len(data)
    cnts=np.zeros(NProcs,dtype=np.int)
    comm.Allreduce([tmp,MPI.INT],[cnts,MPI.INT],op=MPI.SUM)
    del tmp
    dspl=np.zeros(NProcs,dtype=np.int)
    i=0
    for j in xrange(0,NProcs,1):
        dspl[j]=i
        i+=cnts[j]
    data=data
    rslt=np.zeros(i,dtype=data.dtype)
    comm.Allgatherv([data,cnts[MyRank]],[rslt,cnts,dspl,MPI.__TypeDict__[data.dtype.char]])
    del data,cnts,dspl
    return rslt

def varcom(length,var,mean,MyRank=MyRank,comm=comm,NProcs=NProcs):
    #ONLY RETURNS THE VALUE ON CORE 0, OTHERS RETURN 1
    #VAR PART
    #fterm   
    lenv=length*var
    vtop=coresum(lenv)
    vbot=coresum(length)
    fterm=vtop/vbot    
    
    #sterm
    sbot=vbot**2
    tmp={}
    if MyRank>0: comm.Send(length,dest=0,tag=0); comm.Send(mean,dest=0,tag=1); return 1

    if MyRank==0:
        i='dummy'
        for j in range(1,NProcs):
            tmp['N'+str(j)]=np.ones(1)
            try:
                tmp[i+str(j)]=np.ones(len(mean))
            except:
                tmp[i+str(j)]=np.ones(1)
            comm.Recv(tmp['N'+str(j)],source=j,tag=0)
            comm.Recv(tmp[i+str(j)],source=j,tag=1)

        stop=0
        tmp['N0']=length
        tmp[i+str(0)]=mean
        for j in range(NProcs):
            for k in range(j):
                stop=stop+tmp['N'+str(j)]*tmp['N'+str(k)]*((tmp[i+str(j)]-tmp[i+str(k)])**2)

        sterm=stop/sbot
        vartot=fterm+sterm

        return vartot

def trapezoidCollective(a,b,n,MyRank=MyRank,comm=comm,NProcs=NProcs):
    #Uses collective communication to calculate the integral x^2
    # Source http://materials.jeremybejarano.com/MPIwithPython/collectiveCom.html
    #takes in command-line arguments [a,b,n]
    time0=time()
    a = float(a)
    b = float(b)
    n = int(n)

    #we arbitrarily define a function to integrate
    def f(x):
            return x*x

    #this is the serial version of the trapezoidal rule
    #parallelization occurs by dividing the range among processes
    def integrateRange(a, b, n):
            integral = -(f(a) + f(b))/2.0
            # n+1 endpoints, but n trapazoids
            for x in np.linspace(a,b,n+1):
                            integral = integral + f(x)
            integral = integral* (b-a)/n
            return integral


    #h is the step size. n is the total number of trapezoids
    h = (b-a)/n
    #local_n is the number of trapezoids each process will calculate
    #note that NProcs must divide n
    local_n = n/NProcs

    #we calculate the interval that each process handles
    #local_a is the starting point and local_b is the endpoint
    local_a = a + MyRank*local_n*h
    local_b = local_a + local_n*h

    #initializing variables. mpi4py requires that we pass numpy objects.
    integral = np.zeros(1)
    total = np.zeros(1)

    # perform local computation. Each process integrates its own interval
    integral[0] = integrateRange(local_a, local_b, local_n)

    # communication
    # root node receives results with a collective "reduce"
    comm.Reduce(integral, total, op=MPI.SUM, root=0)

    time1=time()

    # root process prints results
    if MyRank == 0:
            print "Collective calc with n =", n, "trapezoids, our estimate of the integral from \n", a, "to", b, "is", total, '\nTime taken was: ',time1-time0,' s \n'

def trapezoidSerial(a,b,n):
    #Serial Trap int
    #Source: http://materials.jeremybejarano.com/MPIwithPython/pointToPoint.html

    #takes in command-line arguments [a,b,n]
    time0=time()
    a = float(a)
    b = float(b)
    n = int(n)

    def f(x):
            return x*x

    def integrateRange(a, b, n):
            '''Numerically integrate with the trapezoid rule on the interval from
            a to b with n trapezoids.
            '''
            integral = -(f(a) + f(b))/2.0
            # n+1 endpoints, but n trapazoids
            for x in np.linspace(a,b,n+1):
                    integral = integral + f(x)
            integral = integral* (b-a)/n
            return integral

    integral = integrateRange(a, b, n)
    time1=time()
    print "Serial calc with n =", n, "trapezoids, our estimate of the integral \n", a, "to", b, "is", integral,'\nTime taken was: ',time1-time0,' s \n'

def trapezoidP2P(a,b,n,MyRank=MyRank,comm=comm,NProcs=NProcs):
    # trap calc using Point to Point communication
    # Source: http://materials.jeremybejarano.com/MPIwithPython/pointToPoint.html
    #takes in command-line arguments [a,b,n]
    time0=time()
    a = float(a)
    b = float(b)
    n = int(n)

    #we arbitrarily define a function to integrate
    def f(x):
            return x*x

    #this is the serial version of the trapezoidal rule
    #parallelization occurs by dividing the range among processes
    def integrateRange(a, b, n):
            integral = -(f(a) + f(b))/2.0
            # n+1 endpoints, but n trapazoids
            for x in np.linspace(a,b,n+1):
                            integral = integral + f(x)
            integral = integral* (b-a)/n
            return integral


    #h is the step size. n is the total number of trapezoids
    h = (b-a)/n
    #local_n is the number of trapezoids each process will calculate
    #note that NProcs must divide n
    local_n = n/NProcs

    #we calculate the interval that each process handles
    #local_a is the starting point and local_b is the endpoint
    local_a = a + MyRank*local_n*h
    local_b = local_a + local_n*h

    #initializing variables. mpi4py requires that we pass numpy objects.
    integral = np.zeros(1)
    recv_buffer = np.zeros(1)

    # perform local computation. Each process integrates its own interval
    integral[0] = integrateRange(local_a, local_b, local_n)

    # communication
    # root node receives results from all processes and sums them
    if MyRank == 0:
            total = integral[0]
            for i in range(1, NProcs):
                    comm.Recv(recv_buffer, MPI.ANY_SOURCE)
                    total += recv_buffer[0]
    else:
            # all other process send their result
            comm.Send(integral)

    # root process prints results
    time1=time()
    if MyRank == 0:
            print "P2P calc with n =", n, "trapezoids, our estimate of the integral from \n", a, "to", b, "is", total, '\nTime taken was: ',time1-time0,' s \n'


def UpVLow(MyRank=MyRank,comm=comm,NProcs=NProcs):

    time0=time()
    data = None
    if MyRank == 0:
        data = np.arange(int(1e6),dtype=np.float64)
        comm.send(data, dest=1, tag=11)
    elif MyRank == 1:
        #print 'on task',MyRank,'before recv:   data = ',data
        data = comm.recv(source=0, tag=11)
        #print 'on task',MyRank,'after recv:    data = ',data

    time1=time()

    if MyRank==0: print 'Lowercase time elapsed= ',time1-time0

    del data
    comm.Barrier()

    time0=time()

    if MyRank == 0:
        data = np.arange(int(1e6),dtype=np.float64)
        comm.Send([data,3,MPI.DOUBLE],1,11)
    elif MyRank == 1:
        data = np.zeros(int(1e6),dtype=np.float64)
        #print 'on task',MyRank,'before Recv:   data = ',data
        comm.Recv(data,source=0,tag=11)
        #print 'on task',MyRank,'after Recv:    data = ',data

    time1=time()

    if MyRank==0: print 'Uppercase time elapsed= ',time1-time0

    comm.Barrier()

    return
