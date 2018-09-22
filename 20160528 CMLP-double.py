# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:52:08 2016

@author: nomuraryoji
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
import time
#import csv
#from multiprocessing import Pool
#from multiprocessing import Process
#import functools

def gaborc(H,k): #H is size of filter, l is direction, k is coeff for sigma
    buf = np.zeros([H, H])
    l = 2*math.pi*math.cos(k)/H
    m = 2*math.pi*math.sin(k)/H
    n = 1/math.sqrt(2*math.pi*H*H)
    for i in range(H):
        for j in range(H):
            buf[i,j] = math.cos((l*(i-H/2)+m*(H/2-j))) * n * math.exp(-((i-H/2)**2+(H/2-j)**2)/(2*H*H))
    return buf

def gabors(H,k): #H is size of filter, l is direction, k is coeff for sigma
    buf = np.zeros([H, H])
    l = 2*math.pi*math.cos(k)/H
    m = 2*math.pi*math.sin(k)/H
    n = 1/math.sqrt(2*math.pi*H)
    for i in range(H):
        for j in range(H):
            buf[i,j] = math.sin((l*(i-H/2)+m*(H/2-j))) * n * math.exp(-((i-H/2)**2+(H/2-j)**2)/(2*H*H))
    return buf

def zpadding(x, n, m, p): # x is (nxm)x1 array, p is addtional zeros
    y = x.reshape([n,m])    
    ab = np.zeros([p,m])
    side = np.zeros([n+2*p, p])
    y = np.r_[ab,y,ab]
    y = np.c_[side,y,side]
    return y.reshape([(n+2*p)*(m+2*p)])
    
def Wconv(h, n, m): # h is filter matrix. nxm is conv-ed matrix
    nh = len(h)    
    nc = n-nh+1
    mc = m-nh+1
    buf = np.zeros([nc*mc,n*m])
    for i in range(mc):
        for j in range(nc):
            for p in range(nh):
                for q in range(nh): 
                    buf[i+j*mc,i+j*m+p+q*m] = h[p,q]
    return buf
    
def pooling(x,n,m,p,s): # p is size of pooling, s is stride, maximum pooling
    nc = int((n-1)/s)
    mc = int((m-1)/s)
    buf = np.zeros([nc,mc])
    x = x.reshape([n,m]) 
    w = np.zeros([nc*mc,n*m])
    for i in range(nc):
        for j in range(mc):
            maxr = 0
            maxq = 0
            for r in range(p):
                for q in range(p):
                    if (buf[i,j] < x[i*s+r,j*s+q]):
                        maxr = r #update 
                        maxq = q #update                 
            buf[i,j] = x[i*s+maxr,j*s+maxq]
            w[i*mc+j,(i*s+maxr)*m+j*s+maxq] = 1                     
    buf = buf.reshape([nc*mc])
    return (buf, w) 
    
def relu(x): # x is 1-dim array
    n = x.size
    buf = np.zeros([n])    
    pos = np.zeros([n])
    for i in range(n):
        if (x[0,i]>0):
            buf[i] = x[0,i]
            pos[i] = 1 # pos is activated position
    return (buf, pos) 

def connorm(x):
    c = 1e-3
    return (x - np.mean(x))/math.sqrt((c**2+ np.std(x**2)))
    
def sigmoid(y):
    return np.tanh(y)
	
def softmax(y):
    buf = np.exp(y)
    return buf/np.sum(buf) 
    
def palconv(p,x,y):
    return x[p,:,:].dot(y.T).T

def main():    
# load mnist data . 70000 sumple, 28x28 pix
    mnist = fetch_mldata('MNIST original', data_home=".")
    X = mnist.data #place on memory
    X = X.astype(np.float64) # change type from int to float64
    X /= X.max() #adjust maximum as 1

# slice data
    N = 60000
    X_train, X_test = np.split(X, [N]) # divide X at the position of N
    Y_train, Y_test = np.split(mnist.target, [N]) # same above
    n_test = Y_test.size # the amount of Y_test

    T_train = LabelBinarizer().fit_transform(Y_train) # transoform to 1-of-k form
    T_test = LabelBinarizer().fit_transform(Y_test) # same as above

# dimension for each layer
    n = 28 # dim of image (X) row
    m = 28 # dim of image (X) column

# parameters
    cl = 10 # dimension of class
    bs = 10# batch size
    pd = 4 # zero padding size
    dh = 11 # dim of filter
    dh2 = 3 # dim of second filter
    nh = 12 # number of filter
    nh2 = 12 # number of filter
    st = 2 # stride
    pl = 3 # pooling size
    eta = 0.1 # learning speed
    eta2 = 0.1 # learning speed
    lam = 0.0001 # weigt decay 
    mu = 0.01 # momentum

    train_cnt = 1000
    test_cnt = 100
    
# dimensions
    nzp = n+2*pd # dim of padded X row 28+4+4 = 36
    mzp = m+2*pd # dim of padded X column 28+4+4 = 36
    ncv = nzp-dh+1 # dim of conv-ed X row 36-11+1 = 26
    mcv = mzp-dh+1 # dim of conv-ed X column 36-11+1 = 26
    npl = int((ncv)/st)+1-2*int(pl/2)  # dim of pooled X row 12
    mpl = int((mcv)/st)+1-2*int(pl/2) # dim of pooled X column 12
    ncv2 = npl-dh2+1 # dim of conv-ed X row  12-4+1 = 9
    mcv2 = npl-dh2+1 # dim of conv-ed X column 12-4+1 = 9
    npl2 =  4 # dim of pooled X row
    mpl2 =  4 # dim of pooled X column 
    
# first convolution filter [dh1 x dh1]matrix x nh1 as gabor filter
    H = np.zeros([nh,dh,dh]) # filter of convolution
    WC = np.zeros([nh,ncv*mcv,nzp*mzp]) # matrix of convolution
    for i in range(int(nh/2)):
        H[i,:,:] = gaborc(dh,i*math.pi/nh*2)
        WC[i,:,:] = Wconv(H[i,:,:], nzp, mzp)
        H[i+int(nh/2),:,:] = gabors(dh,i*math.pi/nh*2)
        WC[i+int(nh/2),:,:] = Wconv(H[i+int(nh/2),:,:], nzp, mzp)

# bias of convolution
    Bc = 0.1 * np.random.random(nh)
    Bc1 = np.zeros([nh,ncv*mcv])

# bias of convolution
    Bc2 = 0.1 * np.random.random(nh2)
    Bc21 = np.zeros([nh2,ncv2*mcv2])

# second convolution filter [dh1 x dh1]matrix x nh1 as gabor filter
    H2 = 0.1 * np.random.random(nh2*dh2*dh2).reshape([nh2,dh2,dh2])
    WC2 = np.zeros([nh2,ncv2*mcv2,npl*mpl]) # matrix of convolution

# variables for learning
    Xzp = np.zeros([bs,nzp*mzp])
    Xcv = np.zeros([bs,nh,ncv*mcv])
    Xre = np.zeros([bs,nh,ncv*mcv])
    Pos = np.zeros([bs,nh,ncv*mcv])               

    Xpl = np.zeros([bs,nh,npl*mpl])
    WP = np.zeros([bs,nh,npl*mpl,ncv*mcv])
    Xcn = np.zeros([bs,nh,npl*mpl])

    Xcv2 = np.zeros([bs,nh2,ncv2*mcv2])
    Xre2 = np.zeros([bs,nh2,ncv2*mcv2])
    Pos2 = np.zeros([bs,nh2,ncv2*mcv2])               

    Xpl2 = np.zeros([bs,nh2,npl2*mpl2])
    WP2 = np.zeros([bs,nh2,npl2*mpl2,ncv2*mcv2])
    Xcn2 = np.zeros([bs,nh2,npl2*mpl2])
    Zcn2 = np.zeros([bs,nh2*npl2*mpl2])        


# all connection  
    WF = 0.01 * np.random.random(cl*nh2*npl2*mpl2).reshape([cl,nh2*npl2*mpl2])
    Bo = 0.01 * np.random.random(cl) # bias

# output 
    Y = np.zeros([cl,bs])

# differential
    dY = np.zeros([cl,bs])
    dXcn2 = np.zeros([bs,nh2,npl2*mpl2]) # 144 x nh1 dim, d of pooling 
    dXpl2 = np.zeros([bs,nh2,ncv2*mcv2]) # 676 dim, d of ReLU, only backprop.
    dXre2 = np.zeros([bs,nh2,ncv2*mcv2])
    dXre2a = np.zeros([bs,nh2,ncv2,mcv2]) 
    dDH2 = np.zeros([bs,nh2,dh2,dh2]) # 6x6 dim  diff of filter
    dXpl = np.zeros([bs,nh,npl*mpl])
    dXre = np.zeros([bs,nh,ncv*mcv])
    dXcv = np.zeros([bs,nh,ncv*mcv])
    dXcva = np.zeros([bs,nh,ncv,mcv])
    dDH = np.zeros([bs,nh,dh,dh])

    dBc2 = np.zeros([nh2])
    
    dBc = np.zeros([nh])

    dWF = np.zeros([cl,nh2*npl2*mpl2])
    dBo = np.zeros([cl])
    dH2 = np.zeros([nh2,dh2,dh2]) 
    dH = np.zeros([nh,dh,dh]) 

# for momentum
    dWF_b = dWF
    dBo_b = dBo
    dH2_b = dH2  
    dBc2_b = dBc2
    dH_b = dH 
    dBc_b = dBc

# learning
# forward propergation
    start = time.time()
    epoch = int(train_cnt/bs)
    TT = np.zeros([cl,epoch,bs])
    YY = np.zeros([cl,epoch,bs])
    HH = np.zeros([dh*dh,epoch])
    HH2 = np.zeros([dh2*dh2,epoch])
    for e in range(epoch):   
        for k in range(nh):
            WC[k,:,:] = Wconv(H[k,:,:], nzp, mzp)
        for l in range(int(nh2)):
            WC2[l,:,:] = Wconv(H2[l,:,:], npl, mpl)
        for k in range(nh):
            for j in range(ncv*mcv):
                Bc1[k,j] = Bc[k]
        for l in range(nh2):
            for j in range(ncv2*mcv2):
                Bc21[l,j] = Bc2[l]            

        pp = np.random.random_integers(0, N-1, bs)
        for b in range(len(pp)):
            x = X_train[pp[b]]
            Xzp[b,:] = zpadding(x,n,m,pd)
            xzp1 = Xzp[b,:].reshape([nzp,mzp])  
            
            for k in range(nh):
                Xcv[b,k,:] = WC[k,:,:].dot(Xzp[b,:].T).T + Bc1[k,:]
                Xre[b,k,:], Pos[b,k,:] = relu(Xcv[b,k,:].reshape([1,ncv*mcv]))
                Xpl[b,k,:], WP[b,k,:,:] = pooling(Xre[b,k,:],ncv,mcv, pl, st) 
                Xcn[b,k,:]= connorm(Xpl[b,k,:])
            
            for l in range(nh2):
                for k in range(nh):
                    Xcv2[b,l,:] += WC2[l,:,:].dot(Xcn[b,k,:].T).T + Bc21[l,:]
                Xre2[b,l,:], Pos2[b,l,:] = relu(Xcv2[b,l,:].reshape([1,ncv2*mcv2]))
                Xpl2[b,l,:], WP2[b,l,:,:] = pooling(Xre2[b,l,:],ncv2,mcv2,pl, st) 
                Xcn2[b,l,:]= connorm(Xpl2[b,l,:])
            Zcn2[b,:] = Xcn2[b,:,:].reshape([1,nh2*npl2*mpl2])
            Y[:,b] = softmax(WF.dot(Zcn2[b,:])+Bo)
            dY[:,b] = Y[:,b]-T_train[pp[b]] 
            TT[:,e,b] = T_train[pp[b]]
            YY[:,e,b] = Y[:,b] 
            
            dBo -= eta2*dY[:,b]
            for i in range(cl):
                for j in range(nh2*npl2*mpl2):
                    dWF[i,j] -= eta2*dY[i,b]*Zcn2[b,j]
            dXcn2[b,:,:] = WF.T.dot(dY[:,b]).T.reshape([nh2,npl2*mpl2])
            for l in range(nh2):
                dXpl2[b,l,:] = WP2[b,l,:,:].T.dot(dXcn2[b,l,:]) 
                dXre2[b,l,:] = Pos2[b,l,:]*dXpl2[b,l,:] 
                dXre2a[b,l,:,:] = dXre2[b,l,:].reshape([ncv2,mcv2])      
                xpl1 = Xpl[b,l,:].reshape([npl,mpl])

                for i in range(ncv2):
                    for j in range(mcv2):
                        dBc2 -= eta2 * dXre2a[b,l,i,j]
                        for s in range(dh2):
                            for t in range(dh2):
 
                                dDH2[b,l,s,t] += dXre2a[b,l,i,j]*xpl1[i+s,j+t] 
                dH2[l,:,:] -= eta2*dDH2[b,l,:,:]

            for k in range(nh):
                for l in range(nh2):
                    dXpl[b,k,:] += WC2[l,:,:].T.dot(dXre2[b,l,:]).T
                dXre[b,k,:] = WP[b,k,:,:].T.dot(dXpl[b,k,:]) 
                dXcv[b,k,:] = Pos[b,k,:]*dXre[b,k,:] 
                dXcva[b,k,:,:] = dXcv[b,k,:].reshape([ncv,mcv])
                for i in range(ncv):
                    for j in range(ncv):
                        dBc[k] -= eta * dXcva[b,k,i,j]
                        for s in range(dh):
                            for t in range(dh):
                                dDH[b,k,s,t] += dXcva[b,k,i,j]*xzp1[i+s,j+t]
                dH[k,:,:] -= eta*dDH[b,k,:,:]
    
        WF += mu * dWF_b + dWF / bs - eta2 * lam * WF
        Bo += mu * dBo_b + dBo / bs - eta2 * lam * Bo
        H2 += mu * dH2_b + dH2 / bs - eta2 * lam * H2  
        Bc2 += mu * dBc2_b + dBc2 / bs - eta2 * lam * Bc2
        H += mu * dH_b + dH / bs - eta * lam * H
        Bc += mu * dBc_b + dBc / bs - eta * lam * Bc

        dWF_b = mu * dWF_b + dWF / bs - eta * lam * WF
        dBo_b = mu * dBo_b + dBo / bs - eta * lam * Bo
        dH2_b = mu * dH2_b + dH2 / bs - eta * lam * H2 
        dBc2_b = mu * dBc2_b + dBc2 / bs - eta * lam * Bc2
        dH_b = mu * dH_b + dH / bs - eta * lam * H 
        dBc_b = mu * dBc_b + dBc / bs - eta * lam * Bc

        HH[:,e] = H[0,:,:].reshape([dh*dh])
        HH2[:,e] = H2[0,:,:].reshape([dh2*dh2])
                
#    TEST = T.reshape([cl,epoch*bs])

    plt.figure(1)    
    plt.pcolor(TT.reshape([cl,epoch*bs]))

    plt.figure(2)    
    plt.pcolor(YY.reshape([cl,epoch*bs]))
#    pylab.pcolor(xpl1)
    plt.figure(3)    
    plt.pcolor(HH)
    
    plt.figure(4)    
    plt.pcolor(HH2)

    plt.figure(5)    
    plt.pcolor(H[0,:,:])
           
    plt.figure(6)    
    plt.pcolor(H2[0,:,:])
    
    print("time", time.time() - start)   


# final WC (convolution matrix)    
    for k in range(nh):
        WC[k,:,:] = Wconv(H[k,:,:], nzp, mzp)
    for l in range(int(nh2)):
        WC2[l,:,:] = Wconv(H2[l,:,:], npl, mpl)


# variables for testing
    xzp = np.zeros([nzp*mzp])
    xcv = np.zeros([nh,ncv*mcv])
    xre = np.zeros([nh,ncv*mcv])
    pos = np.zeros([nh,ncv*mcv])               

    xpl = np.zeros([nh,npl*mpl])
    wp = np.zeros([nh,npl*mpl,ncv*mcv])
    xcn = np.zeros([nh,npl*mpl])

    xcv2 = np.zeros([nh2,ncv2*mcv2])
    xre2 = np.zeros([nh2,ncv2*mcv2])
    pos2 = np.zeros([nh2,ncv2*mcv2])               

    xpl2 = np.zeros([nh2,npl2*mpl2])
    wp2 = np.zeros([nh2,npl2*mpl2,ncv2*mcv2])
    xcn2 = np.zeros([nh2,npl2*mpl2])
    out = np.zeros([cl, cl])   

    y = np.zeros([cl])
    pp = np.random.random_integers(0, n_test-1, test_cnt)
    for b in range(len(pp)):
        x = X_test[pp[b]]
        xzp = zpadding(x,n,m,pd)
        xzp1 = xzp.reshape([nzp,mzp])  
        
        for k in range(nh):
            xcv[k,:] = WC[k,:,:].dot(xzp.T).T
            xre[k,:], pos[k,:] = relu(xcv[k,:].reshape([1,ncv*mcv]))
            xpl[k,:], wp[k,:,:] = pooling(xre[k,:],ncv,mcv, pl, st) 
            xcn[k,:]= connorm(xpl[k,:])
        
        for l in range(nh2):
            for k in range(nh):
                xcv2[l,:] += WC2[l,:,:].dot(xcn[k,:].T).T
            xre2[l,:], pos2[l,:] = relu(xcv2[l,:].reshape([1,ncv2*mcv2]))
            xpl2[l,:], wp2[l,:,:] = pooling(xre2[l,:],ncv2,mcv2,pl, st) 
            xcn2[l,:]= connorm(xpl2[l,:])

        zcn2 = xcn2.reshape([nh2*npl2*mpl2])
        y = softmax(WF.dot(zcn2)+Bo)
        d = int(Y_test[pp[b]])
        out[d, np.argmax(y)] += 1 # trace is correct answer
    print(out)

if __name__ == '__main__':
    main()
