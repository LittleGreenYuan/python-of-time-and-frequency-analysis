# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 11:52:11 2020

@author: 33088
"""

import numpy as np
import matplotlib as matlab
import matplotlib.pyplot as plt
from scipy.io import loadmat as load
from scipy.fftpack import fft,ifft
import time
import random
import os
import math
from scipy.special import gamma
from matplotlib.colors import LogNorm
from datetime import datetime
from scipy import fftpack
import pywt#小波变换的库

def hermf(N, M, tm):
    dt=2*tm/(N-1);
    tt=np.linspace(-tm,tm,N);
    tt=tt.reshape(tt.size,1)
    
    P=np.zeros([M+1,N]);
    Htemp=np.zeros([M+1,N]);
    Dh=np.zeros([M,N]);
    
    g=np.exp((-tt**2)/2);
           
    TEMP=np.ones([1,N]);
    TEMP=TEMP.reshape(1,TEMP.size);
    P[0,:]=TEMP;
    TEMP=2*tt;
    TEMP=TEMP.reshape(1,TEMP.size);
    P[1,:]=TEMP;
    
    #range(1, 101)表示从1开始，到101为止（不包括101），取其中所有的整数。
    for k in range(2,M+1):
        TEMP=2*tt*P[k-1,:]-2*(k-2)*P[k-2,:];
        TEMP=TEMP.reshape(1,TEMP.size);
        P[k,:]=TEMP;
        
    for k in range(1,M+2):
        j=k-1;
        TEMP=(P[j,:]*g.reshape(1,g.size)/(math.sqrt(math.sqrt(math.pi)*(2**((k-1)*gamma(k))))))*math.sqrt(dt);
        TEMP=TEMP.reshape(1,TEMP.size);
        Htemp[j,:]=TEMP;
        
    del TEMP;#删除中间变量释放内存
    h=Htemp[0:M,:]
    for k in range(1,M+1):
        j=k-1;
        Dh[j,:]=(tt.reshape(1,tt.size)*Htemp[j,:]-math.sqrt(2*j)*Htemp[k,:])*dt;
    return h,Dh,tt


def STFT(x, lowFreq, highFreq, alpha, tDS, h, Dh):
    h=h.T;
    Dh=Dh.T;
    [xrow,xcol]=x.shape;
    t=np.arange(1,x.size+1,1);
    TEMP=t[0:x.size:tDS];#在python中A[x:y:z]表示从x到y中间隔z取样
    tLen=TEMP.size;
    t=t.reshape(1,t.size);
    del TEMP;
    tfrsqtic=np.arange(lowFreq,highFreq+alpha,alpha);
    tfrsqtic=tfrsqtic.reshape(1,tfrsqtic.size)
    
    N=tfrsqtic.size;
    Lidx=math.ceil( (N/2)*lowFreq/0.5 );
    Hidx=math.floor( (N/2)*highFreq/0.5 )
    fLen=Hidx-Lidx+1;
    tfrsqtic=np.linspace(lowFreq,highFreq,fLen);
    tfrsqtic=tfrsqtic.reshape(tfrsqtic.size,1);
    
    if( xcol != 1 ):
        raise Exception('ERROR(function_sqSTFTdef 00x1):X must have one column') 
    elif( highFreq > 0.5 ):
        raise Exception('ERROR(function_sqSTFTdef 00x2):TopFreq must be a value in [0, 0.5]');
    elif ((tDS<1)|(tDS%1)):
        raise Exception('ERROR(function_sqSTFTdef 00x3):tDS must be an integer value >= 1');
        
    [hrow,hcol]=h.shape;
    Lh=(hrow-1)/2;
    
    tfr=np.zeros([N,tLen]);
    tf3=np.zeros([N,tLen]);
    tfrtic=np.linspace(0,0.5,math.floor(N/2)).T;
    tfrtic=tfrtic.reshape(tfrtic.size,1);
    
    for tidx in range(tLen):
        #for tidx in range(1,1024):
        ti=t[0,((tidx-1)*tDS+1)];
        TEMP=np.array([round(N/2+0.5)-1,Lh,ti-1]);
        TEMP=TEMP.reshape(1,TEMP.size);
        if(TEMP.min() == 0):
            tau_min=(TEMP.min());
        elif(TEMP.min() != 0):
            tau_min=-(TEMP.min());
        TEMP=np.array([round(N/2+0.5)-1,Lh,xrow-ti]);
        TEMP=TEMP.reshape(1,TEMP.size);
        tau_max=TEMP.min();
        tau=np.arange(tau_min,tau_max+1,1);
        tau=tau.reshape(1,tau.size);
        
        indices=(N+tau)%N+1;
        indices=indices.astype(np.int);
        text_index=(Lh+1+tau)-1;
        text_index=text_index.astype(np.int);
        U,S,V=np.linalg.svd(h[text_index,0]);
        norm_h=S.max();#求取最大奇异值
        
        text_index_1=ti+tau-1;
        text_index_1=text_index_1.astype(np.int);
        tfr[indices-1,tidx]=x[text_index_1,0]*np.conj(h[text_index,0])/norm_h;
        tf3[indices-1,tidx]=x[text_index_1,0]*np.conj(Dh[text_index,0])/norm_h;
        
    del TEMP;

    [M_tfr,N_tfr]=tfr.shape;
    imag_number=np.zeros((M_tfr,N_tfr), dtype=np.complex);
    tfr=imag_number+tfr;
    
    for i in range(N_tfr):
        tfr[:,i]=fft(tfr[:,i]);
    
    [M_tf3,N_tf3]=tf3.shape;
    tf3=imag_number+tf3;
    
    for i in range(N_tf3):
        tf3[:,i]=fft(tf3[:,i]);


    #tfr=tfr.reshape(1,M_tfr*N_tfr);
    #tf3=tfr.reshape(1,M_tf3*N_tf3);
    [drop_x,avoid_warm]=np.nonzero(tfr);
    avoid_warm=avoid_warm.reshape(1,avoid_warm.size);
    drop_x=drop_x.reshape(1,drop_x.size);
    
    #tf3=abs(tf3);#;这里原程序里并没有用到tf3的abs，在python中如果调用就会出现tf3跟tfr变成0的情况，出问题的原因未知
    TEMP=np.zeros((M_tf3,N_tf3), dtype=np.complex);
    for i in range(drop_x.size):
        j=i;
        TEMP[drop_x[0,i],avoid_warm[0,j]]=complex(N,0)*tf3[drop_x[0,i],avoid_warm[0,j]]/tfr[drop_x[0,i],avoid_warm[0,j]]/complex(2.0*math.pi,0);
    
    TEMP=np.round(TEMP.imag);
    tf3=-TEMP;
    del TEMP;
    
    tfr=abs(tfr)
    tfr=tfr[0:round(N/2),:];
    return tfr, tfrtic



def imageSQ(t, tfrsqtic, tfrsq):
    tfrsq=abs(tfrsq);
    fz=20;
    [Sx,Sy]=tfrsq.shape;
    Q=tfrsq.reshape(Sx*Sy,1);
    #q=np.quantile(Q,0.998);
    q=np.quantile(Q,0.9978888)#matlab里是上面0.998，但python里同样的函数求不出来
    [index_x,index_y]=np.where(tfrsq>q);
    for i in range(index_x.size):
        tfrsq[index_x[i],index_y[i]]=q;
    x, y = t, tfrsqtic
    z = tfrsq
    plt.figure();
    plt.imshow(z+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
               norm=LogNorm(), origin ='lower', aspect=5)
    plt.title('sqSTFT')
    plt.show()
    return 0;

def choose_windows(name='Hamming', N=20):
    # Rect/Hanning/Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':
        window = np.ones(N)
    return window

def g_window(length,freq,lamda,p,factor):
    #广义ST变换高斯窗
    vector1=np.arange(0,length,1,dtype = float);
    vector1=vector1.reshape(1,vector1.size);
    vector2=np.arange((-length),(0),1,dtype = float);
    vector2=vector2.reshape(1,vector2.size);
    vector=np.vstack((vector1,vector2));
    del vector1,vector2;
    vector=vector**2;
    vector=vector*((-factor)*(lamda**2)*(math.pi**2))/(freq**(2*p));
    [M,N]=vector.shape
    gauss=np.zeros([M,N],dtype = float);
    for i in range(M):
        gauss[i,:]=np.exp(vector[i,:]);
    gauss=sum(gauss);
    gauss=gauss.reshape(1,gauss.size);
    return gauss;
    
    


def gst(timeseries,*params):
    #,minfreq,maxfreq,samplingrate,freqsamplingrate
    
    timeseries=x;
    TURE=1;
    FALSE=0;
    verbose=TURE;
    removeedge=TURE;
    analytic_signal=TURE;
    factor=1;
    
    lamda=1.5;
    p=0.8;
    
    if verbose:
        print('S-transform')
        
    [M,N]=timeseries.shape;
    if N>M:
        timeseries=timeseries.T;
    [M,N]=timeseries.shape;
    if( N > 1 ):
        raise Exception('ERROR(function_gst_def 00x1):Please enter a *vector* of data, not matrix')
    elif((M==1)&(N==1)):
        raise Exception('ERROR(function_gst_def 00x2):Please enter a *vector* of data, not scalar')

    if (len(params)==0):
        minfreq=0
        maxfreq=math.floor(len(timeseries)/2);
        samplingrate=1;
        freqsamplingrate=1;
    elif(len(params)==1):
        minfreq=params[0];
        maxfreq=math.floor(len(timeseries)/2);
        samplingrate=1;
        freqsamplingrate=1;
    elif(len(params)==2):
        minfreq=params[0];
        maxfreq=params[1];
        samplingrate=1;
        freqsamplingrate=1;
    elif(len(params)==3):
        minfreq=params[0];
        maxfreq=params[1];
        samplingrate=params[2];
        freqsamplingrate=1;
    elif(len(params)==4):
        minfreq=params[0];
        maxfreq=params[1];
        samplingrate=params[2];
        freqsamplingrate=params[3];
        
    if verbose:
        print('Minfreq=',minfreq);
        print('Maxfreq=',maxfreq);
        print('Sampling Rate (time domain)=',samplingrate);
        print('Sampling Rate (freq. domain)=',freqsamplingrate);
        print('The length of the timeseries is ',len(timeseries),'points\n');
        
    t=np.arange(0,len(timeseries),1,dtype = float);
    t=t*samplingrate;
    spe_nelements=math.ceil((maxfreq-minfreq+1)/freqsamplingrate);
    TEMP=np.arange(0,spe_nelements,1,dtype = float);
    f=(minfreq+TEMP*freqsamplingrate)/(samplingrate*len(timeseries));
    f=f.reshape(1,f.size);
    del TEMP;
    if verbose:
        print('The number of frequency voices is ',spe_nelements)
        
    #这里开始进行S变换，如果可以以后提取成单独的strans
    n=len(timeseries);
    original=timeseries;
    if removeedge:
        if verbose:
            print('Removing trend with polynomial fit.');
        ind=np.arange(0,n,1,dtype = float);
        r = np.polyfit(ind, timeseries, deg = 2);
        fit = np.polyval(r,ind);
        fit = fit.reshape(fit.size,1);
        ind = ind.reshape(ind.size,1);
        
        timeseries = timeseries-fit;
        if verbose:
            print('Removing edges with 5% hanning taper.');
        sh_len=math.floor(len(timeseries)/10);
        #wn=choose_windows(name='Hanning',N=sh_len);
        wn=np.hanning(sh_len);
        wn=wn.reshape(wn.size,1);
        if (sh_len==0):
            sh_len=len(timeseries);
            TEMP=np.arange(1,sh_len+1,1,dtype = float);
            TEMP=TEMP.reshape(1,TEMP.size);
            for i in range(len(TEMP)):
                if (TEMP<1):
                    wn[i,0]=1;
            
        
        [M,N]=wn.shape;
        if (N>M):
            wn=wn.T;
        timeseries[0:math.floor(sh_len/2),0]=timeseries[0:math.floor(sh_len/2),0]*wn[0:math.floor(sh_len/2),0];
        TEMP=timeseries[(len(timeseries)-math.floor(sh_len/2)-1):(n),0]*wn[(sh_len-math.floor(sh_len/2)-1):(sh_len),0];   
        timeseries[(len(timeseries)-math.floor(sh_len/2)-1):(n),0]=TEMP;
        del TEMP
        
        if analytic_signal:
            if verbose:
                print('Calculting analytic signal (using Hilbert transform)');
            [M_timeseries,N_timeseries]=timeseries.shape;
            imag_number=np.zeros((M_timeseries,N_timeseries), dtype=np.complex);
            timeseriesx=imag_number+timeseries;
            for i in range(N_timeseries):
                ts_spe=fft(timeseriesx[:,i]);
            ts_spe=ts_spe.reshape(ts_spe.size,1);
            h1=np.ones([1,1]);
            h2=2*np.ones([math.floor((n-1)/2),1]);
            h3=np.ones([(1-n%2),1]);
            h4=np.zeros([math.floor((n-1)/2),1]);
            h=np.vstack((h1,h2));
            h=np.vstack((h,h3));
            h=np.vstack((h,h4));
            del h1,h2,h3,h4;
            ts_spe=ts_spe*h;
            for i in range(N_timeseries):
                timeseries=ifft(ts_spe[:,i]);
            timeseries=timeseries.reshape(timeseries.size,1);
            
            tic = datetime.now()
            for i in range(N_timeseries):
                vector_fft=fft(timeseries[:,i]);
            vector_fft=vector_fft.reshape(vector_fft.size,1);
            toc = datetime.now()
            
            vector_fft=np.hstack((vector_fft,vector_fft));
            tim_est=(toc-tic).total_seconds()*math.floor((maxfreq-minfreq+1)/freqsamplingrate);
            if verbose:
                print('Estimated time is',tim_est);
                
            st=np.zeros([math.floor((maxfreq-minfreq+1)/freqsamplingrate),n], dtype=np.complex);
            if verbose:
                print('Calculating S transform...');
            if (minfreq==0):
                TEMP=np.ones([1,len(st[0,:])]);
                st[0,:]=timeseries.mean()*TEMP;
                del TEMP;
            else:
                [M,N]=vector_fft.shape;
                TEMP1=np.zeros((M,1), dtype=np.complex);
                TEMP1=vector_fft[minfreq:minfreq+n-1,0]*g_window(n,minfreq,lamda,p,factor);
                for i in range(N_timeseries):
                    st[0,:]=ifft(TEMP1[:,i]);
                del TEMP1;
            [M,N]=vector_fft.shape;
            
            
            TEMP_vector=vector_fft[:,0];
            TEMP_vector=TEMP_vector.reshape(1,TEMP_vector.size);
            [M,N]=TEMP_vector.shape;
            #for banana in range(2):
            for banana in range((maxfreq-minfreq-1)):
                TEMP_index=int(banana/freqsamplingrate+1)-1;
                i=minfreq+banana+1-1;
                j=minfreq+banana+n-1;
                if (j<=(N-1)):
                    TEMP=TEMP_vector[0,minfreq+banana+1-1:minfreq+banana+n]*g_window(n,minfreq+banana+1,lamda,p,factor);
                elif(j>(N-1)):
                    guass=g_window(n,minfreq+banana+1,lamda,p,factor);
                    TEMP=TEMP_vector[0,minfreq+banana+1-1:(N)];
                    TEMP=TEMP.reshape(1,TEMP.size);
                    TEMP_part=TEMP_vector[0,0:(j-n+1)];
                    TEMP_part=TEMP_part.reshape(1,TEMP_part.size);
                    TEMP=np.hstack((TEMP,TEMP_part))
                    TEMP=TEMP*guass;
                    
                st[TEMP_index,:]=TEMP;
                del TEMP;
            [M,N]=st.shape;
            for i in range(M):
                st[i,:]=ifft(st[i,:]);
            if verbose:
                print('Finished Caulculation')
    return st,t,f;

def nextpow2(data):
    n=math.log10(FreqBins)/math.log10(2);
    n=round(n);
    return n;


def cwt(Sig,*params):
    if (len(params)==0):
        wavename = "cgau8";
        totalscal = 256+1;
        sampling_rate=1024;
    elif(len(params)==1):
        wavename = params[0];
        totalscal = 256+1;
        sampling_rate=1024;
    elif(len(params)==2):
        wavename = params[0];
        totalscal =params[1];
        sampling_rate=1024;
    elif(len(params)==3):
        wavename = params[0];
        totalscal =params[1];
        sampling_rate=params[3];
        
    fc = pywt.central_frequency(wavename);#中心频率
    cparam = 2 * fc * totalscal;
    scales = cparam/np.arange(totalscal,1,-1);
    
    Sig=Sig.reshape(Sig.size,);
    
    [cwtmatr, frequencies] = pywt.cwt(Sig,scales,wavename,1.0/sampling_rate)#连续小波变换
    cwtmatr=np.flip(cwtmatr,0)
    return cwtmatr, frequencies;
    
def imshow_filtter(matr):
    #将矩阵中小于某个数的元素变成0
    matr=matr.real;
    
    # 方式一
    #np.maximum(a, 0)
    # 方式二
    #(a + abs(a)) / 2
    # 方式三
    matr[matr<(-0.5)]=0;
    E=matr.mean();
    # 方式四
    np.where(matr > (E), matr, 0);
    
    return matr;


#这里提供了STFT一个测试函数的用例
'''
t=np.linspace(0,3,1024);
t=t.reshape(t.size,1);
#x=np.sin(25*math.pi*t*t);
x=np.sin(5*math.pi*t*t*t+25*math.pi*t)+np.sin(10*math.pi*t*t*t+30*math.pi*t);
#plt.plot(t,y)#绘制x的曲线图
#Tools > Preferences > IPython Console > Graphics > Graphics backend, inline 即终端输出，Qt5则是新窗口输出。
lowFreq=0;
highFreq=0.5;
alpha=0.5/x.size;
#[M,N]=y.shape
tDS=1;

N=121;
M=1;
tm=5;
[h,Dh,tt]= hermf(N, M, tm);


[tfr, tfrtic] = STFT(x, lowFreq, highFreq, alpha, tDS, h, Dh);
imageSQ(t, tfrtic, tfr)
'''

#这里提供了一个S测试函数的用例
'''
t=np.linspace(0,3,1024);
t=t.reshape(t.size,1);
x=np.sin(25*math.pi*t*t+25*math.pi*t);

SampFreq=80;

n=t.size;
fre_start=(SampFreq/2)/(n/2);
fre_ma=(SampFreq/2)/(n/2);
fre_end=(SampFreq/2)+(SampFreq/2)/(n/2);
fre=np.arange(fre_start,fre_end,fre_ma,dtype = float);
fre=fre.reshape(1,fre.size);

hlength=55;
minfreq=0
maxfreq=512;
samplingrate=1;
freqsamplingrate=1;

st,tx,f=gst(x);
x, y = t, f[0,0:512]
z = st.real;
plt.figure();
plt.imshow(z+100, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
           norm=LogNorm(), origin ='lower', aspect=5)
plt.title('S')
plt.show()
'''

#这里提供了一个小波变换测试函数的用例
'''
t=np.linspace(0,3,1024);
t=t.reshape(t.size,1);
x=np.sin(25*math.pi*t*t+25*math.pi*t);

[cwtmatr, frequencies] = cwt(x);
x, y = t, f
z = abs(cwtmatr);
plt.figure();
plt.imshow(z+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
           norm=LogNorm(), origin ='lower', aspect=5)
plt.title('CWT')
plt.show()
'''


    
    