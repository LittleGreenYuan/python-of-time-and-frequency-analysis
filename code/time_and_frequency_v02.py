# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 11:52:11 2020

@author: Green_yuan
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
#import pywt#小波变换的库
from scipy.signal import hilbert, chirp

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


def sqSTFT(x, lowFreq, highFreq, alpha, tDS, h, Dh):
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
    tfrsq=np.zeros([fLen,tLen]);
    Ex=np.mean(abs(x[t.min():t.max(),0]))
    Threshold=1.08e-8*Ex;
    #tfr=tfr.reshape(M_tfr,N_tfr);
    #tf3=tf3.reshape(M_tfr,N_tfr);
    
    for icol in range(tLen):
        for  jcol in range(round(N/2)):
            if (abs(tfr[jcol,icol]) > Threshold):
                jcolhat = jcol - tf3[jcol,icol];
                jcolhat=jcolhat.astype(np.int);
                lhat = ((((jcolhat-1)%N)+N)%N)+1;
                lhat=lhat.astype(np.int);
                if ((lhat < Hidx + 1) & (lhat >= Lidx)):
                    tfrsq[lhat-Lidx+1-1,icol] = tfrsq[lhat-Lidx+1-1, icol] + tfr[jcol,icol] ;
    return tfr, tfrtic, tfrsq, tfrsqtic

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
    #plt.title('STFT')
    plt.show()
    '''
    #这里是绘制图像的三个小例子
    import matplotlib.pyplot as plt
    import numpy as np
    
    grid = np.random.random((10,10))
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))
    
    ax1.imshow(grid, extent=[0,100,0,1])
    ax1.set_title('Default')
    
    ax2.imshow(grid, extent=[0,100,0,1], aspect='auto')
    ax2.set_title('Auto-scaled Aspect')
    
    ax3.imshow(grid, extent=[0,100,0,1], aspect=100)
    ax3.set_title('Manually Set Aspect')
    
    plt.tight_layout()
    plt.show()
    '''
    return 0;

def SET_Y(x, hlength=55):
    [xrow,xcol]=x.shape;
    N=xrow;
    
    if( xcol != 1 ):
        raise Exception('ERROR(function_SET_Ydef 00x1):X must have one column') 

    #这里原本有一个判断，如果输入参数小于2，就将helength设定为round(xrow/8)
    
    t=np.arange(1,N+1,1,dtype = float);
    t=t.reshape(1,t.size);
    
    ft=np.arange(1,np.round(N/2+0.5)+1,1,dtype = float);
    ft=ft.reshape(1,ft.size);
    
    [trow,tcol]=t.shape;
    
    hlength=(hlength+1)-hlength%2;
    ht=np.linspace(-0.5,0.5,hlength);
    ht=ht.reshape(ht.size,1);
    
    #高斯窗
    h=np.exp((-math.pi)/(0.32**2)*(ht**2));
    dh=(-2*math.pi/(0.32**2))*ht*h;
    
    [hrow,hcol]=h.shape;
    Lh=(hrow-1)/2;
    
    tfr1=np.zeros([N,tcol],dtype = float);
    tfr2=np.zeros([N,tcol],dtype = float);
        
    for icol in range(tcol):
        ti=t[0,icol];
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
        
        indices=((N+tau)%N)+1;
        indices=indices.astype(np.int);
        r_index=(ti+tau);
        r_index=r_index.astype(np.int);
        rSig=x[r_index-1,0];
        
        tf_index=Lh+1+tau;
        tf_index=tf_index.astype(np.int);
        
        tfr1[indices-1,icol]=rSig*np.conj(h[tf_index-1,0]);
        tfr2[indices-1,icol]=rSig*np.conj(dh[tf_index-1,0]);
        
    del TEMP;
    
    [M_tfr1,N_tfr1]=tfr1.shape;
    imag_number=np.zeros((M_tfr1,N_tfr1), dtype=np.complex);
    tfr1=imag_number+tfr1;
    for i in range(N_tfr1):
        tfr1[:,i]=fft(tfr1[:,i]);
        
        [M_tfr2,N_tfr2]=tfr2.shape;
        tfr2=imag_number+tfr2;
    for i in range(N_tfr2):
        tfr2[:,i]=fft(tfr2[:,i]);
        
    tfr1=tfr1[0:round(M_tfr2/2),:];
    tfr2=tfr2[0:round(M_tfr2/2),:];
    
    va=N/hlength;
    IF=np.zeros((round(N/2),tcol), dtype=np.complex);
    tfr=np.zeros((round(N/2),tcol), dtype=np.complex);
    
    E=np.mean(abs(x),0) # 0压缩列，对各行求均值;1压缩行，对各列求均值
    
    TEMP=0+1j;
    TEMP=va*TEMP;
    
    for i in range(round(N/2)):
        for j in range(N):
            TEMP_COM=((TEMP*tfr2[i,j]/2)/math.pi)/tfr1[i,j];
            COMPARE=abs(-TEMP_COM.real);
            if COMPARE<0.4:
                IF[i,j]=1;
                
    del TEMP;
    del COMPARE;        
    tfr=tfr1/(sum(h)/2);
    Te=tfr*IF;
    return IF,Te,tfr;
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

def WignerVille(Sig,*params):
    #,SampFreq,FreqBins
    
    hx_image = fftpack.hilbert(Sig.real);#这里只能提取出希尔伯特变换的虚部，实部依旧是原信号
    hx=Sig-1j*hx_image;
    '''
    hx_image=hilbert(Sig.real);
    hx=Sig.real-np.imag(hx_image);
    '''
    
    Sig=hx;
    SigLen=len(Sig)
    if (len(params)==0):
        SampFreq=1;
        FreqBins=512;
    elif(len(params)==1):
        SampFreq=params[0];
        FreqBins=512;
    elif(len(params)==2):
        SampFreq=params[0];
        FreqBins=params[1];
    FreqBins=2**nextpow2(FreqBins);
    FreqBins=min(FreqBins,SigLen);
    WVD=np.zeros([FreqBins,SigLen], dtype=np.complex);
    #for iIndex in range(5):
    for iIndex in range(SigLen):
        Mtau=min(iIndex,SigLen-(iIndex+1),round(FreqBins/2)-1);
        if (Mtau==0):
            tau=0;
        else:
            tau=np.arange(-Mtau,Mtau+1,1,dtype = int);
        Temp=(FreqBins+tau)%FreqBins;
        WVD[Temp,iIndex]=Sig[iIndex+tau,0]*np.conj(Sig[iIndex-tau,0]);
        
    
    [M,N]=WVD.shape;
    WVD_fft=np.zeros([M,N], dtype=np.complex);
    for i in range(N):
        WVD_fft[:,i]=fft(WVD[:,i]);
    
    WVD=WVD_fft/FreqBins;
    f=np.linspace(0,1,FreqBins);
    f=f*(SampFreq/2);
    t=np.arange(0,SigLen,1,dtype = float);
    return WVD,t,f;

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

'''
用VMD分解算法时只要把信号输入进行分解就行了，只是对信号进行分解，和采样频率没有关系，VMD的输入参数也没有采样频率。
VMD分解出的各分量在输出量 u 中，这个和信号的长度、信号的采样频率没有关系。迭代时各分量的中心频率在输出量omega，
可以用2*pi/fs*omega求出中心频率，但迭代时的频率是变化的。

'''
def vmd( signal, alpha, tau, K, DC, init, tol):
    '''
    Input and Parameters:
    signal  - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                       2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6

    Output:
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    '''
    # Period and sampling frequency of input signal
    #分解算法中的采样频率和时间是标准化的，分解信号的采样时间为1s,然后就得到相应的采样频率。采样时间间隔：1/ length(signal)，频率： length(signal)。
    save_T = len(signal)
    fs = 1 / save_T
    # extend the signal by mirroring镜像延拓
    T = save_T
    f_mirror = []
    temp = signal[0:T//2]
    f_mirror.extend(temp[::-1]) #temp[::-1] 倒序排列
    f_mirror.extend(signal)
    temp = signal[T//2:T]
    f_mirror.extend(temp[::-1])

    f = f_mirror

    # Time Domain 0 to T (of mirrored signal)
    T = len(f)
    t = [(i + 1) / T for i in range(T)]  # 列表从1开始
    # Spectral Domain discretization
    #freqs 进行移位是由于进行傅里叶变换时，会有正负对称的频率，分析时一般只有正频率，所以看到的频谱图是没有负频率的
    freqs = np.array( [i - 0.5 - 1 / T for i in t] )

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)
    # Construct and center f_hat
    transformed = np.fft.fft(f)  # 使用fft函数对信号进行快速傅里叶变换。
    f_hat = np.fft.fftshift(transformed)  # 使用fftshift函数进行移频操作。
    f_hat_plus = f_hat
    f_hat_plus[0:T // 2] = 0
    # f_hat_plus[0:T] = 1                #????????????????????????????////////////

    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = [np.zeros((N, len(freqs)), dtype=complex) for i in range(K)]
    # Initialization of omega_k
    omega_plus = np.zeros((N, K))

    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(K)))
    else:
        omega_plus[0, :] = 0
        # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0
        # start with empty dual variables
    lambda_hat = np.zeros( (N, len(freqs)), dtype=complex)
    # other inits
    eps = 2.2204e-16  # python里没有eps功能
    uDiff = tol + eps  # update step
    n = 1  # loop counter
    sum_uk = 0  # accumulator


    #----------- Main loop for iterative updates----------
    while (uDiff > tol and  n < N ):    #not converged and below iterations limit
        #update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[K-1][n-1,:]+ sum_uk - u_hat_plus[0][n-1,:]  #sum_uk 一直都等于0（1,2000）????????????????
        #update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[k][n,:] = (f_hat_plus - sum_uk - lambda_hat[n-1,:]/2)/(1+Alpha[k]*(freqs - omega_plus[n-1,k])**2)
        #update first omega if not held at 0
        if not DC:
            omega_plus[n,k] = (freqs[T//2:T]*np.mat(np.abs(u_hat_plus[k][n, T//2:T])**2).H)/np.sum(np.abs(u_hat_plus[k][n,T//2:T])**2)

        #update of any other mode
        for k in range(K-1):
            #accumulator
            sum_uk = u_hat_plus[k][n,:] + sum_uk - u_hat_plus[k+1][n-1,:]
            #mode spectrum
            u_hat_plus[k+1][n,:] = (f_hat_plus - sum_uk - lambda_hat[n-1,:]/2)/(1+Alpha[k+1]*(freqs - omega_plus[n-1,k+1])**2)
            #center frequencies
            omega_plus[n,k+1] = (freqs[T//2:T]*np.mat(np.abs(u_hat_plus[k+1][n, T//2:T])**2).H)/np.sum(np.abs(u_hat_plus[k+1][n,T//2:T])**2)

        #Dual ascent
        lambda_hat[n,:] = lambda_hat[n-1,:] + tau*(np.sum([ u_hat_plus[i][n,:] for i in range(K)],0) - f_hat_plus)
        #loop counter
        n = n+1
        #converged yet?
        uDiff = eps
        for i in range(K):
            uDiff = uDiff + 1/T*(u_hat_plus[i][n-1,:]-u_hat_plus[i][n-2,:])*np.mat((u_hat_plus[i][n-1,:]-u_hat_plus[i][n-2,:]).conjugate()).H
        uDiff = np.abs(uDiff)


    # ------ Postprocessing and cleanup-------

    #discard empty space if converged early
    N = min(N,n)
    omega = omega_plus[0:N,:]
    #Signal reconstruction
    u_hat = np.zeros((T, K), dtype=complex)
    temp = [u_hat_plus[i][N-1,T//2:T] for i in range(K) ]
    u_hat[T//2:T,:] = np.squeeze(temp).T

    temp = np.squeeze(np.mat(temp).conjugate())
    u_hat[1:(T//2+1),:] = temp.T[::-1]

    u_hat[0,:] = (u_hat[-1,:]).conjugate()

    u = np.zeros((K,len(t)))

    for k in range(K):
        u[k,:]=np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))
    #remove mirror part
    u = u[:,T//4:3*T//4]
    #recompute spectrum
    u_hat = np.zeros((T//2, K), dtype=complex)
    for k in range(K):
        u_hat[:,k]= np.squeeze( np.mat( np.fft.fftshift(np.fft.fft(u[k,:])) ).H)

    return u, u_hat, omega 
    
def integ2d(tfr,*params):
    [M,N]=tfr.shape;
    if (len(params)==0):
        t = np.arange(0,N,1,dtype = int);
        f = np.arange(0,M,1,dtype = int);
    elif(len(params)==1):
        t = params[0];
        f = np.arange(0,M,1,dtype = int);
    elif(len(params)==2):
        t = params[0];
        f = params[1];

    t=t.reshape(1,t.size);
    f=f.reshape(1,f.size);
    f=np.sort(f);
    #这里开始是原函数的integ2d函数
    [Mt,Nt]=t.shape;
    [Mf,Nf]=f.shape;
    
    if((Mt>Nt)&(Mt!=1)):
        raise Exception('ERROR(function_integ2ddef 00x1):t must be a row-vector');
    elif((Mf>Nf)&(Mf!=1)):
        raise Exception('ERROR(function_integ2ddef 00x2):f must be a row-vector');
    elif(N!=Nt):
        raise Exception('ERROR(function_integ2ddef 00x3):Mat must have as many colums as t');
    elif(M!=Nf):
        raise Exception('ERROR(function_integ2ddef 00x4):Mat must have as many rows as f');
        
    #Renyi_part1=np.sum(tfr,1);
    Renyi_part1=np.zeros([1,M], dtype=np.double);
    for i in range(M):
        Renyi_part1[0,i]=np.sum(abs(tfr[i,:]));
    
    Renyi_part2=tfr[:,0]/2;
    Renyi_part3=tfr[:,N-1]/2;
    
    Renyi_part=(Renyi_part1-Renyi_part2-Renyi_part3)*(t[0,1]-t[0,0]);
    
    
    dtfr=Renyi_part[0,0:M-1]+Renyi_part[0,1:M];
    df=(-f[0,0:M-1]+f[0,1:M])/2;
    
    som=sum(dtfr*df);
    #som=450*sum(dtfr*df);
    return som;    
    
#计算瑞利熵的函数，还存在一丢丢的问题
def Renyi(tfr,*params):
    eps=2.2204e-16;
    [M,N]=tfr.shape;
    if (len(params)==0):
        t = np.arange(0,N,1,dtype = int);
        f = np.arange(0,M,1,dtype = int);
        alpha = 3;
    elif(len(params)==1):
        t = params[0];
        f = np.arange(0,M,1,dtype = int);
        alpha = 3;
    elif(len(params)==2):
        t = params[0];
        f = params[1];
        alpha = 3;
    elif(len(params)==3):
        t = params[0];
        f = params[1];
        alpha = params[3];
    tfr=tfr/integ2d(tfr);
    if (alpha == 1):
        if(tfr.min()<0):
            raise Exception('ERROR(function_renyidef 00x1):distribution with negative values => alpha=1 not allowed');
        else:
            r=-integ2d(tfr*np.log2(tfr+eps),t,f);
    else:
        r=-np.log2(integ2d(abs(tfr)**alpha,t,f)+eps)/(1+alpha);
    
    return r;
    

#这里提供了sqSTFT一个测试函数的用例

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


[tfr, tfrtic, tfrsq, tfrsqtic] = sqSTFT(x, lowFreq, highFreq, alpha, tDS, h, Dh);
imageSQ(t, tfrsqtic, tfrsq)
#imageSQ(t, tfrsqtic, tfr)
rx=Renyi(abs(tfrsq))


#这里提供了一个SEO-S测试函数的用例
'''
t=np.linspace(0,3,1024);
t=t.reshape(t.size,1);
#x=np.sin(10*math.pi*t*t*t+25*math.pi*t);
x=np.sin(5*math.pi*t*t*t+25*math.pi*t)+np.sin(10*math.pi*t*t*t+30*math.pi*t);
SampFreq=80;

n=t.size;
fre_start=(SampFreq/2)/(n/2);
fre_ma=(SampFreq/2)/(n/2);
fre_end=(SampFreq/2)+(SampFreq/2)/(n/2);
fre=np.arange(fre_start,fre_end,fre_ma,dtype = float);
fre=fre.reshape(1,fre.size);

hlength=55;
IF,Te,tfr=SET_Y(x,hlength);#aspect=0.07

minfreq=0
maxfreq=512;
samplingrate=1;
freqsamplingrate=1;

st,tx,f=gst(x);
GST_SEO=st[0:512,:]*IF;
x, y = t, f[0,0:512]
z = GST_SEO.real;
plt.figure();
plt.imshow(z+100, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
           norm=LogNorm(), origin ='lower', aspect=5)
plt.title('SEO-S')
plt.show()
rx=Renyi(z,t)
'''


#这里提供了一个WVD测试函数的用例
#交叉项方面也很好的被表现了出来，因此对于WVD来说进行交叉项的抑制的算法也是同样可以直接进行的
'''
t=np.linspace(0,3,1024);
t=t.reshape(t.size,1);

#x=np.sin(5*math.pi*t*t*t+25*math.pi*t)+np.sin(10*math.pi*t*t*t+100*math.pi*t);
x=np.sin(5*math.pi*t*t*t+25*math.pi*t)+np.sin(10*math.pi*t*t*t+30*math.pi*t);

FreqBins=256;
SampFreq=1;
WVD,tx,f=WignerVille(x,SampFreq,FreqBins);

x, y = t, f
z = abs(WVD);
plt.figure();
plt.imshow(z+10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
           norm=LogNorm(), origin ='lower', aspect=5)
plt.title('WVD')
plt.show()
rx=Renyi(z,t)
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
rx=Renyi(z,t)
'''


'''
#分段函数的定义
sampling_rate=1024;
t = np.arange(0,1.0,1.0/sampling_rate)
f1 = 100#频率
f2 = 200
f3 = 300
data = np.piecewise(t,[t<1,t<0.8,t<0.3],
                    [lambda t : np.sin(2*np.pi *f1*t),
                     lambda t : np.sin(2 * np.pi * f2 * t),
                     lambda t : np.sin(2 * np.pi * f3 * t)])
[cwtmatr, frequencies] = pywt.cwt(data,scales,wavename,1.0/sampling_rate)#连续小波变换
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(t, data)
plt.xlabel(u"time(s)")
plt.title(u"300Hz 200Hz 100Hz Time spectrum")
plt.subplot(212)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)
plt.show()

'''

    
    
