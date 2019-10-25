# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:52:04 2019

@author: lumir
"""
import numpy as np
import matplotlib.pyplot as plt
import random


a = 0.2

theta = np.linspace(-10,10,1000)

variance = [1, 0.0001]

for i in range(len(variance)):
    alpha = variance[i] / theta ** 2


    KL_std =  0.5 * np.log(1+ 1 / alpha )

    plt.plot(theta, KL_std)
    plt.plot(theta-a, KL_std)
    plt.plot(theta+a, KL_std)
    plt.xlim((-1,1))
    plt.title('variance = {}'.format(variance[i]))
    plt.show()

a = 3
theta = 0
variance = 1
q = np.random.normal(theta, variance ** (0.5), 3000)
p1 = np.random.normal(theta, variance ** (0.5), 1000)
p2 = np.random.normal(theta+a, variance ** (0.5), 1000)
p3 = np.random.normal(theta-a, variance ** (0.5), 1000)
p = np.concatenate((p1,p2,p3),0)
plt.hist(p,bins=50,color='red')
plt.show()
plt.hist(q,bins=50,color='blue')
plt.show()

def gaussian_dist(x, mean = 0, std_err = 1):
    y = np.exp(-1 * 0.5 * (x-mean) ** 2 / std_err ** 2 ) / (np.sqrt(2 * np.pi) * std_err)
    return y

def GMM(x,scale_num = 3, mean = [1,0,-1], std = [1,1,1]):
    y = 0
    for i in range(scale_num):
        y += 1 / scale_num * gaussian_dist(x,mean[i],std[i])
    return y

def MCMCforTwoGauss(para1 = [0,1], para2 = [3,2],sample_num = 30000):
    N = sample_num
    mean1 = para1[0]
    mean2 = para2[0]
    std1 = para1[1]
    std2 = para2[1]

    x_sample = np.random.normal(mean1,std1,N)

    f_sample = gaussian_dist(x_sample,mean1,std1)
    g_sample = gaussian_dist(x_sample,mean2,std2)

    D_mc = 0
    for i in range(N):
        D_mc += np.log(f_sample[i]/g_sample[i])
    D_mc = 1/N * D_mc
    print(D_mc) 

def KL4twogauss(para1 = [0,1], para2 = [3,2]):
    mean1 = para1[0]
    mean2 = para2[0]
    std1 = para1[1]
    std2 = para2[1]
    KL = np.log(std2/std1) + (std1**2 + (mean1 - mean2)**2)/(2 * std2 **2) - 0.5
    return KL

def MCforGaussandGMM(GMM_scale_num = 3,GMM_mean = [1,0,-1],GMM_std = [1,1,1],
                     Gauss_para =[0,1],sample_num = 30000):
    N = sample_num
    gs_mean = Gauss_para[0]
    gs_std =  Gauss_para[1]
    x_sample = np.random.normal(gs_mean,gs_std,N)
    
    Gauss_sample = gaussian_dist(x_sample,gs_mean,gs_std)
    GMM_sample = GMM(x_sample,GMM_scale_num,GMM_mean,GMM_std)
    
    D_mc = 0
    for i in range(N):
        D_mc += np.log(Gauss_sample[i]/GMM_sample[i])
    D_mc = 1/N * D_mc
    print(D_mc) 
    
def samplefromGMM(N = 30000, 
                  GMM_scale_num = 3,GMM_mean = [1,0,-1],GMM_std = [1,1,1]):
    
    if np.mod(N,GMM_scale_num):
        raise ValueError('Sample numbers cannot be well divided by GMM scale numbers!')
    else:
        unit_sample = int(N/GMM_scale_num)
        p = np.zeros(N)
        
        for i in range(GMM_scale_num):
            mean = GMM_mean[i]
            std = GMM_std[i]
            p[i * unit_sample : (i+1) * unit_sample] = np.random.normal(mean, 
                                                              std, unit_sample)
    return p
        
p = samplefromGMM(GMM_mean = [3,0,-1])
plt.hist(p,bins=50,color='red')
plt.show()  
    

def MCforTwoGMM(GMM_scale_num1 = 3,GMM_mean1 = [1,0,-1],GMM_std1 = [1,1,1],
                GMM_scale_num2 = 3,GMM_mean2 = [1,0,-1],GMM_std2 = [1,1,1],
                sample_num = 30000):
    N = sample_num
    
    # sample from the fist GMM
    x_sample = samplefromGMM(N, GMM_scale_num1, GMM_mean1,GMM_std1)
    
    # 
    GMM_sample1 = GMM(x_sample,GMM_scale_num1,GMM_mean1,GMM_std1)
    GMM_sample2 = GMM(x_sample,GMM_scale_num2,GMM_mean2,GMM_std2)
    
    D_mc = 0
    for i in range(N):
        D_mc += np.log(GMM_sample1[i]/GMM_sample2[i])
    D_mc = 1/N * D_mc
    # print(D_mc)
    return D_mc
    
    
    
    
    
    
    MCforGaussandGMM(1,[3],[2],[0,1],3000)
    
    x_sample = np.linspace(-10,10,10000)
    y = GMM(x_sample,mean = [-2,0,2])
    plt.scatter(x_sample,y)


KL4gauss()