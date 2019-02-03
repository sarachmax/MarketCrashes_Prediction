# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:39:30 2019

@author: sarac
"""

import numpy as np
import scipy.stats as scs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.cluster import SpectralClustering, KMeans 


# Log periodic function
def y(x, o, m, A, B, C, tau):
    return A - B * tau**m * x**m + C * tau**m * x**m * np.cos(o * np.log(x))


def res_fun(params, x, y_fit):
    return y(x, params[0], params[1], params[2], params[3], params[4],
             params[5]) - y_fit


# Log periodic function #2
def y_2(x, o, m, tau):
    ''' Function of the nonlinear parameters o, m, tau only.  Solves
        for the optimal 'slave' parameters X_hat(o, m, tau)
        OUTPUT: Function value
    '''
    A_hat, B_hat, C_1_hat, C_2_hat = minABC_y(xd, o, m, tau)

    return A_hat \
           + B_hat * (tau - x)**m \
           + C_1_hat * (tau - x)**m * np.cos(o * np.log(tau - x)) \
           + C_2_hat * (tau - x)**m * np.sin(o * np.log(tau - x))

def res_fun_2(params, x, y_fit):
    ''' Residual function -- required for the minimization problem
        OUTPUT: Residual
    '''
    return y_2(x, params[0], params[1], params[2]) - y_fit


def minABC_y(x, o, m, tau):
    ''' Solves for the 'slave' variables A, B, C_1, C_2
        INPUT: Nonlinear parameters o, m, tau
        OUTPUT: Returns linear parameter solutions (A, B, C_1, C_2)
    '''
    yi = yd
    fi = (tau - x)**m
    gi = (tau - x)**m * np.cos(o * np.log(tau - x))
    hi = (tau - x)**m * np.sin(o * np.log(tau - x))
    N = float(len(x))

    mat = np.array([[N, np.sum(fi), np.sum(gi), np.sum(hi)],
                    [np.sum(fi), fi.dot(fi), fi.dot(gi), fi.dot(hi)],
                    [np.sum(gi), fi.dot(gi), gi.dot(gi), gi.dot(hi)],
                    [np.sum(hi), fi.dot(hi), gi.dot(hi), hi.dot(hi)]])

    s_vec = np.array([np.sum(yi),
                      yi.dot(fi),
                      yi.dot(gi),
                      yi.dot(hi)]).reshape(-1, 1)
    vec_hat = np.linalg.inv(mat).dot(s_vec)

    return tuple(vec_hat.flatten())

def init_fit_params(data): 
    # Fit parameters 
    o = 20.0    #omega
    m = 1.0
    A = 1.0
    B = -1.0
    C = 0.5
    t = 3.0     #tc 
    sp = data[-1]
    p0 = (sp, m, A, B, C, t)
    return p0 

if __name__ == '__main__':
    # dataset 
#    dataset = pd.read_csv("Dataset/SET_1D.csv")
    dataset = pd.read_csv("Dataset/SEHK_30Min_2800.csv")
    fit_df = pd.DataFrame(columns=['winsize','o', 'm', 'A', 'B', 'C', 'tau','raw_tau'])
    
    raw_data = dataset.Close
    data = np.array(dataset.Close)
    data = np.log(data)
    
    
    window_sizes = [1100, 2200, 3300, 4400, 5500, 6600] 
    mean_crash_point = [] 
    all_crash_point = [] 
    crash_point = {} 
    for size in window_sizes :
        crash_point[size] = []
    start = np.max(window_sizes)
    stop = len(data) 
    jump_size = np.min(window_sizes)
    if jump_size >= 1000 : 
        jump_size = int(jump_size/2)
    max_windows = np.max(window_sizes)
    num_windows = len(window_sizes)
    for i in range(start, stop, jump_size):
        c_point = [] 
        print("Fitting on data point on " + str(i) + " to " + str(i+max_windows))
        for size in window_sizes : 
            xd = np.linspace(0.1, size, size)
            yd = data[i-size:i]
            
            p0 = init_fit_params(yd)
            maxfev = 5000
            while True : 
                try :
                    popt, pcov = curve_fit(y, xd, yd, p0=p0, maxfev=maxfev)
                    break 
                except KeyboardInterrupt : 
                    break
                except : 
                    if maxfev >= 20000 :
                        print("No fitting point on " + str(i) + " to " + str(i+max_windows))
                        break 
                    else : 
                        maxfev = 20000
            c_time = int(popt[5]+i+size)
            if c_time > i + size  and c_time < stop: 
                crash_point[size].append(c_time)
                c_point.append(c_time)
                all_crash_point.append(c_time)
                fit_df = fit_df.append({'o':  popt[0], 
                                            'm':  popt[1], 
                                            'A':  popt[2], 
                                            'B':  popt[3], 
                                            'C':  popt[4],
                                            'winsize': size,
                                            'raw_tau': int(popt[5]),
                                            'tau' :  c_time},ignore_index=True)
            median_point = np.median(c_point) 
            if not np.isnan(median_point) : 
                mean_crash_point.append(median_point)


    print('======= Clustering ============')
    n_clusters = 4


    X = fit_df.drop(columns=['A','tau', 'winsize'])
    clustering = SpectralClustering(n_clusters=n_clusters,
        assign_labels="discretize",
        random_state=0).fit(X)
    y = clustering.labels_
    fit_df['label'] = y 
    
    # clustering = KMeans(n_clusters=n_clusters).fit_predict(X)
    # fit_df['label'] = clustering
    print('======= Clustering End ========')
    
    plot_all = False 
    plot_mean = False
    plot_each_window= False 
    plot_clustering = True 
    
    if plot_all : 
        survive_time = [] 
        mem = [] 
        for x in all_crash_point : 
            if not (x in mem):
                mem.append(x)
            else : 
                if not (x in survive_time):
                    survive_time.append(x) 
        plt.figure()         
        plt.title("Crash point all")
        plt.plot(raw_data, label='prices')
        for x in all_crash_point :
            plt.axvline(x, color='red')
        plt.legend()
        plt.show();
    
    if plot_mean :     
        survive_time = [] 
        mem = [] 
        for x in mean_crash_point : 
            if not (x in mem):
                mem.append(x)
            else : 
                if not (x in survive_time):
                    survive_time.append(x) 
        plt.figure(figsize=(1920,1080), dpi=80) 
        plt.title("Crash point mean")
        plt.plot(raw_data, label='prices')
        for x in survive_time :
            plt.axvline(x, color='green')
        plt.legend()
        plt.show();
    
    if plot_each_window :     
        for size in window_sizes : 
            plt.figure()
            title = 'Fit on '+ str(size) + ' windows size'
            plt.title(title)
            plt.plot(raw_data)
            for x in crash_point[size] :
                plt.axvline(x, color='red')
            plt.show()

    if plot_clustering : 
        for group_num in range(n_clusters): 
            plt.figure()
            plt.plot(raw_data)
            title = 'Group  ' + str(group_num) 
            plt.title(title)
            for i in range(len(fit_df)):
                if fit_df['label'][i] == group_num : 
                    x = fit_df['tau'][i]
                    plt.axvline(x, color='blue')    
            plt.show()  

            
                
     

        
            
        
            

    
