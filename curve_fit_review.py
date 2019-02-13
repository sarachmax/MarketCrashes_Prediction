import pandas as pd 
import numpy as np 
from scipy.optimize import curve_fit  
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from random import uniform, randint
from GA_initparam import GeneticAlgo 

ga = GeneticAlgo() 

# Log periodic function, as per https://arxiv.org/abs/cond-mat/0201458v1
def y(x, o, m, A, B, C, tau):
    ''' Target Log-Periodic Power Law (LPPL) function
        Note: Scaled 'B' -> -1.0
        TODO: Perhaps we should check that parameters passed result in
              reasonable computations
    ''' 
    ret = A + B*tau**m * x**m + C * tau**m * x**m * np.cos(o * np.log(x))
    return ret

def init_fit_params(data): 
    # Fit parameters 
    sp = np.mean(data)

    # o = np.std(data) * 200  # Frequency = omega 
    # m = np.std(data) 
    # A = sp     # Intercept
    # B = -(np.mean(data)/np.median(data))      
    # C = np.std(data)/np.mean(data) # Coefficient
    # t = abs(np.argmax(data)-np.argmin(data))/2 # Critical time

    # o = 10  # Frequency = omega 
    # m = 0.5   # Power
    # A = sp     # Intercept
    # B = -1    
    # C = -0.05
    # t = 50   # Critical time
    o, m, A, B, C, t = ga.generate_params(data,50)
    print("Params: Start         \no:", o, 
                                "\nm:",  m, 
                                "\nA:",  A, 
                                "\nB:", B, 
                                "\nC:",  C,
                                "\ntau:", t)
    p0 = [o, m, A, B, C, t]
    return p0 

def is_pass_params_rules(params):
    """
    0.1 ≤ m ≤ 0.9, 6 ≤ omega ≤ 13, |C| < 1, B < 0.
    """
    o, m, A, B, C, tau = params[0], params[1], params[2], params[3], params[4], params[5] 
    if abs(m) > 1 : 
        return False 
    if o < 0 or o > 40 : 
        return False 
    if abs(C) > 1 : 
        return False 
    if B > 0 :
        return False 
    if A < 0 :
        return False 
    if int(tau) < 1 or tau >= 220 :
        return False 
    return True 

def curve_fitting(yd, maxfev=1000000, p0=None):
    if p0 == None : 
        p0 = init_fit_params(yd)
    try : 
        popt, pcov = curve_fit(y, xd, yd, p0=p0, maxfev=maxfev, check_finite=False) 
    except : 
        print("Cannot find fitted parameters ")   
        popt = [None, None, None, None, None, None]         
    return popt 

def get_local_minmax(data, n=55):
    lmin = argrelextrema(data, np.less_equal, order=n)[0]
    lmax = argrelextrema(data, np.greater_equal, order=n)[0]
    return lmin, lmax

def find_nearest(array, value, start):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin() 
    for i in range(idx,len(array)):
        if array[i] >= start : 
            return array[i]
    return array[idx]

dataset = pd.read_csv("Dataset/SEHK_1D_2800.csv")
fit_df = pd.DataFrame(columns=['winsize','t','o', 'm', 'A', 'B', 'C', 'tau','near_min','near_max','pred_err_max','pred_err_min','mse','pass','figname'])
    
data = dataset['Close']
data = np.array(data)
data = np.log(data)

day_size = 1
window_sizes = [120, 240, 480, 900] 
predict_size = 220
start = np.max(window_sizes)+1
stop = len(data) - predict_size
jump_size = day_size*5
p0={}
st_p0 = True 
for i in range(start, stop, jump_size):
    for size in window_sizes : 
        if st_p0 : 
            p0[size] = None
        xd = np.linspace(0.1, size, size) 
        yd = data[i-size:i] 
        print("Curve Fitting process has been started.") 
        params_fit = curve_fitting(yd, p0=p0[size])
        o, m, A, B, C, tau = params_fit[0], params_fit[1], params_fit[2], params_fit[3], params_fit[4], params_fit[5]
        # params_fit_new = curve_fitting(yd, p0=None)
        # on, mn, An, Bn, Cn, taun = params_fit_new[0], params_fit_new[1], params_fit_new[2], params_fit_new[3], params_fit_new[4], params_fit_new[5]
        # if tau != None and taun != None :
        #     y1 = y(xd, o, m, A, B, C, tau)
        #     y2 = y(xd, on, mn, An, Bn, Cn, taun)
        #     _mse1 = mean_squared_error(yd, y1)
        #     _mse2 = mean_squared_error(yd, y2) 
        #     if _mse2 < _mse1 : 
        #         o, m, A, B, C, tau = on, mn, An, Bn, Cn, taun 
        # if tau == None and taun != None : 
        #     o, m, A, B, C, tau = on, mn, An, Bn, Cn, taun
        if tau != None : 
            curve_result = y(xd, o, m, A, B, C, tau)
            mse = mean_squared_error(yd, curve_result)
            pred_size = int(size+tau+1)
            try : 
                x_pred = np.linspace(0.1,pred_size, pred_size)
            except : 
                x_pred = xd 
            y_pred = y(x_pred, o, m, A, B, C, tau)
            if abs(np.min(curve_result)-np.max(curve_result)) > 0.001 and np.mean(curve_result) != np.max(curve_result) and tau<=pred_size:                        
                # print("MSE : ", mse,'\n')
                # if is_pass_params_rules(params_fit):
                true_data = data[i-size:i+predict_size]
                lmin, lmax = get_local_minmax(true_data, n=20) 
                lmin, lmax = np.array(lmin), np.array(lmax)
                near_min = find_nearest(lmin, tau+size, size)
                near_max = find_nearest(lmax, tau+size, size)
                pass_rule = is_pass_params_rules(params_fit)
                plt.figure(figsize=(16,9))
                title = "Fitting between "+ str(i-size) + " to " + str(i) + " window : "+str(size)
                plt.title(title)
                plt.plot(true_data, color='black', label='y_true')
                plt.plot(y_pred, color='blue', label='y_curve')
                plt.axvline(near_max, color='black', label='Nearest Peak') 
                plt.axvline(near_min, color='red', label='Nearest Minimum')  
                plt.axvline(tau+size, color='blue', label='tc_pred')
                plt.axvline(size, color='yellow', label='trained_point')
                plt.legend()
                # plt.show()
                figname = "win_"+ str(size)+ "_from_"+str(i-size) + "_to_" + str(i) + '.png'
                plt.ylim(np.min(true_data)-0.1, np.max(true_data+0.1))
                plt.savefig("CurveFitResult/img/"+figname, dpi=300)
                print("Saved img to : " + figname)
                fit_df = fit_df.append( {'o':  o, 
                                    'm':  m, 
                                    'A':  A, 
                                    'B':  B, 
                                    'C':  C,
                                    'winsize': size,
                                    't': i,
                                    'near_min' : near_min+i,
                                    'near_max' : near_max+i,
                                    'pred_err_max' : near_max-(tau+size),
                                    'pred_err_min' : near_min-(tau+size),
                                    'mse': mse,
                                    'pass': pass_rule,
                                    'figname': figname,
                                    'tau' :  tau+i},ignore_index=True)
                print("#"*20)
                print("\nParams: fitted \no:", o, 
                                "\nm:",  m, 
                                "\nA:",  A, 
                                "\nB:", B, 
                                "\nC:",  C,
                                "\ntau:", tau,
                                "\npass:", pass_rule,
                                "\nmse:", mse,
                                "\n" +'^'*20)
                
                p0[size] = [o, m, A, B, C, tau] 
    if st_p0 : 
        st_p0 = False 
fit_df.to_csv('CurveFitResult/fitting_log.csv')
print("Fitting done !")

print("Summarize")
abs_pred_err = (abs(fit_df['pred_err_max']))
print("Average Error : ", abs_pred_err.mean())

fit_df = pd.read_csv('CurveFitResult/fitting_log.csv')
t_train = fit_df['t']
tc_pred = fit_df['tau']
plt.title("Review t vs tc")
plt.figure(figsize=(16,9))
plt.scatter(t_train, tc_pred)
plt.xlabel('t_train')
plt.ylabel('tc')
# plt.show() 
figname = 't_crash.png'
plt.savefig("CurveFitResult/"+figname, dpi=300)