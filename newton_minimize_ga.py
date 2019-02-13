import pandas as pd 
import numpy as np 
from scipy.optimize import curve_fit , minimize
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from random import uniform
from GA_initparam import GeneticAlgo

ga_generator = GeneticAlgo() 

def sum_squared_error(y_true, y_pred):
    cost = 0 
    for i in range(len(y_true)):
        cost += (y_true[i]-y_pred[i]) ** 2
    return cost 

def y(x, o, m, A, B, C, tau):
    ''' Target Log-Periodic Power Law (LPPL) function
        Note: Scaled 'B' -> -1.0
        TODO: Perhaps we should check that parameters passed result in
              reasonable computations
    ''' 
    ret = A + B*tau**m * x**m + C * tau**m * x**m * np.cos(o * np.log(x))
    return ret

def y_minimize(params, x, y):
    o, m, A, B, C, tau = params[0], params[1], params[2], params[3], params[4], params[5]
    ret = A + B*tau**m * x**m + C * tau**m * x**m * np.cos(o * np.log(x))
    return sum_squared_error(y, ret)
    
def init_fit_params(data): 
    # Fit parameters 
    sp = np.mean(data)

    # o = np.std(data) * 200  # Frequency = omega 
    # m = np.std(data) 
    # A = sp     # Intercept
    # B = -(np.mean(data)/np.median(data))      
    # C = np.std(data)/np.mean(data) # Coefficient
    # t = abs(np.argmax(data)-np.argmin(data))/2 # Critical time

    # o = 11  # Frequency = omega 
    # m = 0.5   # Power
    # A = sp     # Intercept
    # B = -1      
    # C = 0.5
    # t = 110   # Critical time
    o, m, A, B, C, t = ga_generator.generate_params(data, 100)
    print("Params: init         \no:", o, 
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
    if int(tau) < 11 or tau >= 480 :
        return False 
    return True 

def fitting_params(yd, p0=None):
    if p0 == None : 
        p0 = init_fit_params(yd)
    try : 
        bounds=((6, 0.1, 1, -1,-1000, 0.01),(40, 0.9, 10, 1,0,220))
        res = minimize(y_minimize, p0, (xd,yd),method='Nelder-Mead', bounds=bounds, options={'disp': True})
        params = res.x
    except : 
        print("Cannot find fitted parameters ")   
        params = [None, None, None, None, None, None]         
    return params 

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
window_sizes = [60,120,240,480,900] 
predict_size = 240
start = np.max(window_sizes)+1
stop = len(data) - predict_size
jump_size = day_size*1
for i in range(start, stop, jump_size):
    for size in window_sizes : 
        xd = np.linspace(0.1, size, size) 
        yd = data[i-size:i] 
        print("Curve Fitting process has been started.") 
        # params_fit = curve_fitting(yd, p0=p0[size])
        params_fit = fitting_params(yd)
        o, m, A, B, C, tau = params_fit[0], params_fit[1], params_fit[2], params_fit[3], params_fit[4], params_fit[5]
        if tau != None : 
            curve_result = y(xd, o, m, A, B, C, tau)
            mse = mean_squared_error(yd, curve_result)
            x_pred = np.linspace(0.1, size+int(tau)+1, size+int(tau)+1)
            y_pred = y(x_pred, o, m, A, B, C, tau)
            if tau+size <= size+predict_size :                        
                print("MSE : ", mse,'\n')
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
                plt.axvline(near_max, color='black', label='Nearest Lowest') 
                plt.axvline(tau+size, color='blue', label='tc_pred')
                plt.axvline(size, color='yellow', label='trained_point')
                plt.ylim(np.min(true_data)-0.1, np.max(true_data)+0.1)
                plt.xlim(0, len(true_data))
                plt.legend()
                # plt.show()
                figname = "win_"+ str(size)+ "_from_"+str(i-size) + "_to_" + str(i) + '.png'
                plt.savefig("GAFitResult/img/"+figname, dpi=300)
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
                                "\n" +'^'*20)
                

fit_df.to_csv('GAFitResult/fitting_log.csv')
print("Fitting done !")

print("Summarize")
abs_pred_err = (abs(fit_df['pred_err_max']))
print("Average Error : ", abs_pred_err.mean())

fit_df = pd.read_csv('GAFitResult/fitting_log.csv')
t_train = fit_df['t']
tc_pred = fit_df['tau']
plt.title("Review t vs tc")
plt.figure(figsize=(16,9))
plt.scatter(t_train, tc_pred)
plt.xlabel('t_train')
plt.ylabel('tc')
# plt.show() 
figname = 't_crash.png'
plt.savefig("GAFitResult/"+figname)