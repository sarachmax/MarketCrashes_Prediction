
import numpy as np
import scipy.stats as scs
from scipy.optimize import curve_fit, basinhopping
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.cluster import SpectralClustering, KMeans  

"""
LPPL  

E[ln p(t)] = A + B(tc -t)^m + (C*(tc-t)^m)*cos(omega*ln(tc-t) - phi) 

The LPPL model is described by 3 linear parameters (A,B,C) and 4 nonlinear parameters
(m, omega, tc, phee). These parameters are subjected to the following constrains. Since the
integral of the hazard rate  over time up to t = tc gives the probability of the occurrence
of a crash, it should be bounded by 1, which yields the condition m < 1. At the same time,
the log-price  should also remain finite for any t ≤ tc, which imply the other condition
m > 0. In addition, the requirement of the existence of an acceleration of the hazard rate as
time converges towards tc implies B < 0. Additional constraints emerge from a compilation
of a significant number of historical bubbles that can be summarized as follows:
    
        0.1 ≤ m ≤ 0.9, 6 ≤ omega ≤ 13, |C| < 1, B < 0.

"""

# Log periodic function, as per https://arxiv.org/abs/cond-mat/0201458v1
def y(x, o, m, A, B, C, tau):
    ''' Target Log-Periodic Power Law (LPPL) function
        Note: Scaled 'B' -> -1.0
        TODO: Perhaps we should check that parameters passed result in
              reasonable computations
    '''
    ret = A + B*tau**m * x**m + C * tau**m * x**m * np.cos(o * np.log(x))
    return ret

class MyStepper(object):
    ''' Implements some simple reasonable modifications to parameter stepping

            * Frequencies are kept positive
            * Powers are exponentially suppressed from getting 'large'
            * 
    '''
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize

    def __call__(self, x):
        ''' Updater for stepping x[*] parameters
        '''

        s = self.stepsize

        # relative parameter scales o/m, etc.
        scale_om = 10.0
        scale_Am = 100.0
        scale_Cm = 100.0
        scale_tm = 10.0

        # Treat the frequency differently -- only interested in positive
        # frequency values.
        inc = scale_om * np.random.uniform(-s, s)

        if (x[0] + inc <= 0):
            x[0] += abs(inc)
        else:
            x[0] += inc

        # Treat the power differently -- large powers (>>1) can easily cause
        # numerical problems, so we exponentially penalize steps in the power
        # parameter 'm'.
        x[1] += np.exp(-abs(x[1])) * np.random.uniform(-s, s)

        x[2] += scale_Am * np.random.uniform(-s, s)
        x[3] += scale_Cm * np.random.uniform(-s, s)

        # Treat 'critical time' differently
        inc = scale_tm * np.random.uniform(-s, s)

        if (x[4] + inc <= 0):
            x[4] += abs(inc)
        else:
            x[4] += inc

        return x


class MyBounds(object):
    ''' This class implements a set of reasonable bounds (depends on units and
        domain knowledge):
            * frequencies less than 200
            * powers less than 1
            * intercept/coefficient less than 1e4
            * critical time
    '''
    #                        o, m, A, B, C, tau
    def __init__(self, xmax=[13, 0.1, 10, 0, 1.0, 550],
                       xmin=[6, 0.9, 0, -1e4, -1.0, 11]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        ''' Evaluater -- Returns True if we accept the solution, False
            otherwise.
        '''
        x = kwargs["x_new"]

        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))

        # Only accept solutions within all parameter bounds
        return tmax and tmin


def init_fit_params(data): 
    # Fit parameters 
    sp = data[-1]
    o = 13    # Frequency = omega 
    m = 0.5    # Power
    A = sp     # Intercept
    B = -1      
    C = 0.5     # Coefficient
    t = 20.0    # Critical time
    p0 = [o, m, A, B, C, t]
    return p0 

# Set up our 'energy'/loss function -- Mean squared distance
def E_func(params):
    ret = y(xd, params[0], params[1], params[2], params[3], params[4], params[5])

    n = float(len(ret))
    try:
        er = (ret - yd).dot(ret - yd) / n
    except : 
        pass 
        
    if np.isnan(er):
        er = 1e10
    return er

def get_critrical_label(params):
    if B > 0 : 
        return 1
    return -1 
    
    

if __name__ == '__main__':
    dataset = pd.read_csv("Dataset/SEHK_30Min_2800.csv")
    fit_df = pd.DataFrame(columns=['winsize','o', 'm', 'A', 'B', 'C', 'tau','raw_tau'])
    raw_data = dataset.Close

    data = dataset['Close'].ewm(span=55,min_periods=0,adjust=False,ignore_na=False).mean()
    data = np.array(data)
    data = np.log(data)

    print('========== Heuristic Fitting ============') 

    # Step size can greatly affect the solution/convergence.
    step_size = 0.1 #learning rate 
    mystep = MyStepper(step_size)
    mybound = MyBounds()

    # window_sizes = [1100, 2200, 3300, 4400, 5500, 6600] 
    window_sizes = [2200] 
    mean_crash_point = [] 
    all_crash_point = [] 
    crash_point = {} 
    for size in window_sizes :
        crash_point[size] = []
    start = np.max(window_sizes)
    stop = len(data)
    jump_size = np.min(window_sizes)
    jump_size = 11*60
    if jump_size >= 1000 : 
        jump_size = int(jump_size/2)
    max_windows = np.max(window_sizes)
    num_windows = len(window_sizes)

    for i in range(start, stop, jump_size):
        c_point = [] 
        for size in window_sizes : 
            xd = np.linspace(0.1, size, size)
            yd = data[i-size:i]
            
            p0 = init_fit_params(yd) 
            try :
                ret = basinhopping(E_func, p0, take_step=mystep, accept_test=mybound, disp=False)
                params_fit = ret.x 
                o, m, A, B, C, tau = params_fit[0], params_fit[1], params_fit[2], params_fit[3], params_fit[4], params_fit[5]
                c_time = int(tau) + i
                print("\n"*3)
                print("Fitting on window size :", size)
                print("Fitting on data point on " + str(i-max_windows) + " to " + str(i) + " of " + str(stop))
                print("Global Minimum: E(params_fit) = {}".format(ret.fun))
                print("Params: params_fit   '\no':", o, 
                                            "\nm:",  m, 
                                            "\nA:",  A, 
                                            "\nB:", B, 
                                            "\nC:",  C,
                                            "\ntau:", tau,
                                            "\nc_time:", c_time)      
                print("\n"*3)                                
                crash_point[size].append(c_time)
                c_point.append(c_time)
                all_crash_point.append(c_time)
                fit_df = fit_df.append({    'o':  o, 
                                            'm':  m, 
                                            'A':  A, 
                                            'B':  B, 
                                            'C':  C,
                                            'winsize': size,
                                            'raw_tau': int(tau),
                                            'tau' :  c_time}, ignore_index=True)
                median_point = np.median(c_point) 
                if not np.isnan(median_point) : 
                    mean_crash_point.append(median_point) 
            except : 
                print("No fitting point on " + str(i) + " to " + str(i+max_windows))
                
    fit_df.to_csv("Dataset/fitted_result_basin.csv",index=False)

    print('======= Clustering ============')
    k_mean = False

    n_clusters = 2

    X = fit_df.drop(columns=['tau', 'winsize','raw_tau'])
    if not k_mean : 
        clustering = SpectralClustering(n_clusters=n_clusters,
            assign_labels="discretize",
            random_state=0).fit(X)
        y = clustering.labels_
        fit_df['label'] = y 
    else : 
        clustering = KMeans(n_clusters=n_clusters).fit_predict(X)
        fit_df['label'] = clustering
    print('======= Clustering End ========')
    
    plot_all = True 
    plot_mean = False 
    plot_each_window= False   
    plot_clustering = False 
    
#    plt.figure()
#    plt.plot(data)
#    plt.show();
    
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