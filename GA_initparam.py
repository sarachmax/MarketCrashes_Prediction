import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error
from random import uniform, randint

def sum_squared_error(y_true, y_pred):
    cost = 0 
    for i in range(len(y_true)):
        cost += (y_true[i]-y_pred[i]) ** 2
    return cost 

# Log periodic function, as per https://arxiv.org/abs/cond-mat/0201458v1
def y(x, o, m, A, B, C, tau):
    ''' Target Log-Periodic Power Law (LPPL) function
        Note: Scaled 'B' -> -1.0
        TODO: Perhaps we should check that parameters passed result in
              reasonable computations
    ''' 
    ret = A + B*tau**m * x**m + C * tau**m * x**m * np.cos(o * np.log(x))
    return ret

class GeneticAlgo:
    def __init__(self):
        self.member = [] 
        self.df_columns = ['o', 'm', 'A', 'B', 'C', 'tc', 'mse']
        self.next_generation = pd.DataFrame(columns=self.df_columns) 
        
        # init max-min range 
        self.o_min, self.o_max = 6, 20  # Frequency = omega 
        self.m_min, self.m_max = 0.1, 0.9 
        self.A_min, self.A_max = 1, 10
        self.B_min, self.B_max = -100, 0
        self.C_min, self.C_max = -1, 1 
        self.tc_min,self.tc_max = 0.1, 20

    def generate_params(self, data, iter=1000):
        while True : 
            self.next_generation = pd.DataFrame(columns=self.df_columns) 
            for i in range(iter):
                selection_params, all_selection_params = self.selection_mechanism(data)
                breeding_params = self.breeding_mechanism(data, selection_params)
                mutation_params = self.mutaion_mechanism(selection_params, all_selection_params, data)
                self.culling_mechanism(selection_params, breeding_params, mutation_params)
            n = len(self.next_generation) - 1
            if n >= 2 : 
                choose = randint(1,n) 
                try : 
                    m = self.next_generation['m'][choose]  
                    A = self.next_generation['A'][choose]  
                    o = self.next_generation['o'][choose]  
                    B = self.next_generation['B'][choose]  
                    C = self.next_generation['C'][choose]  
                    tc = self.next_generation['tc'][choose]     
                    return o, m, A, B, C, tc
                except : 
                    pass

    def select_best_fit(self, params, n_best=25):
        params = params.sort_values(by=['mse'], ascending=False) 
        return params[:n_best+1] 

    """
    >>>>> 1 . Selection Mechanism >>>>>  
    """
    def init_selection_params(self):
        o = uniform(self.o_min, self.o_max)
        m = uniform(self.m_min, self.m_max)
        A = uniform(self.A_min, self.A_max)
        B = uniform(self.B_min, self.B_max)
        C = uniform(self.C_min, self.C_max)
        tc = uniform(self.tc_min, self.tc_max)
        return o, m, A, B, C, tc


    def selection_mechanism(self, data, n_select=25, n_random=50):
        n_xd = len(data)
        xd = np.linspace(0.1, n_xd, n_xd)
        params = pd.DataFrame(columns=self.df_columns)
        for i in range(n_random):
            o, m, A, B, C, tc = self.init_selection_params()
            y_fit = y(xd, o, m, A, B, C, tc)
            mse = sum_squared_error(data, y_fit)
            params = params.append({'o' : o, 
                                    'm' : m, 
                                    'A' : A, 
                                    'B' : B, 
                                    'C' : C, 
                                    'tc': tc, 
                                    'mse': mse}, ignore_index=True)
        if len(self.next_generation) < 2 :   
            best_params = self.select_best_fit(params, n_random)
        else : 
            best_params = self.select_best_fit(params)
        return best_params, params 
    """
    ^^^^^ Selection Mechanism ^^^^^
    
    >>>>> 2. Breeding Mechanism >>>>>
    """
    def random_parent_number(self, min_element=0, max_element=25):

        def is_passed_rule(r1, r2, r3, r4):
            if r1 == r2 or r1 == r3 or r1 == r4: 
                return False 
            if r2 == r3 or r2 == r4 :
                return False 
            if r3 == r4 : 
                return False 
            return True 

        while True : 
            r1 = randint(min_element, max_element)
            r2 = randint(min_element, max_element)
            r3 = randint(min_element, max_element)
            r4 = randint(min_element, max_element)
            
            if is_passed_rule(r1,r2,r3,r4):
                return r1,r2,r3,r4

    def generate_offspring(self, parent_params):
        n = len(parent_params) -1 
        r1, r2, r3, r4 = self.random_parent_number(max_element=n)
        p3 = parent_params.iloc[r3,:]
        p4 = parent_params.iloc[r4,:] 
        p2 = parent_params.iloc[r2,:]
        p1 = parent_params.iloc[r1,:]
        
        c1 = {} 
        c2 = {}
        for col in self.df_columns : 
            rand_val = uniform(0,1) 
            # select p1, p3 
            if rand_val >= 0.5 : 
               c1[col] = p1[col]
               c2[col] = p3[col]
            else :
                c1[col] = p2[col]
                c2[col] = p4[col]
        # take average 
        o = (c1['o'] + c2['o'])/2
        m = (c1['m'] + c2['m'])/2
        A = (c1['A'] + c2['A'])/2
        B = (c1['B'] + c2['B'])/2
        C = (c1['C'] + c2['C'])/2
        tc = (c1['tc'] + c2['tc'])/2
        return  o, m, A, B, C, tc   


    def breeding_mechanism(self, data, best_selection_params, n_select=25, iter=25):
        params = pd.DataFrame(columns=self.df_columns)
        n_xd = len(data)
        xd = np.linspace(0.1, n_xd, n_xd)
        # o_list = best_selection_params['o']
        # m_list = best_selection_params['m']
        # A_list = best_selection_params['A']
        # B_list = best_selection_params['B']
        # C_list = best_selection_params['C']
        # tc_list = best_selection_params['tc']
        for i in range(iter):
            o, m, A, B, C, tc = self.generate_offspring(best_selection_params) 
            y_fit = y(xd, o, m, A, B, C, tc)
            mse = sum_squared_error(data, y_fit)
            params = params.append({'o' : o, 
                                    'm' : m, 
                                    'A' : A, 
                                    'B' : B, 
                                    'C' : C, 
                                    'tc': tc, 
                                    'mse': mse}, ignore_index=True)
        return params

    """
    ^^^^^ Breeding Mechanism ^^^^^
    
    >>>>> 3. Mutation Mechanism >>>>>
    """
    def generate_mutation_params(self, all_selection_params, row, m_range, o_range, A_range, B_range, C_range, tc_range):
        o_eps = uniform(-o_range, o_range)
        m_eps =uniform(-m_range, m_range)
        A_eps =uniform(-A_range, A_range)
        B_eps =uniform(-B_range, B_range)
        C_eps =uniform(-C_range, C_range)
        tc_eps =uniform(0, tc_range)
        m = all_selection_params['m'][row] + m_eps
        A = all_selection_params['A'][row] + A_eps
        o = all_selection_params['o'][row] + o_eps
        B = all_selection_params['B'][row] + B_eps
        C = all_selection_params['C'][row] + C_eps
        tc = all_selection_params['tc'][row] + tc_eps
        return  o, m, A, B, C, tc

    def mutaion_mechanism(self, best_selection_params, all_selection_params, data, n_select=25, n_random=50, k=0.8):
        params = all_selection_params    
        o_range = best_selection_params['o'].max() -  best_selection_params['o'].min()
        m_range = best_selection_params['m'].max() -  best_selection_params['m'].min()
        A_range = best_selection_params['A'].max() -  best_selection_params['A'].min()
        B_range = best_selection_params['B'].max() -  best_selection_params['B'].min()
        C_range = best_selection_params['C'].max() -  best_selection_params['C'].min()
        tc_range = best_selection_params['tc'].max() - best_selection_params['tc'].min()

        m_range *= k
        o_range *= k
        A_range *= k
        B_range *= k
        C_range *= k
        tc_range *= k

        n_xd = len(data)
        xd = np.linspace(0.1, n_xd, n_xd)

        for i in range(len(all_selection_params)):
            o, m, A, B, C, tc = self.generate_mutation_params(all_selection_params, i, m_range, o_range, A_range, B_range, C_range, tc_range)
            y_fit = y(xd, o, m, A, B, C, tc)
            mse = sum_squared_error(data, y_fit)
            params = params.append({'o' : o, 
                                    'm' : m, 
                                    'A' : A, 
                                    'B' : B, 
                                    'C' : C, 
                                    'tc': tc, 
                                    'mse': mse}, ignore_index=True)
        best_params = self.select_best_fit(params)
        return best_params

    """
    ^^^^^ Mutation Mechanism ^^^^^
    
    >>>>> 4. Culling Mechanism >>>>>
    """
    def culling_mechanism(self, selection_params, breeding_params, mutation_params):
        params = self.next_generation.append(selection_params, ignore_index=True).append(breeding_params, ignore_index=True).append(mutation_params, ignore_index=True)
        params = params[(params.o >= self.o_min) & (params.o <= self.o_max)]
        # params = params[params.o >= 1]
        params = params[(params.m >= self.m_min) & (params.m <= self.m_max)]
        # params = params[(params.A >= self.A_min) & (params.A <= self.A_max)]
        # params = params[(params.B >= self.B_min) & (params.B <= self.B_max)]
        # params = params[(params.C >= self.C_min) & (params.C <= self.C_max)]
        params = params[(params.tc >= self.tc_min) & (params.tc <= self.tc_max)]
        self.next_generation = self.select_best_fit(params)
            








    






        
        
        

 

        
        

        