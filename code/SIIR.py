import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from countryinfo import CountryInfo
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import datetime as date
from lmfit import Model, Parameters
from scipy.stats import nbinom
import warnings
import pystan

class SIIR:
    #all simulation will be done with hours so the rates are in 1/hour, the delay, risetime, iso days are in days. The c is a percentage.
    def __init__(self, tfinal, tinitial = 0, reproduction_number = 2, beta=None ,c = 0.01, total_pop = 40e6, gamma = 7.58e-4, lamda = 0.014, alpha0 = 0.04, isolation_days = 30, isolation_delay = 10, 
                I0 = 10, care_probability = 0.1, delta = 1, risefalltime = 2, isolation=False, death_rate = 0.025, health_care_threshold = 1e-3, model="SIIRD", country = ""):
        """
        tinitial and tfinal in days!
        """
        ##main parameters
        self._reproduction_number = reproduction_number
        
        self._beta = beta
        self.gamma = gamma
        self.death_prob = death_rate
        ##consutation parameters:
        self.delta = delta
        self.care_probability = care_probability
        ##isolation parameters
        #compulsory isolation
        self.alpha0 = alpha0
        #compulsory and self isolation
        self.c = c
        self.total_pop = total_pop
        self.gamma = gamma
        self.lamda = lamda
        self.isolation_time = isolation_days
        self.isolation_delay = isolation_delay

        self.risetime = risefalltime
        self.t0 = tinitial
        self.t1 = tfinal

        self.isolation = isolation
        

        

        

        self.model = model
        self.finished = False
        self.country = country
        #S0, I0, Qs0 = 0, Qh0 = 0, R0 = 0, D0 = 0
        
        self._initial = np.array([self.total_pop-I0,I0, 0, 0, 0, 0]) if self.model == "SIIRD" else np.array([self.total_pop-I0,I0, 0, 0, 0])
        #not implemented.
        self.health_care_threshold = health_care_threshold
    def _stan_parameters(self):
        #this function will generate the parameters used in the model for the code:
        self.params = """
        parameters {
            #This are the parameters of the model: R0, gamma, I0, c, delta, lambda, care_probability (detection prob)
            real<lower=0, upper=10> reproduction_number;
            real<lower=0, upper=1> gamma;
            real<lower=1> I0;
            real<lower=0, upper=1> c;
            real<lower=0, upper=1> delta;
            real<lower=0, upper=10> lambda;
            real<lower=0, upper=1> care_probability;
        }
        """
    @property
    def beta(self):
        if self._beta is None:
            return self._reproduction_number*self.gamma
        return self._beta
    @beta.setter
    def beta(self, value):
        if self.beta is None:
            warnings.warn("The variable beta was not initialized and you are trying to set a value for it. Are you sure?\
                        Wouldn't you want to set R0, and the infectious rate instead?")
        self._beta = value
    @property
    def reproduction_number(self):
        return self._reproduction_number
    @reproduction_number.setter
    def reproduction_number(self, value):
        #check that value is > 0
        if value <= 0:
            warnings.warn("Value for R0 is equal or less than 0. It should be stricly greater than 0. Value will not change.")
        else:
            self._reproduction_number = value
    # @property
    # def infectious_rate(self):
    #     return self._infectious_rate
    # @infectious_rate.setter
    # def infectious_rate(self, value):
    #     if value <= 0:
    #         warnings.warn("Value for infectious rate is equal or less than 0. It should be stricly greater than 0. Value will not change.")
    #     else:
    #         self._infectious_rate = value
    @property
    def I0(self):
        return self.initial[1]
    @property
    def R0(self):
        return self.initial[4]
    @property
    def S0(self):
        return self.total_pop - self.I0 - self.R0 - self.D0
    @property
    def D0(self):
        if self.model == "SIIRD":
            return self.initial[5]
        return None
    @I0.setter
    def I0(self, value):
        if value <= 0:
            warnings.warn("Value for initial infected is equal or less than 0. It should be stricly greater than 0. Value will not change.")
        else:
            self.initial[1] = value
    @R0.setter
    def R0(self, value):
        if value <= 0:
            warnings.warn("Value for initial recovered is equal or less than 0. It should be stricly greater than 0. Value will not change.")
        else:
            self.initial[4] = value
    @S0.setter
    def S0(self, value):
        if value <= 0:
            warnings.warn("Value for initial susceptible cannot be fixed. It depends on other values. Value will not change.")
    
    
    @property
    def initial(self):
        return self._initial
    @initial.setter
    def initial(self, _initial : np.array):
        if self.model == "SIIRD" and len(_initial) == 6:
            self._initial = _initial
        elif self.model == "SIIR" and len(_initial) == 5:
            self._initial = _initial
        else:
            warnings.warn("Warning, wrong size for initial vector.")
    def alpha(self, t):
        if not self.isolation:
            return 0
        if (t - self.t0) < self.isolation_delay:
            return 0
        if (t - self.t0 - self.isolation_delay) <= self.risetime:
            v = self.alpha0/self.risetime*(t - self.t0-self.isolation_delay)
            return v
        if (t - self.t0 - self.isolation_delay - self.risetime) < self.isolation_time:
            return self.alpha0
        if (t - self.t0 - self.isolation_delay - self.risetime - self.isolation_time) < self.risetime:
            return self.alpha0-self.alpha0/self.risetime*(t-self.t0 - self.isolation_delay - self.risetime - self.isolation_time)
        else:
            return 0
    def alphaprime(self, t):
        if not self.isolation:
            return 0
        if (t - self.t0 - self.isolation_delay - 2*self.risetime - self.isolation_time) < 0:
            return 0
        if (t - self.t0 - self.isolation_delay - 2*self.risetime - self.isolation_time) <= self.risetime:
            v = self.alpha0/self.risetime*(t - self.t0 - self.isolation_delay - 2*self.risetime - self.isolation_time)
            return v
        # if (t - self.t0 - self.isolation_delay - 3*self.risetime - 2*self.isolation_time)  < 0:
        #     return self.alpha0
        # if (t - self.t0 - self.isolation_delay - 4*self.risetime - 2*self.isolation_time) < 0:
        #     return self.alpha0-(self.alpha0*0.1)/self.risetime*(t-self.t0 - self.isolation_delay - 3*self.risetime - 2*self.isolation_time)
        else:
            return self.alpha0
    def SIIR_Model(self, X, t):
        S = X[0]
        I = X[1]
        Qs = X[2]
        Qh = X[3]
        R = X[4] #I/self.total_pop in Qs dot # (S/self.total_pop)
        Sdot = -self.beta*S/self.total_pop*I-self.alpha(t)*S+self.alphaprime(t)*S*Qh/self.total_pop-self.beta*self.c*Qs*S/self.total_pop #*I/self.total_pop in Idot
        Idot = self.beta*S/self.total_pop*I+self.beta*self.c*I*Qh/self.total_pop+self.beta*self.c*S*Qs/self.total_pop-self.lamda*I-self.gamma*I-self.alpha(t)*I*S/self.total_pop#+self.alphaprime(t)*Qs#*(S+R)/self.total_pop
        Qsdot = self.lamda*I-self.gamma*Qs+self.alpha(t)*I*S/self.total_pop#-self.alphaprime(t)*Qs#*(S+R)/self.total_pop
        Qhdot = self.alpha(t)*S-self.alphaprime(t)*S*Qh/self.total_pop-self.beta*self.c*I/self.total_pop*Qh #*(S/self.total_pop)
        Rdot = self.gamma*(I+Qs)
        out = np.array([Sdot, Idot, Qsdot, Qhdot, Rdot])
        #print(np.sum(out))
        return out
    def SIIRD_Model(self, X, t):
        S = X[0]
        I = X[1]
        Qs = X[2]
        Qh = X[3]
        R = X[4]
        D = X[5]
        gamma = self.gamma
        #introduces the notion of a sharp increase in gamma when a threshold is reached.
        #after the thershold is reached, the gamma decreases drastically
        #we see how many people are sick and compare to alive people times the threshold
        # if (I+Qs) > (self.total_pop-D)*self.health_care_threshold:
        #     #recovery rate is halved:
        #     gamma /= 4
         
        #I/self.total_pop in Qs dot # (S/self.total_pop)
        Sdot = -self.beta*S/self.total_pop*I-self.alpha(t)*S+self.alphaprime(t)*S*Qh/self.total_pop-self.beta*self.c*Qs*S/self.total_pop #*I/self.total_pop in Idot
        Idot = self.beta*S/self.total_pop*I+self.beta*self.c*I*Qh/self.total_pop+self.beta*self.c*S*Qs/self.total_pop-self.lamda*I-gamma*I-self.alpha(t)*I*S/self.total_pop - self.death_prob*I #*0 #+self.alphaprime(t)*Qs#*(S+R)/self.total_pop #/(1-self.death_prob) *gamma
        Qsdot = self.lamda*I-gamma*Qs+self.alpha(t)*I*S/self.total_pop - self.death_prob*Qs  #*0   #-self.alphaprime(t)*Qs#*(S+R)/self.total_pop # /(1-self.death_prob)*gamma
        Qhdot = self.alpha(t)*S-self.alphaprime(t)*S*Qh/self.total_pop-self.beta*self.c*I/self.total_pop*Qh #*(S/self.total_pop)
        Rdot = gamma*(I+Qs)
        Ddot = self.death_prob*(Qs+I)#/(1-self.death_prob)#*0 #*gamma*#(1-self.death_prob)*gamma
        #if I+Qs < 1 then no host remain and the disease has been killed and numbers should no longer change.
        
        if I+Qs < 1:
            Sdot, Idot, Qsdot, Qhdot, Rdot, Ddot = (0, 0, 0, 0, 0, 0)
            #Idot = 0
            #Qsdot = 0
            #Rdot = 0
            #Ddot = 0
            if not self.finished:
                print("Disease has been killed. All infected and quarantined infected have either recovered or died.")
                print(f"It happened on day {t:.2f} in {self.country}. ")
                print(f"With 1/beta = {1/self.beta:.1f}, 1/gamma = {1/self.gamma:.1f}, c ={model.c:.2e}, 1/lambda={1/self.lamda}, death_rate = {self.death_prob:.2e}.")
                self.finished = True

            
        out = np.array([Sdot, Idot, Qsdot, Qhdot, Rdot, Ddot])
        #print(np.sum(out))
        return out
    
    def _solve(self, points):
        self.finished = False
        T = np.linspace(self.t0, self.t1, points)
        if self.model == "SIIR":
            Y = odeint(self.SIIR_Model, self.initial, T, full_output = 0)
        if self.model == "SIIRD":
            Y = odeint(self.SIIRD_Model, self.initial, T, full_output = 0)
        self.T = T
        self.Y = Y
        return (T, Y)
    def _consultations(self):
        """
        Calculates the expected number of consultations each day
        """
        self.E = np.zeros(self.Y.size[0])
        self.Cdist = np.empty(self.Y.size[0], dtype=nbinom)
        for i in range(1, self.Y.size[0]):
            #incident cases Z_i = S(i-1)-S(i)
            #expected consultations for Covid 19: E_i = prob * Z_i
            self.E[i] = (self.Y[i-1, 0]-self.Y[i,0])*self.care_probability
            r = pow(self.E[i], self.delta)
            self.Cdist[i] = nbinom(n=r, p=self.E[i]/(r+self.E[i]))
        
        
    def solve(self, points):
        return self._solve(points)