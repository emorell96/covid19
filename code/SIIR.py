import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit
from scipy.integrate import odeint


class SIIR:
    #all simulation will be done with hours so the rates are in 1/hour, the delay, risetime, iso days are in days. The c is a percentage.
    def __init__(self, tfinal, tinitial = 0, beta = 0.01, c = 0.01, total_pop = 40e6, gamma = 7.58e-4, lamda = 0.014, alpha0 = 0.04, isolation_days = 30, isolation_delay = 10, risefalltime = 2, isolation=True):
        """
        tinitial and tfinal in days!
        """
        self.beta = beta
        self.alpha0 = alpha0
        self.c = c
        self.total_pop = total_pop
        self.gamma = gamma
        self.lamda = lamda
        self.isolation_time = isolation_days*24
        self.isolation_delay = isolation_delay*24
        self.risetime = risefalltime*24
        self.t0 = tinitial*24
        self.t1 = tfinal*24
        self.isolation = isolation
    def set_initial_conditions(self, S0, I0, Qs0 = 0, Qh0 = 0, R0 =0):
        self.initial = np.array([S0, I0, Qs0, Qh0, R0])
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
    def solve(self, points):
        T = np.linspace(self.t0, self.t1, points)
        Y = odeint(self.SIIR_Model, self.initial, T, full_output = 0)
        return (T, Y)
    
    

        


beta = 1/(24*4.375)
gamma = 1/(24*55.5)
N = 39.56e6
tinitial = 0
tfinal = 750 #days
points = 20000
days_wo_isolation = 0 #days
isolation_days = 100 #days
risefalltime = 3 #days

model = SIIR(360, total_pop=39.56e6,beta=1/(24*4.375), c=0.5, lamda=1/(24*5), alpha0=1/(24*5), isolation=True ,isolation_delay=20, isolation_days=30, risefalltime=15)
model.set_initial_conditions(model.total_pop, 1190, Qh0=10, R0=20)
T, Y = model.solve(points)
#T = np.linspace(0, 60*24, points)
Y1  = np.vectorize(model.alphaprime, otypes=[np.float64])(T)
Y2 = np.vectorize(model.alpha, otypes=[np.float64])(T)
plt.plot(T/24, Y1)
plt.plot(T/24, Y2)

index = np.argmax(Y[:,1]+Y[:,2])
print(f"Maximum number of infected (quarantined and not quarantined) is {Y[index,1]+Y[index,2]:.2e} and happens on day {T[index]/24:.2f}.")

plt.figure()
plt.plot(T/24, Y[:,0], label="Susceptible")
plt.plot(T/24, Y[:,1]  + Y[:,2], label="Infected + Quarantine Sick")
plt.plot(T/24, Y[:,2], label="Quarantined Sick")
plt.plot(T/24, Y[:,1], label="Infected not quarantined")
plt.plot(T/24, Y[:,3], label = "Healthy in Quarantine")
plt.plot(T/24, Y[:,4], label="Recovered")
plt.legend()
plt.show()
#initial conditions
