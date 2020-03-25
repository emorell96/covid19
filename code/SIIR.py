import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from countryinfo import CountryInfo


mypath = r"C:\Users\emore\Documents\covid19 analysis\data\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

data = {}
for file in onlyfiles:
    path = join(mypath, file)
    try:
        data[file.replace(".txt", "")] = (file, np.loadtxt(path, skiprows=1, usecols=(0,1, 2), converters = {1: lambda s: float(s.decode("UTF-8").replace(",", "")), 2: lambda s: float(s.decode("UTF-8").replace(",", ""))}))
    except:
        data[file.replace(".txt", "")] = (file, np.loadtxt(path, skiprows=1, usecols=(0,1), converters = {1: lambda s: float(s.decode("UTF-8").replace(",", ""))}))

for country, table in data.values():
    print(country)
    if country == "Russia.txt":
        russia = table
    # try:
    #     table[:,1] = table[:,1]-table[:,2] #the second column now properly corresponds only to active cases (only if recovered data is available)
    # except:
    #     pass

class SIIR:
    #all simulation will be done with hours so the rates are in 1/hour, the delay, risetime, iso days are in days. The c is a percentage.
    def __init__(self, tfinal, tinitial = 0, beta = 0.01, c = 0.01, total_pop = 40e6, gamma = 7.58e-4, lamda = 0.014, alpha0 = 0.04, isolation_days = 30, isolation_delay = 10, risefalltime = 2, isolation=True, death_rate = 0.025, health_care_threshold = 1e-3, model="SIIRD", country = ""):
        """
        tinitial and tfinal in days!
        """
        self.beta = beta
        self.alpha0 = alpha0
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
        self.death_prob = death_rate
        self.health_care_threshold = health_care_threshold
        self.model = model
        self.finished = False
        self.country = country
    def set_initial_conditions(self, S0, I0, Qs0 = 0, Qh0 = 0, R0 =0, D0=0):
        if self.model == "SIIRD":
            self.initial = np.array([S0, I0, Qs0, Qh0, R0, D0])
        if self.model == "SIIR":
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
        Idot = self.beta*S/self.total_pop*I+self.beta*self.c*I*Qh/self.total_pop+self.beta*self.c*S*Qs/self.total_pop-self.lamda*I-gamma*I-self.alpha(t)*I*S/self.total_pop - self.death_prob/(1-self.death_prob)*gamma*I*0 #+self.alphaprime(t)*Qs#*(S+R)/self.total_pop
        Qsdot = self.lamda*I-gamma*Qs+self.alpha(t)*I*S/self.total_pop - self.death_prob/(1-self.death_prob)*gamma*Qs  #*0   #-self.alphaprime(t)*Qs#*(S+R)/self.total_pop
        Qhdot = self.alpha(t)*S-self.alphaprime(t)*S*Qh/self.total_pop-self.beta*self.c*I/self.total_pop*Qh #*(S/self.total_pop)
        Rdot = gamma*(I+Qs)
        Ddot = self.death_prob/(1-self.death_prob)*gamma*(Qs+I)#*0
        #if I+Qs < 1 then no host remain and the disease has been killed and numbers should no longer change.
        
        if I+Qs < 1:
            Sdot, Idot, Qsdot, Qhdot, Rdot, Ddot = (0, 0, 0, 0, 0, 0)
            #Idot = 0
            #Qsdot = 0
            #Rdot = 0
            #Ddot = 0
            if not self.finished:
                print("Disease has been killed. All infected and quarantined infected have either recovered or died.")
                print(f"It happened on day {t/24:.2f}.")
                self.finished = True

            
        out = np.array([Sdot, Idot, Qsdot, Qhdot, Rdot, Ddot])
        #print(np.sum(out))
        return out
    def solve(self, points):
        self.finished = False
        T = np.linspace(self.t0, self.t1, points)
        if self.model == "SIIR":
            Y = odeint(self.SIIR_Model, self.initial, T, full_output = 0)
            return (T, Y)
        if self.model == "SIIRD":
            Y = odeint(self.SIIRD_Model, self.initial, T, full_output = 0)
            return (T, Y)
    
    
def exponentialgrowth(x, A, tau):
    return A*np.exp(x/tau)

        

#Overall constants

beta = 1/(6)
gamma = 1/(30)


lamda = 1/5
N = 39.56e6
tinitial = 0
tfinal = 750 #days
points = 2500
days_wo_isolation = 0 #days
isolation_days = 100 #days
risefalltime = 3 #days
c_value = 1
death_rate = 0.03

def fitter(model, time, cases, recovered = None, points = 2000, first_days = -1):
    #cases are the confirmed cases, and it should be a np array with actual dat
    #recovered data for recovered
    #time is the first parameter for curve_fit
    global_gamma = model.gamma
    global_beta = model.beta
    global_c = model.c
    global_lambda = model.lamda
    #we first fit to the # of cases with beta and then will fit to the recovered once fixing beta.
    def _fit_beta_2d(t, _beta, _c = global_c, _gamma = global_gamma, _lambda = global_lambda): #, _lambda):
        model.gamma = _gamma
        model.beta = _beta
        model.c = _c
        model.lamda = _lambda
        T, Y = model.solve(points)
        predicted = Y[:,1]  + Y[:,2]#+Y[:,4]#+Y[:,5]
        return interp1d(T, predicted, kind='cubic')(t)
    def _fit_gamma(t, _gamma):
        model.gamma = _gamma
        model.beta = global_beta
        T, Y = model.solve(points)
        return interp1d(T, Y[:, 4], kind='cubic')(t)
    
    def _fit_multiple(comboData, _beta, _gamma, _c = global_c, _lambda = global_lambda):
        extract1 = comboData[:len(cases)] # first data
        extract2 = comboData[len(cases):]
        model.gamma = _gamma
        model.beta = _beta
        model.c = _c
        model.lamda = _lambda
        T, Y = model.solve(points)
        result1 = interp1d(T, Y[:, 1]+ Y[:,2]+Y[:,4], kind='linear')(extract1)
        result2 = interp1d(T, Y[:, 4], kind='linear')(extract2)
        return np.append(result1, result2)
    #print(1/global_lambda/24)
    # model.gamma, model.beta = (global_gamma, global_beta)
    # model.c = global_c
    if recovered is not None:
        comboDataY = np.append(cases, recovered)
        comboDataX = np.append(time, time)
        popt, pcov = curve_fit(_fit_multiple, comboDataX, comboDataY, p0=[global_beta, global_gamma, global_c, global_lambda], bounds=([0, 0.01, 0, 0], [10, 10, 5, 5]))
        model.beta, model.gamma, model.c, model.lamda = popt
    else: #we  fit to beta
        popt, pcov = curve_fit(_fit_beta_2d, time[:first_days], cases[:first_days], p0=[global_beta, global_c], bounds=([0, 0], [5, 1]))
        #We now write the beta into the global variable and proceed with the recovered cases if different than none
        model.beta = popt[0]#, popt[1] #, global_c, global_lambda
        model.c = popt[1]
    print(f"The time constants (in days) for {model.country} are beta = {1/model.beta:2f} and gamma = {1/model.gamma:.2f}. c = {model.c}, lambda = {model.lamda:.1e}")
    #T, Y = model.solve(points)
    #model parameters have been update, now user can so whatever he wants.
    return model
# def prediction(days, _gamma = 1/(24*143.5), _beta = 1/(24*4.5), _c = 0.1, _lamda =1/(24*5),  points = 2000):
#     model = SIIR(days, model="SIIRD",death_rate=0.015,total_pop=150e6,beta=_beta, gamma=_gamma,c=_c, lamda=_lamda, alpha0=1/(24*5), isolation=False ,isolation_delay=2, isolation_days=30, risefalltime=2)
#     model.set_initial_conditions(model.total_pop, 20, Qh0=0, R0=3, D0=0)
#     return model.solve(points)
Models = {}


for country, table in data.values():
    name = country.replace(".txt", "")
    x = table[:,0]
    y = table[:,1]
    popt, pcov = curve_fit(exponentialgrowth, x, y)
    #rowtime.append((country, float(popt[0]), float(popt[1])))
    model = SIIR(len(table[:,0]), country=name, model="SIIRD",death_rate=death_rate,total_pop=CountryInfo(name).population(),beta=1/(popt[1]), gamma=gamma,c=c_value, lamda=lamda, alpha0=1/(5), isolation=False ,isolation_delay=2, isolation_days=30, risefalltime=2)
    i0 = table[0,1]
    model.set_initial_conditions(model.total_pop, I0=i0, Qh0=0, R0=0, D0=0)
    try:
        model = fitter(model, table[:,0], table[:,1], table[:,2])
    except:
        model = fitter(model, table[:,0], table[:,1])

    
    #add model to main model list for future use
    Models[model.country] = model

#a = CountryInfo('Singapore'.encode("UTF-8"))
  

#beta, gamma, T, Y = fitter((russia[4:, 0]-5)*24, russia[4:,1], russia[4:, 2])


for model in Models.values():
    model.t1 = model.t1+120
    T, Y = model.solve(points)
    T -= len(data[model.country][1])
    index = np.argmax(Y[:,1]+Y[:,2])
    indexcum = np.argmax(Y[:,1]+Y[:,2]+Y[:,4])
    print(f"Maximum number of infected (quarantined and not quarantined) for {model.country} is {Y[index,1]+Y[index,2]:.2e} ({(Y[index,1]+Y[index,2])/model.total_pop*100:.2f} %) and happens on day {T[index]:.2f}.")
    index2 = np.argmax(Y[:,5])
    plt.figure()
    plt.title(f"Simulation using a deterministic model, based on current data, for the Covid 19 pandemic in {model.country}. t = 0 is 24th of March.")
    # plt.plot(T, Y[:,0], label="Susceptible")
    plt.scatter(data[model.country][1][:, 0]-len(data[model.country][1]), data[model.country][1][:,1], label = f"Actual Cases: {model.country}")
    plt.plot(T, Y[:,1]  + Y[:,2]+Y[:,4], label= f"Total (Cumulative) Infected: {model.country}; Max = {(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum,4])/1e6:.1f} million people ({(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum, 4])/model.total_pop*100:.2f} %)")
    plt.plot(T, Y[:,1]  + Y[:,2], label= f"Total (NOT Cumulative) Infected: {model.country}; Max = {(Y[index,1]+Y[index,2])/1e6:.1f} million people ({(Y[index,1]+Y[index,2])/model.total_pop*100:.2f} %)")
    try:
        plt.scatter(data[model.country][1][:,0]-len(data[model.country][1]), data[model.country][1][:,2], label=f"Actual Recovered: {model.country}")
    except:
        pass
    plt.plot(T, Y[:,4], label=f"Recovered: {model.country}")
    plt.plot(T, Y[:,5], label=f"Deaths: {model.country}, max = {Y[index2,5]/1e6:.1f} million people ({Y[index2,5]/model.total_pop*100:.2f} %)")
    plt.xlabel("Time (days)")
    plt.legend()
    # plt.figure()
    # Y1  = np.vectorize(model.alphaprime, otypes=[np.float64])(T)
    # Y2 = np.vectorize(model.alpha, otypes=[np.float64])(T)
    # plt.plot(T, Y1)
    # plt.plot(T, Y2)

    
    #plt.plot(T/24, Y[:,2], label="Quarantined Sick")
    #plt.plot(T/24, Y[:,1], label="Infected not quarantined")
    # plt.plot(T/24, Y[:,3], label = "Healthy in Quarantine")
    
    
    
plt.show()


        

# T, Y = model.solve(points)
#T = np.linspace(0, 60*24, points)


# #initial conditions
