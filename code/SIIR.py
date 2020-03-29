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


mypath = r"C:\Users\emore\Documents\covid19 analysis\data\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

data = {}
for file in onlyfiles:
    path = join(mypath, file)
    try:
        data[file.replace(".txt", "")] = (file, np.loadtxt(path, skiprows=1, usecols=(0,1, 2, 3), converters = {1: lambda s: float(s.decode("UTF-8").replace(",", "")), 2: lambda s: float(s.decode("UTF-8").replace(",", "")), 3: lambda s: float(s.decode("UTF-8").replace(",", ""))}))
    except:
        try:
            data[file.replace(".txt", "")] = (file, np.loadtxt(path, skiprows=1, usecols=(0,1, 2, 3), converters = {1: lambda s: float(s.decode("UTF-8").replace(",", "")), 2: lambda s: float(s.decode("UTF-8").replace(",", ""))}))    
        except:
            pass
        data[file.replace(".txt", "")] = (file, np.loadtxt(path, skiprows=1, usecols=(0,1), converters = {1: lambda s: float(s.decode("UTF-8").replace(",", ""))}))

for country, table in data.values():
    print(country)
    if country == "Russia.txt":
        russia = table
    table[:,0] -= 1 #set t=1 as t=0 as it should be.
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
                print(f"With 1/beta = {1/model.beta:.1f}, 1/gamma = {1/model.gamma:.1f}, c ={model.c:.2e}, 1/lambda={1/model.lamda}, death_rate = {model.death_prob:.2e}.")
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

beta = 1/(2)
gamma = 1/(100)
#death_prob = 0.015

lamda = 1/2
N = 39.56e6
tinitial = 0
tfinal = 750 #days
points = 2500
days_wo_isolation = 0 #days
isolation_days = 100 #days
risefalltime = 3 #days
c_value = 0.5
death_rate = 0.003

def fitter(model, time, cases, recovered = None, deaths = None, points = 2000, first_days = -1):
    #cases are the confirmed cases, and it should be a np array with actual dat
    #recovered data for recovered
    #time is the first parameter for curve_fit
    global_gamma = model.gamma
    global_beta = model.beta
    global_c = model.c
    global_lambda = model.lamda
    global_death = model.death_prob
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
    
    def _fit_multiple_all(comboData, _beta, _gamma, _c = global_c, _lambda = global_lambda, _death_rate = global_death):
        #arrays = len(comboData)/len(cases)
        extract1 = comboData[:len(cases)] # first data
        extract2 = comboData[len(cases):2*len(cases)]
        extract3 = comboData[2*len(cases):]
        model.death_prob = _death_rate
        model.gamma = _gamma
        model.beta = _beta
        model.c = _c
        model.lamda = _lambda
        T, Y = model.solve(points)
        result1 = interp1d(T, Y[:, 1]+ Y[:,2]+Y[:,4]+Y[:,5], kind='linear')(extract1)
        result2 = interp1d(T, Y[:, 4], kind='linear')(extract2)
        result3 = interp1d(T, Y[:,5])(extract3)
        return np.append(np.append(result1, result2), result3)
    def _fit_multiple_cr(comboData, _beta, _gamma, _c = global_c, _lambda = global_lambda, _death_rate = death_rate):
        arrays = len(comboData)/len(cases)
        extract1 = comboData[:len(cases)] # first data
        extract2 = comboData[len(cases):]
        model.death_prob = death_rate
        model.gamma = _gamma
        model.beta = _beta
        model.c = _c
        model.lamda = _lambda
        T, Y = model.solve(points)
        result1 = interp1d(T, Y[:, 1]+ Y[:,2]+Y[:,4]+Y[:,5], kind='linear')(extract1)
        result2 = interp1d(T, Y[:, 4], kind='linear')(extract2)
        #result3 = interp1d(T, Y[:,5])(extract3)
        return np.append(result1, result2)
    #print(1/global_lambda/24)
    # model.gamma, model.beta = (global_gamma, global_beta)
    # model.c = global_c
    if recovered is not None and deaths is not None:
        comboDataY = np.append(np.append(cases, recovered), deaths)
        comboDataX = np.append(np.append(time, time), time)
        if model.country == "Italy":
            global_beta = 1/80
            global_gamma = 1/10
            global_c = 1
            global_lambda = 1/4
            popt, pcov = curve_fit(_fit_multiple_all, comboDataX, comboDataY, p0=[global_beta, global_gamma, global_c, global_lambda, death_rate], bounds=([1/100, 1/200, 1e-19, 0, 0], [200, 100, 1, 500, 1]))
        else:
            popt, pcov = curve_fit(_fit_multiple_all, comboDataX, comboDataY, p0=[global_beta, global_gamma, global_c, global_lambda, death_rate], bounds=([1/100, 1/200, 1e-19, 0, 0], [200, 100, 1, 500, 1]))
        model.beta, model.gamma, model.c, model.lamda, model.death_prob = popt
    elif recovered is not None and deaths is None:
        comboDataY = np.append(cases, recovered)
        comboDataX = np.append(time, time)
        popt, pcov = curve_fit(_fit_multiple_cr, comboDataX, comboDataY, p0=[global_beta, global_gamma, global_c, global_lambda], bounds=([0, 0.0061, 1e-3, 0], [10, 10, 5, 5]))
        model.beta, model.gamma, model.c, model.lamda = popt

    else: #we  fit to beta
        popt, pcov = curve_fit(_fit_beta_2d, time[:first_days], cases[:first_days], p0=[global_beta, global_c], bounds=([0, 1e-3], [5, 1]))
        #We now write the beta into the global variable and proceed with the recovered cases if different than none
        model.beta = popt[0]#, popt[1] #, global_c, global_lambda
        model.c = popt[1]
    print(f"The time constants (in days) for {model.country} are beta = {1/model.beta:2f} and gamma = {1/model.gamma:.2f}. c = {model.c}, lambda = {model.lamda:.1e}, death rate = {model.death_prob}")
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
    d0 = 0
    r0 = 0
    try:
        d0 = table[0,3]
        try:
            r0 = table[0,2]
        except:
            pass
    except:
        pass
    model.set_initial_conditions(model.total_pop-i0-d0-r0, I0=i0-r0, Qh0=0, R0=r0, D0=d0)
    try:
        # if model.country == "France":
        #     offset = 
        #     model = fitter(model, table[:offset,0], table[:offset,1], table[:offset,2], table[:offset,3])
        # else:
        model = fitter(model, table[:,0], table[:,1], table[:,2], table[:,3])
    except:
        try:
            model = fitter(model, table[:,0], table[:,1], table[:,2])
        except:
            model = fitter(model, table[:,0], table[:,1])

    
    #add model to main model list for future use
    Models[model.country] = model

#a = CountryInfo('Singapore'.encode("UTF-8"))
  

#beta, gamma, T, Y = fitter((russia[4:, 0]-5)*24, russia[4:,1], russia[4:, 2])


for model in Models.values():
    sim_days = 60
    model.t1 = model.t1 + sim_days

    #activate a 30 days quarantine:
    model.isolation = False
    model.isolation_time = 250
    model.risefalltime = 1
    #it happens today (so at the last data point)
    model.isolation_delay = len(data[model.country][1])
    #how long do they take to get to safety:
    model.alpha0 = 1/6 #1/(3 days)
    T, Y = model.solve(points)
    T -= len(data[model.country][1])-1
    index = np.argmax(Y[:,1]+Y[:,2])
    indexcum = np.argmax(Y[:,1]+Y[:,2]+Y[:,4]+Y[:,5])
    index2 = np.argmax(Y[:,5])

    #Default label in English
    title = f"Simulation using a deterministic model, based on current data, for the Covid 19 pandemic in {model.country}. t = 0 is 29th of March."
    labelcumulative = f"Total (Cumulative) Infected: {model.country}; Max = {(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum,4]+Y[indexcum,5])/1e6:.3f} million people ({(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum, 4])/model.total_pop*100:.3f} %)"
    actualcaseslabel = f"Actual Cases: {model.country}"
    labeldeaths = f"Deaths: {model.country}, max = {Y[index2,5]/1e6:.3f} million people ({Y[index2,5]/model.total_pop*100:.3f} %)"
    xlabel = "Time (days)"
    actualdeathslabel = f"Actual Deaths: {model.country}"
    actualrecoveredlabel = f"Actual Recovered: {model.country}"
    labelrecovered = f"Recovered: {model.country}"
    infectedlabel = f"(NOT Cumulative) Infected on a given day: {model.country}; Max = {(Y[index,1]+Y[index,2])/1e6:.1f} million people ({(Y[index,1]+Y[index,2])/model.total_pop*100:.2f} %)"
    #Translation for different countries:
    if model.country == "Russia":
        title = f"Моделирование перехода Covid 19 в России с использованием модели SIIR (восприимчивый, зараженный, изолированный, восстановленный).\nСледующие {sim_days} дней\nt = 0 - 29 марта 2020 г."
        labelcumulative = f"Общее количество случаев (накопительное) в России; Максимальное количество дел =  {(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum,4]+Y[indexcum,5])/1e6:.3f} ({(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum,4]+Y[indexcum,5])/model.total_pop*100:.3f} %) "
        actualcaseslabel = f"Общее количество случаев в России (накопительное)"
        labeldeaths = f"Смертность в России, макс. = {Y[index2,5]/1e6:.3f} миллионов человек ({Y[index2,5] /model.total_pop*100:.3f}%) \n (# число смертей гораздо менее точное, чем другие, поскольку в настоящее время \n недостаточно данных для того, чтобы сделать прогноз с достаточной уверенностью, \n, хотя это не влияет на другие показатели)"
        xlabel = u"Время (дни)\n Примечание. Это прогноз на основе модели, который означает, что по своей природе будут сделаны упрощения,\n и предполагаются предположения о передаче вируса, что означает, что цифры не являются надежными на 100%. \n Существует определенный запас ошибка, которая может быть довольно большой, так как ситуация меняется каждый день. \n Также меры, принятые правительством, могут занять несколько дней, чтобы быть отраженными в данных. \n Сказав это, номера моделей можно рассматривать как наихудший сценарий для России сегодня, если ничего кардинально не изменится \n И это показывает, что для сокращения числа случаев в долгосрочной перспективе необходима продолжительная очень сильная изоляция,\nи что Вы должны относиться к этому вирусу серьезно, так как он явно не исчезнет в течение ДЛИТЕЛЬНОГО времени. \n Текущий источник данных (используется для подгонки модели к текущей ситуации): Минздрав России \n Автор: (Twitter) @emorell96 (Студент ищет докторскую степень по физике)"
        actualdeathslabel = f"Смерти от Covid 19 в России"
        labelrecovered = f"Выздоровели люди в России"
        infectedlabel = f"Зараженные люди в России; максимальная = {(Y[index,1]+Y[index,2])/1e6:.1f} миллионов человек ({(Y[index,1]+Y[index,2])/model.total_pop*100:.2f} %)"

    elif model.country == "Colombia":
        title = f"Simulación de la transimición del Covid 19 en Colombia usando un modelo SIIR (Susceptible, Infectado, Aislado, Recuperado).\nPróximos {sim_days} días.\nt=0 es el 28 de Marzo del 2020."
        labelcumulative = f"Total de casos (cumulativo) en Colombia; Maximo # de Casos = {(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum,4]+Y[indexcum,5])/1e3:.2f} mil personas ({(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum, 4])/model.total_pop*100:.3f} % de la población)"
        actualcaseslabel = f"Numero de casos totales en Colombia (cumulativo)(datos del Instituto Nacional de la Salud)"
        labeldeaths = f"Muertes en {model.country}, max = {Y[index2,5]/1e3:.2f} miles de personas ({Y[index2,5]/model.total_pop*100:.3f} %)\n(# de muertes es un numero que es mucho menos preciso que los demas ya que no hay\nsuficientes datos actualmente para hacer una predicción con buena certitud,\n aunque esto no afecto los demás indicadores)"
        xlabel = u"Tiempo (dias) \nNota: Esto es una prediccion basado en un modelo, lo que significa que por naturaleza se harán simplificaciones, y se asumen hipotesis sobre la transmisión del virus, lo que significa que los numeros no son 100% fiables.\n Hay una cierta margen de error que puede ser bastante grande ya que la situación cambia todos los días. \n Tambien las medidas tomadas por el gobierno puede tomar algo de tiempo en verse reflejado por los datos. \n Diciendo eso, los numeros del model si pueden ser vistos como el peor escenario posible para Colombia actualmente. \n Y demuestra que se necesita continuar con un aislamiento muy fuerte para disminuir el numero de casos a largo plazo, y que se tiene que tomar este virus con seriedad ya que claramente esto no va desaparecer en MUCHO MUCHO tiempo.\nFuente: INS\nAutor:(Twitter) @emorell96 (Estudiante de Doctorado en Física en Estados Unidos)"
        actualdeathslabel = f"Muertes del Covid 19 en {model.country}"
        actualrecoveredlabel = f"Personas recuperadas en {model.country}"
        labelrecovered = f"Predicción de personas recuperadas en {model.country} \n*Este indicador podría mejor en los proximos días ya que aún no hay suficientes datos"
        infectedlabel = f"Numero de personas infectadas en cada día en {model.country}; Max = {(Y[index,1]+Y[index,2])/1e3:.2f} mil personas ({(Y[index,1]+Y[index,2])/model.total_pop*100:.3f} %)"





    print(f"Maximum number of infected (quarantined and not quarantined) for {model.country} is {Y[index,1]+Y[index,2]:.2e} ({(Y[index,1]+Y[index,2])/model.total_pop*100:.2f} %) and happens on day {T[index]:.2f}.")
    
    plt.figure()
    plt.title(title)
    #plt.plot(T, Y[:,0], label="Susceptible")
    plt.scatter(1+data[model.country][1][:, 0]-len(data[model.country][1]), data[model.country][1][:,1], label = actualcaseslabel)
    plt.plot(T, Y[:,1]  + Y[:,2]+Y[:,4]+Y[:,5], label=labelcumulative)
    plt.axes().yaxis.set_minor_locator(MultipleLocator(5))
    plt.plot(T, Y[:,5], label=labeldeaths)
    plt.xlabel(xlabel)
    try:
        try:
            
            plt.scatter(1+data[model.country][1][:,0]-len(data[model.country][1]), data[model.country][1][:,3], label=actualdeathslabel)
        except:
            pass
        plt.scatter(1+data[model.country][1][:,0]-len(data[model.country][1]), data[model.country][1][:,2], label=actualrecoveredlabel)
    except:
        pass
    plt.plot(T, Y[:,4], label=labelrecovered)
    plt.plot(T, Y[:,1]  + Y[:,2], label= infectedlabel)
    plt.legend()
    # plt.figure()
    # Y1  = np.vectorize(model.alphaprime, otypes=[np.float64])(T)
    # Y2 = np.vectorize(model.alpha, otypes=[np.float64])(T)
    # plt.plot(T, Y1)
    # plt.plot(T, Y2)

    
    # plt.plot(T/24, Y[:,2], label="Quarantined Sick")
    # plt.plot(T/24, Y[:,1], label="Infected not quarantined")
    # plt.plot(T/24, Y[:,3], label = "Healthy in Quarantine")
    
    
    
plt.show()


        

# T, Y = model.solve(points)
#T = np.linspace(0, 60*24, points)


# #initial conditions
