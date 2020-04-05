from SIIR import SIIR
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


def exponentialgrowth(x, A, tau):
    return A*np.exp(x/tau)

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
filebase = f"C:\\Users\\emore\\Documents\\covid19 analysis\\data_out\\covid19SIIR_{date.datetime.today().strftime('%Y-%m-%d')}_"

def fitter(model, time, cases, recovered = None, deaths = None, points = 2000, first_days = -1, logarithmic = False):
    #cases are the confirmed cases, and it should be a np array with actual data
    #recovered data for recovered
    #time is the first parameter for curve_fit
    #This function has for aim to find the best parameters for the model in order to best fit to the available data.
    #It will try its best to fit to the total number of cases, the total recovered and total deaths trying to find the best parameters.
    #If some of this data is missing then it will fit to just the available data.
    #A word of caution the predicted deaths is pretty inaccurate since that's usually what we lack the most in data since there's a delay
    #between people falling ill and dying.
    #the logarithmic mode (to try to help the fit by fitting to more linear data is not working well yet DO NOT USE. (leave it as False))
    global_gamma = model.gamma
    global_beta = model.beta
    global_c = model.c
    global_lambda = model.lamda
    global_death = model.death_prob
    global_pop = 1
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
    #This function allows us to fit to the three available metrics at the SAME time. Meaning it will find the best set of parameters which will approximate the solution the best.
    def _fit_multiple_all(comboData, _beta, _gamma, _c = global_c, _lambda = global_lambda, _death_rate = global_death, pop=1):
        """
        #This function allows us to fit to the three available metrics at the SAME time. Meaning it will find the best set of parameters which will approximate the solution the best.
        comboData should be an appended numpy array with cases, recovered and deaths.
        """
        #arrays = len(comboData)/len(cases)
        extract1 = comboData[:len(cases)] # first data
        extract2 = comboData[len(cases):2*len(cases)]
        extract3 = comboData[2*len(cases):]
        model.death_prob = _death_rate
        model.gamma = _gamma
        model.beta = _beta
        model.c = _c
        model.lamda = _lambda
        model.total_pop *= pop
        T, Y = model.solve(points)
        result1 = interp1d(T, Y[:, 1]+Y[:,2]+Y[:,4]+Y[:,5], kind='linear')(extract1) #+ Y[:,2]+Y[:,4]+Y[:,5]
        result2 = interp1d(T, Y[:, 4], kind='linear')(extract2)
        result3 = interp1d(T, Y[:, 5])(extract3)
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
    def cleaner(data):
        if logarithmic:
            #we clean data (replace the zeros whith ones to avoid a log 0 to give infinity 1 is extremely close to one case in the sense of a population of millions)
            data[data < 1] = 1
            return np.log(data)
        else:
            #don't clean if not log
            return data
    #print(1/global_lambda/24)
    # model.gamma, model.beta = (global_gamma, global_beta)
    # model.c = global_c
    if recovered is not None and deaths is not None:
        comboDataY = np.append(np.append(cases, recovered), deaths)
        comboDataX = np.append(np.append(time, time), time)
        if model.country == "Italy":
            global_beta = 1/0.4
            global_gamma = 1/20
            global_c = 1e-10
            global_lambda = 1/1
            popt, pcov = curve_fit(_fit_multiple_all, comboDataX, comboDataY, p0=[global_beta, global_gamma, global_c, global_lambda, death_rate, global_pop], bounds=([1/100, 1/200, 1e-19, 0, 0, 0], [200, 100, 1, 500, 1, 1]))
        else:
            m = Model(_fit_multiple_all)
            parameters = Parameters()
            parameters.add_many(("_beta", global_beta, True, 1/200, 20, None, None),
                                ("_gamma", global_gamma, False, 1/200, 1/10, None, None),
                                ("_c", global_c, True, 1e-9, 1, None, None),
                                ("_lambda", global_lambda, False, 0, 500, None, None),
                                ("_death_rate", global_death, True, 1/200, 1/1, None, None),
                                ("pop", global_pop, False, 0.4, 1, None, None))
            result = m.fit(comboDataY, parameters, comboData = comboDataX)
            print(result.fit_report())
            popt = [result.best_values["_beta"], result.best_values["_gamma"], result.best_values["_c"], result.best_values["_lambda"], result.best_values["_death_rate"], result.best_values["pop"]]
            #popt, pcov = curve_fit(_fit_multiple_all, comboDataX, comboDataY, p0=[global_beta, global_gamma, global_c, global_lambda, death_rate, global_pop], bounds=([1/100, 1/200, 1e-19, 0, 0, 0], [200, 100, 1, 500, 1, 1]))
        model.beta, model.gamma, model.c, model.lamda, model.death_prob = popt
    elif recovered is not None and deaths is None:
        comboDataY = np.append(cases, recovered)
        comboDataX = np.append(time, time)
        popt, pcov = curve_fit(_fit_multiple_cr, comboDataX, comboDataY, p0=[global_beta, global_gamma, global_c, global_lambda], bounds=([0, 0.0061, 1e-3, 0], [10, 10, 5, 5]))
        model.beta, model.gamma, model.c, model.lamda = popt

    else: #we  fit to beta
        try:
            popt, pcov = curve_fit(_fit_beta_2d, time[:first_days], cases[:first_days], p0=[global_beta, global_c], bounds=([0, 1e-3], [5, 1]))
            #We now write the beta into the global variable and proceed with the recovered cases if different than none
            model.beta = popt[0]#, popt[1] #, global_c, global_lambda
            model.c = popt[1]
        except Exception as e:
            print(e)
    print(f"The time constants (in days) for {model.country} are beta = {1/model.beta:2f} and gamma = {1/model.gamma:.2f}. c = {model.c}, lambda = {model.lamda:.1e}, death rate = {model.death_prob}, total_pop = {model.total_pop:.2e}")
    #T, Y = model.solve(points)
    #model parameters have been update, now user can so whatever he wants.
    return model
# def prediction(days, _gamma = 1/(24*143.5), _beta = 1/(24*4.5), _c = 0.1, _lamda =1/(24*5),  points = 2000):
#     model = SIIR(days, model="SIIRD",death_rate=0.015,total_pop=150e6,beta=_beta, gamma=_gamma,c=_c, lamda=_lamda, alpha0=1/(24*5), isolation=False ,isolation_delay=2, isolation_days=30, risefalltime=2)
#     model.initial(model.total_pop, 20, Qh0=0, R0=3, D0=0)
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
    model.initial = np.array([model.total_pop-i0-d0-r0, i0-r0, 0, r0, d0])
    try:
        if model.country == "France":
            offset = -1
            d = table[:,0]
            model = fitter(model, table[:,0], table[:,1], table[:,2], table[:,3])
        else:
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
    sim_days = 100
    model.t1 = model.t1 + sim_days
    
    #activate a 30 days quarantine:
    model.isolation = False
    model.isolation_time = 250
    model.risefalltime = 1
    # if model.country == "United States":
    #     model.beta = 1/2.456
    #     #model.gamma = 1/60
    #     model.c = 0.1
    #     model.lamda = 1/8
    #it happens today (so at the last data point)
    model.isolation_delay = len(data[model.country][1])
    #how long do they take to get to safety:
    model.alpha0 = 1/6 #1/(3 days)
    T, Y = model.solve(points)
    T -= len(data[model.country][1])-1
    index = np.argmax(Y[:,1]+Y[:,2])
    indexcum = np.argmax(Y[:,1]+Y[:,2]+Y[:,4]+Y[:,5])
    index2 = np.argmax(Y[:,5])
    #save Y in file for later use
    np.save(filebase+f"{model.country}.npy", Y)
    #Default label in English
    title = f"Simulation using a deterministic model, based on current data, for the Covid 19 pandemic in {model.country}. t = 0 is 30th of March."
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
        title = f"Моделирование перехода Covid 19 в России с использованием модели SIIR (восприимчивый, зараженный, изолированный, восстановленный).\nСледующие {sim_days} дней\nt = 0 - 30 марта 2020 г."
        labelcumulative = f"Общее количество случаев (накопительное) в России; Максимальное количество дел =  {(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum,4]+Y[indexcum,5])/1e6:.3f} ({(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum,4]+Y[indexcum,5])/model.total_pop*100:.3f} %) "
        actualcaseslabel = f"Общее количество случаев в России (накопительное)"
        labeldeaths = f"Смертность в России, макс. = {Y[index2,5]/1e6:.3f} миллионов человек ({Y[index2,5] /model.total_pop*100:.3f}%) \n (# число смертей гораздо менее точное, чем другие, поскольку в настоящее время \n недостаточно данных для того, чтобы сделать прогноз с достаточной уверенностью, \n, хотя это не влияет на другие показатели)"
        xlabel = u"Время (дни)\n Примечание. Это прогноз на основе модели, который означает, что по своей природе будут сделаны упрощения,\n и предполагаются предположения о передаче вируса, что означает, что цифры не являются надежными на 100%. \n Существует определенный запас ошибка, которая может быть довольно большой, так как ситуация меняется каждый день. \n Также меры, принятые правительством, могут занять несколько дней, чтобы быть отраженными в данных. \n Сказав это, номера моделей можно рассматривать как наихудший сценарий для России сегодня, если ничего кардинально не изменится \n И это показывает, что для сокращения числа случаев в долгосрочной перспективе необходима продолжительная очень сильная изоляция,\nи что Вы должны относиться к этому вирусу серьезно, так как он явно не исчезнет в течение ДЛИТЕЛЬНОГО времени. \n Текущий источник данных (используется для подгонки модели к текущей ситуации): Минздрав России \n Автор: (Twitter) @emorell96 (Студент ищет докторскую степень по физике)"
        actualdeathslabel = f"Смерти от Covid 19 в России"
        labelrecovered = f"Выздоровели люди в России"
        infectedlabel = f"Зараженные люди в России; максимальная = {(Y[index,1]+Y[index,2])/1e6:.1f} миллионов человек ({(Y[index,1]+Y[index,2])/model.total_pop*100:.2f} %)"

    elif model.country == "Colombia":
        title = f"Simulación de la transimición del Covid 19 en Colombia usando un modelo SIIR (Susceptible, Infectado, Aislado, Recuperado).\nPróximos {sim_days} días.\nt=0 es el 29 de Marzo del 2020."
        labelcumulative = f"Total de casos (cumulativo) en Colombia; Maximo # de Casos = {(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum,4]+Y[indexcum,5])/1e3:.2f} mil personas ({(Y[indexcum,1]+Y[indexcum,2]+Y[indexcum, 4])/model.total_pop*100:.3f} % de la población)"
        actualcaseslabel = f"Numero de casos totales en Colombia (cumulativo)(datos del Instituto Nacional de la Salud)"
        labeldeaths = f"Muertes en {model.country}, max = {Y[index2,5]/1e3:.2f} miles de personas ({Y[index2,5]/model.total_pop*100:.3f} %)\n(# de muertes es un numero que es mucho menos preciso que los demas ya que no hay\nsuficientes datos actualmente para hacer una predicción con buena certitud,\n aunque esto no afecto los demás indicadores)"
        xlabel = u"Tiempo (dias) \nNota: Esto es una prediccion basado en un modelo, lo que significa que por naturaleza se harán simplificaciones, y se asumen hipotesis sobre la transmisión del virus, lo que significa que los numeros no son 100% fiables.\n Hay una cierta margen de error que puede ser bastante grande ya que la situación cambia todos los días. \n Tambien las medidas tomadas por el gobierno puede tomar algo de tiempo en verse reflejado por los datos. \n Diciendo eso, los numeros del model si pueden ser vistos como el peor escenario posible para Colombia actualmente. \n Y demuestra que se necesita continuar con un aislamiento muy fuerte para disminuir el numero de casos a largo plazo, y que se tiene que tomar este virus con seriedad ya que claramente esto no va desaparecer en MUCHO MUCHO tiempo.\nFuente: INS\nAutor:(Twitter) @emorell96 (Estudiante de Doctorado en Física en Estados Unidos)"
        actualdeathslabel = f"Muertes del Covid 19 en {model.country}"
        actualrecoveredlabel = f"Personas recuperadas en {model.country}"
        labelrecovered = f"Predicción de personas recuperadas en {model.country} \n*Este indicador podría mejor en los proximos días ya que aún no hay suficientes datos"
        infectedlabel = f"Numero de personas infectadas en cada día en {model.country}; Max = {(Y[index,1]+Y[index,2])/1e3:.2f} mil personas ({(Y[index,1]+Y[index,2])/model.total_pop*100:.3f} %)"
    elif model.country == "United States":
        xlabel += u"\nDisclaimer: This is a prediction based on a model, which means that by default there will be simplifications and assumptions on the spread of the virus (since you can’t possibly model the interaction of 300 million people!).\nThis means that the numbers will have a certain margin of error, and its prediction will certainly evolve with time and as more measures are taken.\nNow having said that, this model can still be thought as a worst case scenario if NOTHING changes, if they continue to downplay this, and ignore it then you can be sure that we will be closer to what the model predicts than not.\nAnd if anything, this data should show you the gravity of the situation so that you stay at home!\nAnd stay in isolation respecting stay-at-home orders as much as possible because if we don’t a LOT of people will be infected.\nSource: Official reports from state health officials.\nAuthor: (Twitter) @emorell96 (PhD student in Physics at UCSB)."




    print(f"Maximum number of infected (quarantined and not quarantined) for {model.country} is {Y[index,1]+Y[index,2]:.2e} ({(Y[index,1]+Y[index,2])/model.total_pop*100:.2f} %) and happens on day {T[index]:.2f}.")
    
    plt.figure(figsize=(20, 8))
    plt.title(title)
    #plt.plot(T, Y[:,0], label="Susceptible")
    plt.scatter(1+data[model.country][1][:, 0]-len(data[model.country][1]), data[model.country][1][:,1], label = actualcaseslabel)
    plt.plot(T, Y[:,1]  + Y[:,2]+Y[:,4]+Y[:,5], label=labelcumulative)
    
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
    plt.axes().xaxis.set_minor_locator(MultipleLocator(5))
    #plt.savefig(filebase+f"{model.country}.png", dpi=600, quality=95)   
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
