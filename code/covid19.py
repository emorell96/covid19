import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit
from scipy.integrate import odeint
mypath = r"C:\Users\emore\Documents\covid19 analysis\data\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

data = []
for file in onlyfiles:
    path = join(mypath, file)
    data.append((file, np.loadtxt(path, skiprows=1, usecols=(0,1), converters = {1: lambda s: float(s.decode("UTF-8").replace(",", ""))})))
for country in data:
    print(country)

def exponentialgrowth(x, A, tau):
    return A*np.exp(x/tau)

#fit the data to an exponential for multiple number of data points.
days = range(8,30)
Ytime =[]
Yamp = []
for i in days:
    rowtime = []
    rowamp = []
    for country, cases in data:
        # if country == "China.txt":
        #     break
        if len(cases) >= i:
            x = cases[:i,0]
            y = cases[:i,1]
            popt, pcov = curve_fit(exponentialgrowth, x, y)
            rowtime.append((country, float(popt[0]), float(popt[1])))
            #rowamp.append(popt[0])
        else:
            rowtime.append(("", 0,0))
    #print(row)
    Ytime.append(np.asarray(rowtime, dtype=object))
    #Yamp.append(rowamp)
Y = np.concatenate(np.asarray(Ytime))
Y = Y.reshape(len(days), len(onlyfiles), 3)
c = Y[:,7,1]
for i in range(1,len(onlyfiles)):
    y = Y[:,i,2]
    plt.plot(days, y , label=Y[0,i,0])
plt.legend()
plt.show()
#def SIR_Solver(beta, gamma)
##Now let's use the SIR model to see how things could evolve
#Overall constants:
beta = 1/(24*4.375)
gamma = 1/(24*55.5)
N = 39.56e6
tinitial = 0
tfinal = 750 #days
points = 30000
days_wo_isolation = 0
isolation_days = 100

#intial conditions:
S0 = N
I0 = 1190
R0 = 2
X0 = np.array([S0, I0, R0])
#SIR model differential equations
def SIR(X, t):
    sdot = -beta*X[0]/N*X[1]
    idot = beta*X[0]/N*X[1]-gamma*X[1]
    rdot = gamma*X[1]
    return np.array([sdot, idot, rdot])
def betaf(t):
    #function that defined the beta of the transmision model
    #time in hours
    #First part uncontrolled (so no isolation -> tau of 2.5 days)
    tauwoiso = 4.3
    bwoi = 1/(24*tauwoiso)
    tauendiso = tauwoiso+isolation_days*0.5-(t/24-days_wo_isolation-isolation_days)*0.1
    if t < days_wo_isolation*24:
        return bwoi
    #isolation part (tau becomes larger progressively (linear))
    #isolation for 
    if t <= (days_wo_isolation+isolation_days)*24:
        return 1/(24*(tauwoiso+(t/24-days_wo_isolation)*0.5))
    elif tauendiso > 0:
        b = 1/(24*tauendiso)
        if b <= bwoi/2.6:
            return b
        else:
            return bwoi/2.6
    else:
        return bwoi/2.6
def SIRTimeDep(X, t):
    sdot = -1*betaf(t)*X[0]/N*X[1]
    idot = betaf(t)*X[0]/N*X[1]-gamma*X[1]
    rdot = gamma*X[1]
    return np.array([sdot, idot, rdot])
#time to solve pandemia, in hours.
T = np.linspace(tinitial*24, tfinal*24, points)
#solve using odeint
Y = odeint(SIR, X0, T)

np.savetxt("SIR.txt", np.c_[T/24,Y[:,0], Y[:,1], Y[:,2]])
#let's check if beta can be gotten by a fit of the first few data points:
popt, pcov = curve_fit(exponentialgrowth, T[600:]/24, Y[600:,2], p0=(28, 30))
print(popt)

plt.figure()
plt.plot(T/24, Y[:,0], label="Susceptible")
plt.plot(T/24, Y[:,1], label="Infected")
plt.plot(T/24, Y[:,2], label="Recovered")
plt.plot(T/24, Y[:,1]*2/100, label=u"Deaths at 2% rate")
plt.xlabel("Time (days) from today (t=0 is today 21st of March)")
plt.ylabel("Population")
plt.title("Simulation of Infection in California using 10 days for the characteristic time (1/beta)(hypothetical)")
plt.legend()

plt.figure()
plt.plot(T/24, np.vectorize(betaf)(T), label="Transmission Rate (lower better)")
plt.legend()
plt.show()

