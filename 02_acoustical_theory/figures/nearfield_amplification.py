import os # for determining file path
import numpy as np # for creating data vectors
import scipy.special # for spherical bessel functions
from matplotlib import pyplot # for plotting

scriptDirectory = os.path.dirname(os.path.realpath(__file__)) # must run entire file
os.environ["PATH"] += os.pathsep + '/usr/local/bin'
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2018/bin/x86_64-darwin'

lmax = 6 # maximum order
num_z = 100 # number of frequencies
zeta = np.logspace(-1.0, 1.0, num=num_z) # log-spaced center frequencies
num_f = 100
freq = np.logspace(1.0, 4, num=num_f) * 2

def spherical_hn1(n,z):
    return scipy.special.spherical_jn(n,z,derivative=False) + 1j * scipy.special.spherical_yn(n,z,derivative=False)

def Flz(l,zeta):
    return zeta * l * spherical_hn1(l, zeta * l)

def Alks0(l,k,s0):
    return k * spherical_hn1(l, k * s0)

def Hlf(l,f):
    return abs(1 - 1 / np.sqrt(1 + np.power(f / (200 * l), l)))

def f2k(f):
    return 2 * np.pi * f / 343

#%% Plot nearfield amplification
FlzPlot = pyplot.figure(figsize=(5, 3.3))
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["text.usetex"] = True
pyplot.rcParams["figure.autolayout"] = True
for l in range(1,lmax+1):
    pyplot.plot(zeta, 20*np.log10(abs(Flz(l,zeta))), str((l-1)/(lmax+1)))
pyplot.xscale('log')
pyplot.xlim(zeta[1],zeta[-1])
pyplot.ylim(-10,110)
pyplot.xticks((0.1,0.2,0.5,1,2,5,10), ('0.1', '0.2', '0.5', '1', '2', '5', '10'), fontsize=12)
pyplot.yticks((0,20,40,60,80,100), ('0', '20', '40', '60', '80', '100'), fontsize=12)
pyplot.xlabel('$\zeta$', fontsize=16)
pyplot.ylabel('$|F_l(\zeta)|$ (dB)', fontsize=16)
pyplot.text(0.14, 5, '$l = 1$', fontsize=14)
pyplot.text(0.34, 58, str(lmax), fontsize=14)
ax = pyplot.axes()
ax.annotate('',xytext =(0.17,11),textcoords = 'data',xy=(0.34,58),xycoords='data',arrowprops=dict(arrowstyle="->"),size=20)
pyplot.show()
FlzPlot.savefig(os.path.join(scriptDirectory,'nearfield_amplification.eps'), format='eps', dpi=1200, transparent=True)

#%% Plot nearfield compensation for 5 cm
s1 = 0.05
Hs1Plot = pyplot.figure(figsize=(5, 3.3))
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["text.usetex"] = True
pyplot.rcParams["figure.autolayout"] = True
for l in range(1,lmax+1):
    pyplot.plot(freq, 20*np.log10(s1 * Hlf(l,freq) * abs(Alks0(l,f2k(freq),s1))), str((l-1)/(lmax+1)))
pyplot.xscale('log')
pyplot.xlim(freq[1],freq[-1])
pyplot.ylim(-100,100)
pyplot.xticks((20,50,100,200,500,1000,2000,5000,10000,20000), ('0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20'), fontsize=12)
pyplot.yticks((-80,-60,-40,-20,0,20,40,60,80), ('$-80$', '$-60$', '$-40$', '$-20$', '0', '20', '40', '60', '80'), fontsize=12)
pyplot.xlabel('Frequency $f$ (kHz)', fontsize=16)
pyplot.ylabel('Magnitude (dB)', fontsize=16)
pyplot.text(65, -10, '$l = 1$', fontsize=14)
pyplot.text(100, 75, str(lmax), fontsize=14)
ax = pyplot.axes()
ax.annotate('',xytext =(90,0),textcoords = 'data',xy=(107,72),xycoords='data',arrowprops=dict(arrowstyle="->"),size=20)
pyplot.show()
Hs1Plot.savefig(os.path.join(scriptDirectory,'nearfield_compensation_5cm.eps'), format='eps', dpi=1200, transparent=True)

#%% Plot nearfield compensation for 1 m
s2 = 1
Hs2Plot = pyplot.figure(figsize=(5, 3.3))
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["text.usetex"] = True
pyplot.rcParams["figure.autolayout"] = True
for l in range(1,lmax+1):
    pyplot.plot(freq, 20*np.log10(s2 * Hlf(l,freq) * abs(Alks0(l,f2k(freq),s2))), str((l-1)/(lmax+1)))
pyplot.xscale('log')
pyplot.xlim(freq[1],freq[-1])
pyplot.ylim(-100,100)
pyplot.xticks((20,50,100,200,500,1000,2000,5000,10000,20000), ('0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20'), fontsize=12)
pyplot.yticks((-80,-60,-40,-20,0,20,40,60,80), ('$-80$', '$-60$', '$-40$', '$-20$', '0', '20', '40', '60', '80'), fontsize=12)
pyplot.xlabel('Frequency $f$ (kHz)', fontsize=16)
pyplot.ylabel('Magnitude (dB)', fontsize=16)
pyplot.text(65, -6, '$l = 1$', fontsize=14)
pyplot.text(100, -98, str(lmax), fontsize=14)
ax = pyplot.axes()
ax.annotate('',xytext =(90,-8),textcoords = 'data',xy=(105,-86),xycoords='data',arrowprops=dict(arrowstyle="->"),size=20)
pyplot.show()
Hs2Plot.savefig(os.path.join(scriptDirectory,'nearfield_compensation_100cm.eps'), format='eps', dpi=1200, transparent=True)