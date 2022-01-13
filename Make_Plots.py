import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

z = 1
dpival = 2000
font = {'family' : 'Times New Roman'}
line = {'linewidth': 2/z, 'linestyle' : '-'}
matplotlib.rc('font', **font)  # pass in the font dict as kwargs
matplotlib.rc('lines', **line)
matplotlib.rc('axes', labelsize=22/z)
matplotlib.rc('legend', fontsize=14/z)
matplotlib.rc('xtick.major', size=5/z)
matplotlib.rc('ytick.major', size=5/z)
matplotlib.rc('xtick', labelsize=15/z)
matplotlib.rc('ytick', labelsize=15/z)

filename="LHF_Test2"
datafolder="Data"

os.chdir("../{}".format(filename))
os.chdir("{}".format(datafolder))

#----------------------------------------
# Constant Parameeters
#----------------------------------------
N=1000
r=3.5
tsteps=2000
dt=0.05
batches=10
steepness=20.0
resistance=10.0

#----------------------------------------
# Variable parameters
#----------------------------------------
taus=[5.0]
Teffs=[0.1, 0.5]
fracs=[0.0, 0.5, 1.0]
nrpats=[1, 10, 20, 50, 100]
Ls=[10, 20]

if not (os.path.exists('Plots')):
    os.mkdir('Plots')

count = 0
for nr_pat in nrpats:
    for Teff in Teffs:
        for frac in fracs:
            for tau in taus:
                for L in Ls:
                    datafile = np.load("Processed.N{}.L{}.nrpat{}.r{}.Teff{}.frac{}.tau{}.R{}.s{}.npz".format(N, L, nr_pat, r, Teff, frac, tau, resistance, steepness))
                    mlist = datafile["mlist"]
                    patchymlist = datafile["patchymlist"]
                    tlist = datafile["timesteps"]
                    fig, ax = plt.subplots(figsize=(5,5))
                    ax.semilogx(tlist, mlist, label="Total Overlap")
                    ax.semilogx(tlist, patchymlist, label="Patchy Overlap")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Overlap")
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig("Plots/N{}.L{}.nrpat{}.r{}.Teff{}.frac{}.tau{}.R{}.s{}.png".format(N, L, nr_pat, r, Teff, frac, tau, resistance, steepness))
                    plt.savefig("Plots/N{}.L{}.nrpat{}.r{}.Teff{}.frac{}.tau{}.R{}.s{}.eps".format(N, L, nr_pat, r, Teff, frac, tau, resistance, steepness))
                    plt.close()
                    count = count +1
                    print(count)




