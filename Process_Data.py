import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

filename="LHF_Test2"
datafolder="Data"

os.chdir("../{}".format(filename))
os.chdir("{}".format(datafolder))

#----------------------------------------
# Constant Parameeters
#----------------------------------------
N=1000
r=3.5
tsteps=20
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

count = 0

for nr_pat in nrpats:
    for Teff in Teffs:
        for frac in fracs:
            for tau in taus:
                for L in Ls:
                    ovlpmat = np.zeros((nr_pat, tsteps))
                    p_ovlpmat = np.zeros((nr_pat, tsteps))
                    for pat_idx in np.arange(1, nr_pat+1, 1):
                        fname = "mlist.N{}.L{}.nrpat{}.r{}.Teff{}.frac{}.tau{}.R{}.s{}.patidx{}.dt{}.tsteps{}.b{}.npz".format(N, L, nr_pat, r, Teff, frac, tau, resistance, steepness, pat_idx, dt, tsteps, batches)
                        datafile = np.load(fname)
                        ovlpmat[pat_idx-1, :] = datafile["OverlapList"]
                        p_ovlpmat[pat_idx-1, :] = datafile["PatchyOverlapList"]
                    
                    ovlpmat = np.mean(ovlpmat, 0)
                    p_ovlpmat = np.mean(p_ovlpmat, 0)
                    np.savez("Processed.N{}.L{}.nrpat{}.r{}.Teff{}.frac{}.tau{}.R{}.s{}.npz".format(N, L, nr_pat, r, Teff, frac, tau, resistance, steepness), mlist = ovlpmat, patchymlist = p_ovlpmat, timesteps = dt*np.linspace(1,tsteps+1,tsteps))

                    count = count +1
                    print(count)




