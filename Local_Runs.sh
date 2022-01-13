#!/bin/bash

# Filename information
sourcefile=Run_Local_HF.jl
sourcedirectory=Local_Hopfield
filename=Test1
datafolder=Data

# Fixed simulation parameters
N=1000
r=3.5
tsteps=20
dt=0.05
batches=10
steepness=20.0
resistance=10.0

# Variable parameters
taus=(5.0)
Teffs=(0.5)
fracs=(0.0 0.5 1.0)
nrpats=(1 10 20 50 100)
Ls=(10 20)

cd ..

if [ ! -d "$filename" ]
then
    mkdir $filename
fi
cp ${sourcedirectory}/* ${filename}/
cd $filename

if [ ! -d "$datafolder" ]
then
    mkdir $datafolder
fi
cp * ${datafolder}/
cd $datafolder

for nr_pat in ${nrpats[@]}
do
    for Teff in ${Teffs[@]}
    do
        for frac in ${fracs[@]}
        do
            for tau in ${taus[@]}
            do
                for L in ${Ls[@]}
                do
                    #for trial_idx in $(seq 0 1 $((nr_trials-1)))
                    for pat_idx in $(seq 1 1 $nr_pat)
                    do
                        echo "#!/bin/sh" > submit.sbatch
                        echo "#SBATCH --job-name=CHF.$N.$Teff.$tau" >> submit.sbatch
                        echo "#SBATCH --output=./%j.out" >> submit.sbatch
                        echo "#SBATCH --error=./%j.err" >> submit.sbatch
                        echo "#SBATCH --partition=broadwl" >> submit.sbatch
                        echo "#SBATCH --account=pi-svaikunt" >> submit.sbatch
                        echo "#SBATCH --constraint=ib" >> submit.sbatch
                        echo "#SBATCH --nodes=1" >> submit.sbatch
                        echo "#SBATCH --ntasks-per-core=1" >> submit.sbatch
                        echo "#SBATCH --time=15:00:00" >> submit.sbatch
                        #echo "#SBATCH --mail-type=ALL" >> submit.sbatch
                        #echo "#SBATCH --mail-user=agnish@uchicago.edu" >> submit.sbatch
                        echo "#SBATCH --mem-per-cpu=2000" >> submit.sbatch
                        echo "module load julia" >> submit.sbatch
                        echo "julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "TensorOperations", "Einsum"])'" >> submit.sbatch
                        echo "julia ${sourcefile} $N $L $nr_pat $r $Teff $frac $tau $resistance $steepness $pat_idx $dt $tsteps $batches" >> submit.sbatch
                        qsub submit.sbatch
                        #module load julia
                        #julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "TensorOperations"])'
                        #julia ${sourcefile} $N $L $nr_pat $r $Teff $frac $tau $resistance $steepness $pat_idx $dt $tsteps $batches 
                    done
                done
            done        
        done
    done
done
cd ../..
cd ${sourcedirectory}/

