#!/bin/bash

# Filename information
sourcefile=Pat_Builder.jl
sourcedirectory=Local_Hopfield
filename=LHF_Test2
datafolder=Data

# Fixed simulation parameters
N=1000

# Variable parameters
Ls=(10 20)
nrpats=(1 10 20 50 100)


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

for L in ${Ls[@]}
do
    for nr_pat in ${nrpats[@]}
    do
        echo "#!/bin/sh" > submit.sbatch
        echo "#SBATCH --job-name=LHF$N.$nr_pat.$L" >> submit.sbatch
        echo "#SBATCH --output=./%j.out" >> submit.sbatch
        echo "#SBATCH --error=./%j.err" >> submit.sbatch
        echo "#SBATCH --partition=svaikunt" >> submit.sbatch
        echo "#SBATCH --account=pi-svaikunt" >> submit.sbatch
        echo "#SBATCH --constraint=ib" >> submit.sbatch
        echo "#SBATCH --nodes=1" >> submit.sbatch
        echo "#SBATCH --ntasks-per-core=1" >> submit.sbatch
        echo "#SBATCH --time=00:10:00" >> submit.sbatch
        #echo "#SBATCH --mail-type=ALL" >> submit.sbatch
        #echo "#SBATCH --mail-user=agnish@uchicago.edu" >> submit.sbatch
        echo "#SBATCH --mem-per-cpu=2000" >> submit.sbatch
        echo "module load julia" >> submit.sbatch
        echo "julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "TensorOperations", "Einsum"])'" >> submit.sbatch
        echo "julia ${sourcefile} $N $nr_pat $L" >> submit.sbatch
        qsub submit.sbatch
        #module load julia
        #julia -e 'using Pkg; Pkg.add(["NPZ", "Distributions", "TensorOperations", "StatsPlots"])'
        #julia ${sourcefile} $N $nr_pat $L
    done
done

cd ../..
cd ${sourcedirectory}/

