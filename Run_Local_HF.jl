using TensorOperations
using LinearAlgebra
using Distributions
using DataStructures
using Einsum
using NPZ
include("Continuous_Model.jl")
include("PostProcessing.jl")

"""
N = parse(Int, ARGS[1]);
L = parse(Float64, ARGS[2]);
nr_pats = parse(Int, ARGS[3]);
r = parse(Float64, ARGS[4]);
Tpas = parse(Float64, ARGS[5]);
Tact = parse(Float64, ARGS[6]);
tau = parse(Float64, ARGS[7]);
resistance = parse(Float64, ARGS[8]);
steepness = parse(Float64, ARGS[9]);
pat_idx = parse(Int, ARGS[10]);
dt = parse(Float64, ARGS[11]);
tsteps = parse(Int, ARGS[12])
batches = parse(Int, ARGS[13])
"""
#---------------------------------------------------------
# For Rough Runs


N = 1000
L = 3.5
nr_pats = 10
r = 3.5;
Tpas = 0.35;
Tact = 0.65;
tau = 5.0;
resistance = 10.0;
steepness = 20.0;
pat_idx = 1;
dt = 0.05;
tsteps = 5;
batches = 10;


"""
mu = zeros(Float64, nr_pats);
sigma = Matrix{Float64}(I, nr_pats, nr_pats);
d = MvNormal(mu, sigma);
patterns = rand(d, N);
patterns_v = logistic_activation(patterns, steepness);
"""

#---------------------------------------------------------

#------------------------------------------------------------------------
# Read patterns and spin positions from Patterns.npz File
#------------------------------------------------------------------------
patfile = npzread("N$(N).nrpats$(nr_pats).npz");
patterns = patfile["Patterns_u"];
patterns_v = logistic_activation(patterns, steepness);
positions = patfile["Positions"]; 

#----------------------------------------------------------------------------------------------------
# Define the systm
#----------------------------------------------------------------------------------------------------
activation = Activation_function(logistic_activation, steepness);
bc = "open";
systm = Systm(N, patterns, "u", "periodic", L, Tpas, Tact, tau, resistance, activation, "off");
#----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------
# Initialize the values of the spins and the eta variables
#----------------------------------------------------------------------------------------------------
vals = patterns[pat_idx,:];

mu = zeros(Float64, N);
sigma = Matrix{Float64}(I, N, N);
d = MvNormal(mu, sigma);
etalist = sqrt(2*systm.Tact)*rand(d, 1);
#----------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initialize the spinlist
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
spinlist = Array{Spin}(undef, N);

for i in 1:N
    pos = Position(positions[1, i], positions[2, i]);
    spin = Spin(i,pos,vals[i], r, Array{Int8}(undef,), Array{Float64}(undef,), Array{Float64}(undef,), Array{Float64}(undef,), Array{Float64}(undef,), etalist[i]);
    spinlist[i] = spin;
end

for i in 1:N
    spinlist[i].neighbors = find_neighbors(spinlist[i], systm, spinlist);
end

@time for i in 1:N
    spinlist[i].Jij, spinlist[i].Jijkl, spinlist[i].Jij_flag, spinlist[i].Jijkl_flag = connection_matrices_raw(spinlist[i], systm);
    #println(i);
end

@time spinlist = connection_matrices(spinlist, systm);

patchy_overlap(spinlist, pat_idx, systm)

"""
tsteps = N;
dt = 0.0001;
@time for i in 1:tsteps
    spin = spinlist[rand(1:N)];
    spinlist = update_spin(spin, systm, spinlist, dt);
    #println(i);
end
"""

#------------------------------------------------------------------------------------------------------------------
# Evolve the system for tsteps
#------------------------------------------------------------------------------------------------------------------


mlist = zeros(Float64, tsteps);
batchsize = Int(N/batches);

for t in 1:tsteps
    @time for i in 1:batches
        global spinlist;
        batchlist = rand(1:N, batchsize);
        spinlist = update_spin_batch(batchlist, systm, spinlist, dt);
    end
    mlist[t] = patchy_overlap(spinlist, 1, systm);
end

npzwrite("mlist.npz", Dict("OverlapList" => mlist));