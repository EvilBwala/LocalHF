using TensorOperations
using LinearAlgebra
using Distributions
using DataStructures
using Einsum
include("Continuous_Model.jl")

L = 10;
r = 3.5;
N = 50;
Tpas = 0.35;
Tact = 0.65;
tau = 5.0;
resistance = 10.0;
steepness = 20.0;
nr_pats = 10;

mu = zeros(Float64, nr_pats);
sigma = Matrix{Float64}(I, nr_pats, nr_pats);
d = MvNormal(mu, sigma);
patterns = rand(d, N);
patterns_v = logistic_activation(patterns, steepness);

activation = Activation_function(logistic_activation, steepness);

bc = "open";
systm = Systm(N, patterns_v, "periodic", L, Tpas, Tact, tau, resistance, activation);

u = Uniform(0,10);
positions = [rand(u,2) for _ in 1:N];
vals = rand(u, N);

mu = zeros(Float64, N);
sigma = Matrix{Float64}(I, N, N);
d = MvNormal(mu, sigma);
etalist = sqrt(2*systm.Tact)*rand(d, 1);


#spinlist = MutableLinkedList{Spin}();
spinlist = Array{Spin}(undef, N);

for i in 1:N
    pos = Position(positions[i][1], positions[i][2]);
    #push!(spinlist, Spin(i,pos,vals[i],r,Array{Int8}(undef,), Array{Float64}(undef,), Array{Float64}(undef,)));
    spin = Spin(i,pos,vals[i], r, Array{Int8}(undef,), Array{Float64}(undef,), Array{Float64}(undef,), Array{Float64}(undef,), etalist[i]);
    spinlist[i] = spin;
end

for i in 1:N
    spinlist[i].neighbors = find_neighbors(spinlist[i], systm, spinlist);
end

for i in 1:N
    spinlist[i].Jij, spinlist[i].Jijkl, spinlist[i].Jij_flag = connection_matrices_raw(spinlist[i], systm);
end

spinlist = connection_matrices(spinlist, systm);