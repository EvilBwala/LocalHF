using TensorOperations
using LinearAlgebra
using Distributions
using DataStructures
using Einsum
include("Continuous_Model.jl")

L = 10;
r = 3.5;
N = 100;

steepness = 20.0;
nr_pats = 10;

mu = zeros(Float64, nr_pats);
sigma = Matrix{Float64}(I, nr_pats, nr_pats);
d = MvNormal(mu, sigma);
patterns = rand(d, N);
patterns_v = logistic_activation(patterns, steepness);

bc = "open";
u = Uniform(0,10);
systm = Systm(patterns_v, "periodic", L);
positions = [rand(u,2) for _ in 1:N];
vals = rand(u, N);

#spinlist = MutableLinkedList{Spin}();
spinlist = Array{Spin}(undef, N);

for i in 1:N
    pos = Position(positions[i][1], positions[i][2]);
    #push!(spinlist, Spin(i,pos,vals[i],r,Array{Int8}(undef,), Array{Float64}(undef,), Array{Float64}(undef,)));
    spin = Spin(i,pos,vals[i], r, Array{Int8}(undef,), Array{Float64}(undef,), Array{Float64}(undef,));
    spinlist[i] = spin;
end

for i in 1:N
    spinlist[i].neighbors = find_neighbors(spinlist[i], systm, spinlist);
end

for i in 1:5
    spinlist[i].Jij, spinlist[i].Jijkl = connection_matrices(spinlist[i], systm);
end
