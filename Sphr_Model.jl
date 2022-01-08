using LinearAlgebra
using TensorOperations

mutable struct Position
    x::Float64;
    y::Float64;
end

mutable struct Spin
    label::Int8;
    pos::Position;
    val::Float64;
    r::Float64;
    neighbors::Array;
    Jij::Array;
    Jijkl::Array;
end

struct Systm
    patterns::Array;
    bc::String;
    L::Float64;
end

"""
This is the logistic activation function
"""
function logistic_activation(u::Array, steepness::Float64)
    f_u = 2 ./ (1 .+ exp.(-steepness .* u)) .- 1;
    return f_u;
end


struct Activation_function
    function_type::Function
    steepness::Float64
end

function distance(spin1::Spin, spin2::Spin, systm::Systm)::Float64
    L = systm.L;
    bc = systm.bc;
    x1 = spin1.pos.x;
    x2 = spin2.pos.x;
    y1 = spin1.pos.y;
    y2 = spin2.pos.y;
    if bc=="open"
        d = sqrt((x1-x2)^2 + (y1-y2)^2);
    elseif bc=="periodic"
        dx = abs(x1-x2)<(L/2) ? abs(x1-x2) : L-abs(x1-x2);
        dy = abs(y1-y2)<(L/2) ? abs(y1-y2) : L-abs(y1-y2);
        d = sqrt(dx^2 + dy^2);
    end
    return d
end

function find_neighbors(spin::Spin, systm::Systm, spinlist::Array{Spin})
    N = length(spinlist);
    neighbor_list = Int64[];
    for i in 1:N
        if spin == spinlist[i]
            continue;
        end
        d = distance(spin, spinlist[i], systm);
        if d<=spin.r
            push!(neighbor_list, spinlist[i].label);
        end
    end
    return neighbor_list
end

function connection_matrices(spin::Spin, systm::Systm)
    local_spins = pushfirst!(spin.neighbors, spin.label);
    local_pats = systm.patterns[:, local_spins];
    l = length(local_spins);
    Jij = zeros(Float64, l, l);
    @tensor begin Jij[a,b] = local_pats[i, a]*local_pats[i, b] end;
    Jij[diagind(Jij)] .= 0.0;
    Jij = (1/l)*Jij;
    Jij = Jij[1, :];
    return Jij
end