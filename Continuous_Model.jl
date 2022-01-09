using LinearAlgebra
using TensorOperations
using Einsum

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
    Jij_flag::Array;
    Jijkl::Array;
    eta::Float64
end

struct Activation_function
    function_type::Function
    steepness::Float64
end
struct Systm
    N::Int64
    patterns::Array;
    bc::String;
    L::Float64;
    Tpas::Float64;
    Tact::Float64;
    tau::Float64;
    resistance::Float64
    activation::Activation_function
end

"""
This is the logistic activation function
"""
function logistic_activation(u::Array, steepness::Float64)
    f_u = 2 ./ (1 .+ exp.(-steepness .* u)) .- 1;
    return f_u;
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
    pushfirst!(neighbor_list, spin.label);
    return neighbor_list
end

function connection_matrices_raw(spin::Spin, systm::Systm)
    local_spins = spin.neighbors;
    local_pats = systm.patterns[:, local_spins];
    l = length(local_spins);
    Jij = zeros(Float64, l, l);
    @tensor begin Jij[a,b] = local_pats[i, a]*local_pats[i, b] end;
    Jij[diagind(Jij)] .= 0.0;
    Jij = (1/l)*Jij;
    Jij = Jij[1, :];
    Jij_flag = 0*Jij;

    Jijkl = zeros(Float64, l, l, l, l);
    @einsum Jijkl[a,b,c,d] = local_pats[i, a]*local_pats[i, b]*local_pats[i, c]*local_pats[i, d];
    Jijkl = (1/(l*l*l))*Jijkl;
    Jijkl = Jijkl[1,:,:,:]
    return Jij, Jijkl, Jij_flag
end

function averaged_Jij(spin1::Spin, spin2::Spin)
    x1 = findall(a->a==spin2.label, spin1.neighbors);
    x2 = findall(a->a==spin1.label, spin2.neighbors);
    if x1==[] || spin1.Jij[x1] == 1
        return spin1, spin2;
    else
        Jij12 = (spin1.Jij[x1] + spin2.Jij[x2])/2;
        spin1.Jij[x1] = Jij12;
        spin2.Jij[x2] = Jij12;
        spin1.Jij_flag[x1] .= 1;
        spin2.Jij_flag[x2] .= 1;
        return spin1, spin2
    end
end

#function averaged_Jijkl(spin1::Spin, spin2::Spin, spin3::Spin, spin4::Spin)
#    x12 = findall(a->a==spin2.label, spin1.neighbors)


function connection_matrices(spinlist::Array{Spin}, systm::Systm)
    N = systm.N;
    for i in 1:N
        for j in 1:N
            spinlist[i], spinlist[j] = averaged_Jij(spinlist[i], spinlist[j]);
        end
    end
    return spinlist
end






function update_spin(spin::Spin, systm::Systm, spinlist::Array{Spin}, dt)
    N = systm.N;
    #-------------------------------------------------------------------------------------------------
    # Update the whitenoise and AOUP noise in the system
    mu = zeros(Float64, N);
    sigma = Matrix{Float64}(I, N, N);
    d = MvNormal(mu, sigma);
    whitenoise = sqrt(2*systm.Tpas)*rand(d, 1);
    aoup_noise = sqrt(2*systm.Tact)*rand(d, 1);
    eta_state = [i.eta for i in spinlist];
    eta_new = (systm.tau/(systm.tau + dt))*eta_state + (1/(systm.tau + dt))*aoup_noise*sqrt(dt);
    for i in 1:N
        spinlist[i].eta = eta_new[i]
    end
    #-------------------------------------------------------------------------------------------------
    
    Jij = spin.Jij;
    Jijkl = spin.Jijkl;
    local_spins = spinlist[spin.neighbors];
    

    local_state = [i.val for i in local_spins];
    local_state_v = systm.activation.function_type(local_state, systm.activation.steepness)
    @tensor begin force1 = local_state_v[a]*Jij[a]; end
    @tensoropt begin force2 = Jijkl[a,b,c]*local_state_v[a]*local_state_v[b]*local_state_v[c]; end
    force = force1 + force2;
    spin.val = spin.val - dt*(force - spin.val/systm.resistance) + whitenoise[spin.label]*sqrt(dt) + spin.eta*dt; 

    spinlist[spin.label] = spin;
    return spinlist
end


