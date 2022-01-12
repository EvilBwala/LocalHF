using LinearAlgebra
using TensorOperations
using Einsum

mutable struct Position
    x::Float64;
    y::Float64;
end

mutable struct Spin
    label::Int64;       # Labels spin with an integer
    pos::Position;      # Position of the spin
    val::Float64;       # Value of the spin/neuron
    r::Float64;         # Radius of influence of the spin
    neighbors::Array;   # Neighbors of the spin
    Jij::Array;         # The quadratic connections of the spin
    Jij_flag::Array;    # Flag to determine whether the quadratic connection matrix has been processed
                        # 0 for raw and 1 for processed
    Jijkl::Array;       # The quartic connections of the spin
    Jijkl_flag::Array;  # Flag similar to described above
    eta::Float64        # The active field value at the spin
end

struct Activation_function
    function_type::Function
    steepness::Float64
end
struct Systm
    N::Int64                        # Number of spins in the system
    patterns::Array;                # patterns stored in the system
    pattern_type::String;           # Type of patterns, u for unactivated, v for activated
    bc::String;                     # Boundary conditions, "periodic" or "open"
    L::Float64;                     # Lattice constant
    Tpas::Float64;                  # Passive temperature
    Tact::Float64;                  # Active temperature
    tau::Float64;                   # Persistence time
    resistance::Float64             # Resistance
    activation::Activation_function # Activation Function for the neurons
    quartic::String                 # Flag to determine whether to include quartic interactions
end

"""
This is the logistic activation function
"""
function logistic_activation(u::Array, steepness::Float64)
    f_u = 2 ./ (1 .+ exp.(-steepness .* u)) .- 1;
    return f_u;
end

# The following functions are for initializing the system
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
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
    if systm.pattern_type == "u"
        local_pats = systm.activation.function_type(local_pats, systm.activation.steepness);
    end
    l = length(local_spins);
    Jij = zeros(Float64, l, l);
    @tensor begin Jij[a,b] = local_pats[i, a]*local_pats[i, b] end;
    Jij[diagind(Jij)] .= 0.0;
    Jij = (1/l)*Jij;
    Jij = Jij[1, :];
    Jij_flag = 0*Jij;
    
    Jijkl = Array{Float64}(undef,);
    Jijkl_flag = Array{Float64}(undef,);
    if systm.quartic == "on"
        Jijkl = zeros(Float64, l, l, l, l);
        @einsum Jijkl[a,b,c,d] = local_pats[i, a]*local_pats[i, b]*local_pats[i, c]*local_pats[i, d];
        Jijkl = (1/(l*l*l))*Jijkl;
        Jijkl = Jijkl[1,:,:,:]
        Jijkl_flag = 0*Jijkl;
    end
    return Jij, Jijkl, Jij_flag, Jijkl_flag
end


# This averaging of Jij uses less memory but take more time.
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


function averaged_Jij(spinlist::Array{Spin})
    N = length(spinlist);
    Jij_new = zeros(Float64, N, N);
    for i in 1:N
        temp_Jij = zeros(Float64, N, N);
        spin = spinlist[i];
        neighbor_list = spin.neighbors;
        Jij = spin.Jij;
        temp_Jij[spin.label, neighbor_list] = Jij;
        temp_Jij[neighbor_list, spin.label] = Jij;
        Jij_new = Jij_new + temp_Jij;
    end
    Jij_new = Jij_new/2;
    for i in 1:N
        spin = spinlist[i];
        spinlist[i].Jij = Jij_new[spin.label, spin.neighbors];
        spinlist[i].Jij_flag = 0*spinlist[i].Jij .+ 1;
    end
    return spinlist
end

function averaged_Jijkl(spin1::Spin, spin2::Spin, spin3::Spin, spin4::Spin)
    x12 = findall(a->a==spin2.label, spin1.neighbors);
    x13 = findall(a->a==spin3.label, spin1.neighbors);
    x14 = findall(a->a==spin4.label, spin1.neighbors);
    x21 = findall(a->a==spin1.label, spin2.neighbors);
    x23 = findall(a->a==spin3.label, spin2.neighbors);
    x24 = findall(a->a==spin4.label, spin2.neighbors);
    x31 = findall(a->a==spin1.label, spin3.neighbors);
    x32 = findall(a->a==spin2.label, spin3.neighbors);
    x34 = findall(a->a==spin4.label, spin3.neighbors);
    x41 = findall(a->a==spin1.label, spin4.neighbors);
    x42 = findall(a->a==spin2.label, spin4.neighbors);
    x43 = findall(a->a==spin3.label, spin4.neighbors);
    x = [x12,x13,x14,x21,x23,x24,x31,x32,x34,x41,x42,x43]
    y = [i==[] for i in x];
    if any(y) || spin1.Jijkl_flag[x12,x13,x14]==1
        return spin1, spin2, spin3, spin4
    else
        Jijkl1234 = (spin1.Jijkl[x12,x13,x14] + spin2.Jijkl[x21,x23,x24] + spin3.Jijkl[x31,x32,x34] + spin4.Jijkl[x41,x42,x43])/4;
        spin1.Jijkl[x12,x13,x14] = Jijkl1234;
        spin2.Jijkl[x21,x23,x24] = Jijkl1234;
        spin3.Jijkl[x31,x32,x34] = Jijkl1234;
        spin4.Jijkl[x41,x42,x43] = Jijkl1234;
        spin1.Jijkl_flag[x12,x13,x14] .= 1;
        spin2.Jijkl_flag[x21,x23,x24] .= 1;
        spin3.Jijkl_flag[x31,x32,x34] .= 1;
        spin4.Jijkl_flag[x41,x42,x43] .= 1;
        return spin1, spin2, spin3, spin4;
    end
end


function connection_matrices(spinlist::Array{Spin}, systm::Systm)
    N = systm.N;
    """
    for i in 1:N
        for j in 1:N
            spinlist[i], spinlist[j] = averaged_Jij(spinlist[i], spinlist[j]);
        end
    end
    """
    spinlist = averaged_Jij(spinlist);
    if systm.quartic == "on"
        for i in 1:N
            for j in 1:N
                for k in 1:N
                    for l in 1:N
                        spinlist[i],spinlist[j],spinlist[k],spinlist[l] = averaged_Jijkl(spinlist[i], spinlist[j],spinlist[k], spinlist[l]);
                    end
                end
            end
        end
    end
    return spinlist
end

#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------
# Function to update a spin at Random

function update_spin(spin::Spin, systm::Systm, spinlist::Array{Spin}, dt)
    N = systm.N;
    #---------------------------------------------------------------------------------------------------------------------
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
    #---------------------------------------------------------------------------------------------------------------------

    Jij = spin.Jij;
    Jijkl = spin.Jijkl;
    local_spins = spinlist[spin.neighbors];
    

    local_state = [i.val for i in local_spins];
    local_state_v = systm.activation.function_type(local_state, systm.activation.steepness)
    @tensor begin force1 = local_state_v[a]*Jij[a]; end
    if systm.quartic == "on"
        @tensoropt begin force2 = Jijkl[a,b,c]*local_state_v[a]*local_state_v[b]*local_state_v[c]; end
    else
        force2 = 0;
    end
    force = force1 + force2;
    spin.val = spin.val - dt*(force - spin.val/systm.resistance) + whitenoise[spin.label]*sqrt(dt) + spin.eta*dt; 

    spinlist[spin.label] = spin;
    return spinlist
end

#-------------------------------------------------------------------------------------------------------------------------


function update_spin_batch(batchlist::Array, systm::Systm, spinlist::Array{Spin}, dt)
    N = systm.N;
    batchsize = length(batchlist);
    batch_dt = (dt*batchsize)/N;
    Jij_new = zeros(Float64, N, N);

    for j in 1:batchsize
        spin = spinlist[batchlist[j]];
        ng = spin.neighbors;
        Jij_new[spin.label, ng] = spin.Jij;
        Jij_new[ng, spin.label] = spin.Jij;
    end
    #---------------------------------------------------------------------------------------------------------------------
    # Update the whitenoise and AOUP noise in the system
    mu = zeros(Float64, N);
    sigma = Matrix{Float64}(I, N, N);
    d = MvNormal(mu, sigma);
    whitenoise = sqrt(2*systm.Tpas)*rand(d, 1);
    aoup_noise = sqrt(2*systm.Tact)*rand(d, 1);
    eta_state = [i.eta for i in spinlist];
    eta_new = (systm.tau/(systm.tau + batch_dt))*eta_state + (1/(systm.tau + batch_dt))*aoup_noise*sqrt(batch_dt);
    for i in 1:N
        spinlist[i].eta = eta_new[i]
    end
    #---------------------------------------------------------------------------------------------------------------------

    state = [i.val for i in spinlist];
    state_v = systm.activation.function_type(state, systm.activation.steepness)
    force1 = Array{Float64}(undef, N);
    @tensor begin force1[b] = state_v[a]*Jij_new[a,b]; end
    if systm.quartic == "on"
        @tensoropt begin force2 = Jijkl[a,b,c]*local_state_v[a]*local_state_v[b]*local_state_v[c]; end
    else
        force2 = 0;
    end
    force = force1;
    state = state - dt*(force - state/systm.resistance) + whitenoise*sqrt(dt) + eta_new*dt; 

    for i in 1:batchsize
        spinlist[batchlist[i]].val = state[batchlist[i]];
    end
    return spinlist
end