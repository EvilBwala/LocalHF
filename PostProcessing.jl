using NPZ
using LinearAlgebra
using TensorOperations

function overlap(spinlist::Array{Spin}, pattern::Array, systm::Systm)
    state = [i.val for i in spinlist];
    state_v = systm.activation.function_type(state, systm.activation.steepness)
    @tensor overlap = state_v[i]*pattern[i];
    return overlap/length(pattern);
end

