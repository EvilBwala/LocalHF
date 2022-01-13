using NPZ
using LinearAlgebra
using TensorOperations

function overlap(spinlist::Array{Spin}, pattern_no::Int64, systm::Systm)
    pattern = systm.patterns[pattern_no, :];
    if systm.pattern_type == "u"
        pattern = systm.activation.function_type(pattern, systm.activation.steepness);
    end
    state = [i.val for i in spinlist];
    state_v = systm.activation.function_type(state, systm.activation.steepness)
    @tensor overlap = state_v[i]*pattern[i];
    return overlap/(norm(state_v)*norm(pattern));
end

function patchy_overlap(spinlist::Array{Spin}, pattern_no::Int64, systm::Systm)
    N = systm.N;
    overlaplist = Array{Float64}(undef, N);
    pattern = systm.patterns[pattern_no, :];
    if systm.pattern_type == "u"
        pattern = systm.activation.function_type(pattern, systm.activation.steepness);
    end
    for i in 1:N
        spin = spinlist[i];
        local_spins = spinlist[spin.neighbors];
        local_state = [j.val for j in local_spins];
        local_state_v = systm.activation.function_type(local_state, systm.activation.steepness);
        local_pattern = pattern[spin.neighbors];
        @tensor ovlp = local_state_v[a]*local_pattern[a];
        overlaplist[i] = ovlp/(norm(local_pattern)*norm(local_state_v));
    end
    return mean(overlaplist)
end

