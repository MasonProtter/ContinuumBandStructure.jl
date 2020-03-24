module ContinuumBandStructure


using SparseArrays, LinearAlgebra, Arpack
using Base.Iterators: product, flatten
using IterTools: ivec, distinct
using StructArrays

δ(n, m) = n == m ? true : false

export δ, ED, triangular_path, collect_band_structure, collect_spin_projections, ivec, distinct, flatten, product, sparse, apply_over_momenta, get_band_structure, materialize_H


function materialize_H(Hfunc::Function, idx1, idx2; mapper=map)   
    H = mapper(product(idx1, idx2)) do (i, j)
        Hfunc(i, j)
    end
    @assert  H ≈ H'
    Hermitian(H)
end

function materialize_H(Kfunc::Function, idx)   
    (Diagonal ∘ map)(Kfunc, idx)
end

function materialize_H(Kfunc::Function, Vfunc::Function, idx1, idx2; mapper=map)   
    H = mapper(product(idx1, idx2)) do (i, j)
        if i == j
            Kfunc(i) + Vfunc(i, i)
        else
            Vfunc(i, j)
        end
    end
    @assert H' ≈ H
    Hermitian(H)
end

function ED!(H::Hermitian, H_V::Hermitian, Kfunc::Function, idx; kwargs...)
    Hk = materialize_H(Kfunc, idx)
    H.data .= Hk .+ H_V 
    ED(H, kwargs...)
end

function ED(H::Hermitian; nev=6, kwargs...)
    λ, ϕ = eigs(Hermitian(H), nev=nev, which=:SR, kwargs...)
    λs = NamedTuple{Tuple(Symbol("λ$i") for i in 1:nev)}(Tuple(real(λ)))
    ϕs = NamedTuple{Tuple(Symbol("ϕ$i") for i in 1:nev)}(Tuple(ϕ[:, i] for i in 1:nev))
    (λs = λs, ϕs = ϕs)
end


function ED(Kfunc::Function, Vfunc::Function, idx; mapper=map, kwargs...)
    H = materialize_H(Kfunc, Vfunc, idx, idx; mapper=mapper)
    ED(Hermitian(H); kwargs...)
end

function eigenvalues(H::Hermitian; kwargs...)
    λs, _ = ED!(H; kwargs...)
    λs
end

function eigenstate_expectation_values(O::Matrix, H::Hermitian; kwargs...)
    _, ϕs = ED!(H; kwargs...)
    keys = Tuple(Symbol("⟨ϕ$i|O|ϕ$i⟩") for i in 1:length(ϕs))
    vals = (ϕ -> ϕ'O*ϕ).(ϕs)
    NamedTuple{keys}(vals)
end


function apply_over_momenta(f::Function, ks; Kclosure, Vfunc, idx, mapper=map, kwargs...)
    H_V = materialize_H(Vfunc, idx, idx; mapper=mapper)
    H_pre = similar(H_V)
    map(ks) do k
        (f ∘ ED!)(H_pre, H_V, Kclosure(k), idx; kwargs...)
    end
end

get_band_structure(ks; kwargs...) = apply_over_momenta(((λ, ϕ),) -> λ, ks; kwargs...)

#---------------------------------------------------------------



function triangular_path(corners::NTuple{3, Tuple{T,T}}; npts=50) where {T}
    rng(a, b) = Base.range(a, b, length=round(Int, npts/3))
    path1 = zip(rng(corners[1][1], corners[2][1]),
                rng(corners[1][2], corners[2][2]))
    path2 = zip(rng(corners[2][1], corners[3][1]),
                rng(corners[2][2], corners[3][2]))
    path3 = zip(rng(corners[3][1], corners[1][1]),
                rng(corners[3][2], corners[1][2]))
    
    ((distinct ∘ Iterators.flatten)((path1, path2, path3)))
end



end
