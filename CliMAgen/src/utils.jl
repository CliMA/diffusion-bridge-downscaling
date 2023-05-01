"""
    expand_dims(x::AbstractVecOrMat, dims::Int)

Adds `dims` dimensions to the front of x, an `AbstractVecOrMat`.
Similar in spirit to TensorFlow's `expand_dims` function.

# References:
https://www.tensorflow.org/api_docs/python/tf/expand_dims
"""
expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x, (ntuple(i -> 1, dims)..., size(x)...))

"""
    expand_dims(x::Real, kwargs...)

Method of `expand_dims` for a scalar argument.

# References:
https://www.tensorflow.org/api_docs/python/tf/expand_dims
"""
expand_dims(x::Real, kwargs...) = x


"""
    struct2dict(s)

Converts a parameter struct (NamedTuple, or custom struct with
named fields) into a dict recursively

The idea is that parameter structs only contain reals, Booleans, or strings
and hence can be easily unpacked into dictionaries.
"""
struct2dict(s) = Dict(x => struct2dict(getfield(s, x)) for x in fieldnames(typeof(s)))

"""
    struct2dict(s::Union{Real,String,Bool})

Method of `struct2dict` which returns `s` when `s` is a Real, 
Boolean, or string.

The assumption is that parameter structs  only contain reals, 
Booleans, or strings and hence can be easily unpacked into 
dictionaries. This method is invoked
when the element cannot be unpacked further.
"""
struct2dict(s::Union{Real,String,Bool}) = s

"""
    dict2nt(s)

Converts a dict into a NamedTuple.

This is an inverse of `struct2dict`, which converts parameter
structs into dictionaries.
"""
dict2nt(d::Dict) = NamedTuple{Tuple(Symbol(k) for k in keys(d))}((dict2nt.(values(d))...,))

"""
    dict2nt(s::Union{Real,String,Bool})

Method of `dict2nt` which returns `s` when `s` is a Real,
Boolean, or string.

Elements of the dict which are of type
Real, Boolean, or String are stored directly into
the NamedTuple using their corresponding key, without
rearrangement.
"""
dict2nt(s::Union{Real,String,Bool}) = s
