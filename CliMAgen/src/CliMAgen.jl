"""
    CliMAgen.CliMAgen

Code for preprocesssing data, training a score-based 
diffusion model, and generating samples using the learned 
score function.
"""
module CliMAgen

using ArgParse
using BSON
using CUDA
using Flux
using Functors
using ProgressMeter
using Random
using Statistics
using DelimitedFiles

include("utils.jl")
include("models.jl")
include("networks.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")
include("sampling.jl")
include("preprocessing.jl")

export struct2dict, dict2nt
export VarianceExplodingSDE
export drift, diffusion, marginal_prob, score
export vanilla_score_matching_loss, score_matching_loss
export NoiseConditionalScoreNetwork,ResnetBlockNCSN
export WarmupSchedule, ExponentialMovingAverage
export train!, load_model_and_optimizer, save_model_and_optimizer
export Euler_Maruyama_sampler
export MeanSpatialScaling, StandardScaling, apply_preprocessing, invert_preprocessing
end
