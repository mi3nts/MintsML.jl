# we can use this file to look through the HPO results for each model
# and generate a set of reasonable defaults to use


# https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
ENV["GKSwstype"] = "100"


using Pkg
Pkg.instantiate()

using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV

include("./config.jl")
include("./utils.jl")
include("./training_functions.jl")


# set up the paths
path_to_data = "/media/john/HSDATA/analysis_full"
ispath(path_to_data)


model_list = ["DecisionTreeRegressor", "RandomForestRegressor", "XGBoostRegressor", "KNNRegressor", "EvoTreeRegressor"]

for (target, info) ∈ targetsDict
    for model_name ∈ model_list
        path = joinpath(path_to_data, String(target), model_name, "hyperparameter_optimized")
        fpath = joinpath(path, model_name*"__hpo.txt")
        open(fpath, "r") do f
            text = read(f, String)
            println(text)
        end

    end
end

