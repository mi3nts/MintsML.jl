using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using ShapML
using ProgressMeter



# set the plotting theme
add_mints_theme()
theme(:mints)

# Plots.showtheme(:mints)
# refs = ["λ_$(i)" for i ∈ 1:462]

include("./config.jl")
include("./utils.jl")

# set default resource for parallelization
# MLJ.default_resource(CPUThreads())
MLJ.default_resource(CPUProcesses())


datapath = "/media/john/HSDATA/datasets/11-23"
outpath = "/media/john/HSDATA/analysis"

isdir(datapath)
isdir(outpath)

# target=:CDOM
# target_name = String(target)
# data_path = joinpath(datapath, target_name)

# X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
# y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

# Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
# ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))



# try out functions
RFR = @load RandomForestRegressor pkg=DecisionTree

@showprogress for (target, info) ∈ targetsDict
    target_name = String(target)
    data_path = joinpath(datapath, target_name)

    X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
    y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

    Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
    ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

    explore_model("RandomForestRegressor",
                  RFR,
                  target,
                  info[2],
                  info[1],
                  X,
                  y,
                  Xtest,
                  ytest,
                  outpath;
                  hpo=false
                  )

    GC.gc()
end




# target = :CDOM
# (y, X), (ytest, Xtest) = makeDatasets(datapath, target, 0.85)

# size(X)
# size(Xtest)

# schema(X)

# # load a model and check its type requirements

# plotattr("margins")

# explore_model("RandomForestRegressor",
#               RFR,
#               :CDOM,
#               targetsDict[:CDOM][2],
#               targetsDict[:CDOM][1],
#               X,
#               y,
#               Xtest,
#               ytest,
#               outpath,
#               )
