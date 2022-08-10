using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using ShapML
using ProgressMeter
using DecisionTree: impurity_importance

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
datapath2 = "/media/john/HSDATA/datasets/12-09"
datapath3 = "/media/john/HSDATA/datasets/12-10"
outpath = "/media/john/HSDATA/analysis_1123"

isdir(datapath)
isdir(outpath)


RFR()

RFR = @load RandomForestRegressor pkg=DecisionTree
# target=:CO
target=:CDOM
target_name = String(target)
data_path = joinpath(datapath, target_name)
data_path2 = joinpath(datapath2, target_name)
data_path3 = joinpath(datapath3, target_name)



X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

X2 = CSV.File(joinpath(data_path2, "X.csv")) |> DataFrame
y2 = vec(Array(CSV.File(joinpath(data_path2, "y.csv")) |> DataFrame))

Xtest2 = CSV.File(joinpath(data_path2, "Xtest.csv")) |> DataFrame
ytest2 = vec(Array(CSV.File(joinpath(data_path2, "ytest.csv")) |> DataFrame))

X3 = CSV.File(joinpath(data_path3, "X.csv")) |> DataFrame
y3 = vec(Array(CSV.File(joinpath(data_path3, "y.csv")) |> DataFrame))

Xtest3 = CSV.File(joinpath(data_path3, "Xtest.csv")) |> DataFrame
ytest3 = vec(Array(CSV.File(joinpath(data_path3, "ytest.csv")) |> DataFrame))


X_full = vcat(X, X2, X3)
X_test_full = vcat(Xtest, Xtest2, Xtest3)
y_full = vcat(y, y2, y3)
y_test_full = vcat(ytest, ytest2, ytest3)


outpath_full = "/media/john/HSDATA/analysis_full"
if !isdir(outpath_full)
    mkpath(outpath_full)
end

explore_via_rfr(target,
                targetsDict[target][2],
                targetsDict[target][1],
                X_full,
                y_full,
                X_test_full,
                y_test_full,
                outpath_full;
                nfeatures=100
              )






@showprogress for (target, info) ∈ targetsDict
    target_name = String(target)
    data_path = joinpath(datapath, target_name)

    X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
    y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

    Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
    ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

    explore_via_rfr(target,
                    targetsDict[target][2],
                    targetsDict[target][1],
                    X,
                    y,
                    Xtest,
                    ytest,
                    outpath;
                    nfeatures=100
                    )

    # explore_model("DecisionTreeRegressor",
    #               DTR,
    #               target,
    #               info[2],
    #               info[1],
    #               X,
    #               y,
    #               Xtest,
    #               ytest,
    #               outpath;
    #               hpo=false
    #               )

    GC.gc()
end

