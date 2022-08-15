using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using ShapML
using ProgressMeter
using DecisionTree: impurity_importance


plotattr("framestyle")

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
datapathFull = "/media/john/HSDATA/datasets/Full"

outpath = "/media/john/HSDATA/analysis_1123"
outpathFull = "/media/john/HSDATA/analysis_full"

isdir(datapath)
isdir(outpath)

RFR = @load RandomForestRegressor pkg=DecisionTree
target=:CO
#target=:CDOM
target_name = String(target)
data_path = joinpath(datapath, target_name)
data_path2 = joinpath(datapath2, target_name)
data_path3 = joinpath(datapath3, target_name)
data_pathFull = joinpath(datapathFull, target_name)


X = CSV.File(joinpath(data_pathFull, "X.csv")) |> DataFrame
y = vec(Array(CSV.File(joinpath(data_pathFull, "y.csv")) |> DataFrame))

Xtest = CSV.File(joinpath(data_pathFull, "Xtest.csv")) |> DataFrame
ytest = vec(Array(CSV.File(joinpath(data_pathFull, "ytest.csv")) |> DataFrame))


for n ∈ names(X)
    if !contains(n, "downwelling")
        println(n)
    end
end



# refs = ["λ_$(i)" for i ∈ 1:462]  # ignore reflectance values since they haven't been helpful
# ignored_for_input = [refs..., targets_vars..., ignorecols..., :MSR_705, :rad_MSR_705, :ilat, :ilon, :row_index, :times]

# for var ∈ ignored_for_input
#     println(var)
# end



outpath_full = "/media/john/HSDATA/analysis_full"
if !isdir(outpath_full)
    mkpath(outpath_full)
end

explore_via_rfr(target,
                targetsDict[target][2],
                targetsDict[target][1],
                X,
                y,
                Xtest,
                ytest,
                outpath_full;
                nfeatures=100,
                name_replacements=name_replacements,
              )



# @showprogress for (target, info) ∈ targetsDict
#     target_name = String(target)
#     data_path = joinpath(datapath, target_name)

#     X = CSV.File(joinpath(data_path, "X.csv")) |> DataFrame
#     y = vec(Array(CSV.File(joinpath(data_path, "y.csv")) |> DataFrame))

#     Xtest = CSV.File(joinpath(data_path, "Xtest.csv")) |> DataFrame
#     ytest = vec(Array(CSV.File(joinpath(data_path, "ytest.csv")) |> DataFrame))

#     explore_via_rfr(target,
#                     targetsDict[target][2],
#                     targetsDict[target][1],
#                     X,
#                     y,
#                     Xtest,
#                     ytest,
#                     outpath;
#                     nfeatures=100
#                     )

#     # explore_model("DecisionTreeRegressor",
#     #               DTR,
#     #               target,
#     #               info[2],
#     #               info[1],
#     #               X,
#     #               y,
#     #               Xtest,
#     #               ytest,
#     #               outpath;
#     #               hpo=false
#     #               )

#     GC.gc()
# end


