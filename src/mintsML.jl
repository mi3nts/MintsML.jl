module mintsML

using Plots, StatsPlots
# using Metrics
# using MLJ
# using DataFrames, CSV
# using ProgressMeter
# using DecisionTree: impurity_importance


# Write your package code here.
include("plot_defaults.jl")
include("mints_recipes.jl")

export add_mints_theme
export quantilequantile


end
