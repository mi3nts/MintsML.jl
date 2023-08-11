module MintsML


using RelocatableFolders

const test_data_path = @path normpath(joinpath(@__DIR__, "../assets", "test-function.csv"))
@assert ispath(test_data_path)


# using Metrics
# using MLJ
# using DataFrames, CSV
# using ProgressMeter
# using DecisionTree: impurity_importance


end
