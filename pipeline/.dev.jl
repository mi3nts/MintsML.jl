include("../src/MintsML.jl")
using .MintsML

using CSV



# generate sample datset CSV
# f(x,y) = x^5 + y^3 - x^4 - y^4

# # generate inputs from -1
# N = 500
# x = 2 .* (rand(100) .- 0.5)
# y = 2 .* (rand(100) .- 0.5)

# Data = (; x=x, y=y, z=f.(x,y))

# CSV.write("../assets/test-function.csv", Data)



data = CSV.File(MintsML.test_data_path)
