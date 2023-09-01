module DAGGenerator

using CausalInference
using CSV
using DataFrames
using GraphRecipes
using Plots

export generate_dag_plot

function generate_dag_plot(
    csv_file::AbstractString,
    algorithm::Symbol,
    test_type::Symbol,
    p_value::Real = 0.01,
    penalty::Real = 1.0,
    parallel::Bool = true
)
    
    df = CSV.File(csv_file) |> DataFrame

    
    variable_names = names(df)

    if algorithm == :pcalg
        if test_type == :cmi
            graph = pcalg(df, p_value, cmitest)
        elseif test_type == :gauss
            graph = pcalg(df, p_value, gausscitest)
        else
            throw(ArgumentError("Invalid test type"))
        end
    elseif algorithm == :fcialg
        if test_type == :cmi
            graph = fcialg(df, p_value, cmitest)
        elseif test_type == :gauss
            graph = fcialg(df, p_value, gausscitest)
        else
            throw(ArgumentError("Invalid test type"))
        end
    elseif algorithm == :ges
        graph, score = ges(df; penalty=penalty, parallel=parallel)
    else
        throw(ArgumentError("Invalid algorithm choice"))
    end

    
    tp = plot_pc_graph_recipes(graph, variable_names)

    return tp  
end

end 







