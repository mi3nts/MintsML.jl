module CorrelationCalculator

using CSV
using DataFrames
using EnergyStatistics
using InformationMeasures
using Statistics
export calculate_correlations

function calculate_correlations(csv_file::AbstractString)
    
    df = CSV.File(csv_file) |> DataFrame

    # Calculate empirical correlation matrix
    empirical_corr_matrix = cor(Matrix(df))

    
    distance_correlations = Dict{Symbol, Float64}()
    mutual_informations = Dict{Symbol, Float64}()

    
    column_names = names(df)
    num_columns = ncol(df)

    for i in 1:num_columns
        for j in (i + 1):num_columns
            col_x = df[!, i]
            col_y = df[!, j]

            # Calculate distance correlation
            distance_corr = dcor(col_x, col_y)
            pair_key = Symbol(column_names[i], "_", column_names[j])
            distance_correlations[pair_key] = distance_corr

            # Calculate mutual information
            mutual_info = get_mutual_information(col_x, col_y)
            mutual_key = Symbol(column_names[i], "_", column_names[j])
            mutual_informations[mutual_key] = mutual_info
        end
    end

    return empirical_corr_matrix, distance_correlations, mutual_informations
end

end  
