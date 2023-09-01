include("../pipeline/exploratory_data_analysis.jl")

# Specify the path to your CSV file
csv_file_path = "../assets/test-function.csv"


# Call the function to calculate correlations
empirical_corr, distance_corr, mutual_info = CorrelationCalculator.calculate_correlations(csv_file_path)

# Display the results
println("Empirical Correlation Matrix:")
println(empirical_corr)

println("\nDistance Correlations:")
for (key, value) in distance_corr
    println("$key: $value")
end

println("\nMutual Information:")
for (key, value) in mutual_info
    println("$key: $value")
end
