# Include the module from the pipeline directory
include("../pipeline/generate_dag.jl")

# Specify the path to your CSV file
csv_file_path = "../assets/test-function.csv"

# Choose the algorithm and parameters you want
algorithm_choice = :pcalg # Choose either :pcalg , :fcialg or :ges
test_type = :cmi  # Choose either :cmi or :gauss
p_value = 0.01


# Call the function with the chosen options
DAGGenerator.generate_dag_plot(csv_file_path, algorithm_choice, test_type, p_value)
