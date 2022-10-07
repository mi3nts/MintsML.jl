module mintsML

using Plots, StatsPlots
using MLJ
using DataFrames, CSV
using ProgressMeter
using DecisionTree: impurity_importance


# Write your package code here.
include("plot_defaults.jl")
include("mints_recipes.jl")
include("utils.jl")
include("feature_importance.jl")

export add_mints_theme
export makeDatasets
export r²
export quantilequantile
export predict_function
export getFeatureImportances



# load in desired models
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree


# define dictionary with default hyperparameter values used for tuning for each model
hyper_params = Dict(
    "RandomForestRegressor" => [(varname=:(model.n_subfeatures), values=[-1,0]),
                                (varname=:(model.n_trees), lower=10, upper=250),
                                (varname=:(model.sampling_fraction), lower=0.7, upper=0.95)
                                ],
    "XGBoostRegressor" => [(varname=:(model.eta), lower=0.05, upper=0.5),
                           (varname=:(model.lambda), lower=0.1, upper=5),  # L2 regularization. Higher makes model more conservative
                           (varname=:(model.alpha), lower=0.0, upper=1.0, scale=:log), # L1 regularization. Higher makes model more sparse
                           (varname=:(model.max_depth), values=[4,5,6,7,8]),
                           (varname=:(model.num_round), values=[50,100,150]),
                           ]
)




function make_ranges(pipe, modelname)
    rs = []
    for item ∈ hyper_params[modelname]
        if :values ∈ keys(item)
            push!(rs, range(pipe, item.varname, values=item.values))
        else
            push!(rs, range(pipe, item.varname, lower=item.lower, upper=item.upper))
        end
    end
    return rs
end




function hpo_model(model_name,
                   model_loader,
                   target,
                   longname,
                   units,
                   X,
                   y,
                   Xtest,
                   ytest,
                   outpath;
                   impt_features=true,
                   nmodels=200,
                   )

    target_name = String(target)

    println("Setting up save directories...")
    outpathtarget = joinpath(outpath, target_name)
    if !isdir(outpathtarget)
        mkdir(outpathtarget)
    end

    outpathmodel = joinpath(outpathtarget, model_name)
    if !isdir(outpathmodel)
        mkdir(outpathmodel)
    end

    outpathdefault = joinpath(outpathmodel, "default")
    if !isdir(outpathdefault)
        mkdir(outpathdefault)
    end

    path_to_features = joinpath(outpath, target_name, "RandomForestRegressor", "important_only", "importance_ranking.csv")
    isfile(path_to_features)


    outpath_featuresreduced = joinpath(outpathmodel, "important_only")
    if !isdir(outpath_featuresreduced)
        mkdir(outpath_featuresreduced)
    end

    outpath_hpo = joinpath(outpathmodel, "hyperparameter_optimized")
    if !isdir(outpath_hpo)
        mkdir(outpath_hpo)
    end


    # instantiate the model
    println("Instantiating model: $(model_name)...")
    mdl = model_loader()


    if impt_features
        features_df = CSV.File(path_to_features) |> DataFrame

        pipe =  Pipeline(
            selector=FeatureSelector(features=Symbol.(features_df.feature_name)),
            model=mdl
        )


        mach_important = machine(pipe, X, y)  # we bind to all data not in test set
        fit!(mach_important)

        ŷ = MLJ.predict(mach_important, X)
        ŷtest = MLJ.predict(mach_important, Xtest)

        p3 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
        savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.png"))
        savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.svg"))
        savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.pdf"))

        p4 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
        savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.png"))
        savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.svg"))
        savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.pdf"))

        MLJ.save(joinpath(outpath_featuresreduced, "$(model_name)__featuresreduced.jls"), mach_important)

    else
        pipe = Pipeline(model=mdl)
    end

    # verify that the schema (i.e.the scientific types) of our features and targets are compatible with the model
    scitype(y) <: target_scitype(mdl)
    scitype(X) <: input_scitype(mdl)

    println("Data sci-types are compatible with the $(model_name)...")


    # define machine with generic hyperparameters and attach to full set of features.
    println("Performing hyperparameter optimization...")
    rs = make_ranges(pipe, model_name)

    tuning = RandomSearch(rng=42)

    tuned_pipe = TunedModel(
        model=pipe,
        range=rs,
        tuning=tuning,
        measures=[rms, rsq, mae],  #first is used for the optimization but all are stored
        resampling=CV(nfolds=6, rng=42), # this does ~ 85:15 split 6 times
        #resampling=Holdout(fraction_train=0.85, shuffle=true),
        acceleration=CPUThreads(),
        n=nmodels # define the total number of models to try
    )
    # we are skipping repeats to save time...

    # bind to data and train:
    mach_tuned_pipe = machine(tuned_pipe, X, y)

    println("Starting training...")
    fit!(mach_tuned_pipe)

    println("Generating plots...")
    ŷ = MLJ.predict(mach_tuned_pipe, X)
    ŷtest = MLJ.predict(mach_tuned_pipe, Xtest)

    p4 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    savefig(p4, joinpath(outpath_hpo, "scatter_result.png"))
    savefig(p4, joinpath(outpath_hpo, "scatter_result.svg"))
    savefig(p4, joinpath(outpath_hpo, "scatter_result.pdf"))

    p5 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    savefig(p5, joinpath(outpath_hpo, "quantile_quantile.png"))
    savefig(p5, joinpath(outpath_hpo, "quantile_quantile.svg"))
    savefig(p5, joinpath(outpath_hpo, "quantile_quantile.pdf"))

    println("saving model")
    MLJ.save(joinpath(outpath_hpo, "$(model_name)__hpo.jls"), mach_tuned_pipe)


    return (fitted_params(mach_tuned_pipe).best_model.model)
end




function explore_model(model_name,
                       model_loader,
                       target, longname,
                       units,
                       X,
                       y,
                       Xtest,
                       ytest,
                       outpath;
                       hpo=true)

    target_name = String(target)

    println("Setting up save directories...")
    outpathtarget = joinpath(outpath, target_name)
    if !isdir(outpathtarget)
        mkdir(outpathtarget)
    end

    outpathmodel = joinpath(outpathtarget, model_name)
    if !isdir(outpathmodel)
        mkdir(outpathmodel)
    end

    outpathdefault = joinpath(outpathmodel, "default")
    if !isdir(outpathdefault)
        mkdir(outpathdefault)
    end

    outpath_featuresreduced = joinpath(outpathmodel, "important_only")
    if !isdir(outpath_featuresreduced)
        mkdir(outpath_featuresreduced)
    end

    outpath_hpo = joinpath(outpathmodel, "hyperparameter_optimized")
    if !isdir(outpath_hpo)
        mkdir(outpath_hpo)
    end

    println(joinpath(outpath, String(target)))
    # set up directories for saving output


    # instantiate the model
    println("Instantiating model: $(model_name)...")
    mdl = model_loader()

    # verify that the schema (i.e.the scientific types) of our features and targets are compatible with the model
    scitype(y) <: target_scitype(mdl)
    scitype(X) <: input_scitype(mdl)

    println("Data sci-types are compatible with the $(model_name)...")


    # define machine with generic hyperparameters and attach to full set of features.
    mach = machine(mdl, X, y)
    println("Training $(model_name) with default parameters on full feature set...")
    fit!(mach)

    # evaluate the result and save to output files in default directory

    ŷ = MLJ.predict(mach, X)  # generate predictions on training set
    ŷtest = MLJ.predict(mach, Xtest)  # generate predictions on testing set


    p1 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    savefig(p1, joinpath(outpathdefault, "scatter_result.png"))
    savefig(p1, joinpath(outpathdefault, "scatter_result.svg"))
    savefig(p1, joinpath(outpathdefault, "scatter_result.pdf"))

    p2 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    savefig(p2, joinpath(outpathdefault, "quantile_quantile.png"))
    savefig(p2, joinpath(outpathdefault, "quantile_quantile.svg"))
    savefig(p2, joinpath(outpathdefault, "quantile_quantile.pdf"))

    # save the model
    MLJ.save(joinpath(outpathdefault, "$(model_name)__default.jls"), mach)


    # now let us perform the feature selection via shap values
    println("Generating feature importance ranking")
    res_df = getFeatureImportances(Xtest, mach, 50, 0.20)
    CSV.write(joinpath(outpath_featuresreduced, "importance_ranking.csv"), res_df)

    # plot the results
    pfi = rankimportances(res_df.feature_name, res_df.rel_importance; title="Ranked Feature Importance", xtickfontsize=7)
    savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.png"))
    savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.svg"))
    savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.pdf"))

    # train a new model on the reduced feature set
    println("Training $(model_name) on reduced feature list...")
    pipe =  Pipeline(
        selector=FeatureSelector(features=Symbol.(res_df.feature_name)),
        model=mdl
    )

    mach_important = machine(pipe, X, y)  # we bind to all data not in test set
    fit!(mach_important)

    ŷ = MLJ.predict(mach_important, X)
    ŷtest = MLJ.predict(mach_important, Xtest)

    p3 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.png"))
    savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.svg"))
    savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.pdf"))

    p4 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.png"))
    savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.svg"))
    savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.pdf"))

    MLJ.save(joinpath(outpath_featuresreduced, "$(model_name)__featuresreduced.jls"), mach_important)


    if hpo
        println("Performing hyperparameter optimization on reduced feature set...")
        rs = make_ranges(pipe, model_name)

        tuning = RandomSearch(rng=42)

        tuned_pipe = TunedModel(
            model=pipe,
            range=rs,
            tuning=tuning,
            measures=[rsq, rms, mae],  #first is used for the optimization but all are stored
            resampling=CV(nfolds=6, rng=42), # this does ~ 85:15 split 6 times
            #resampling=Holdout(fraction_train=0.85, shuffle=true),
            acceleration=CPUThreads(),
            n=200 # define the total number of models to try
        )
        # we are skipping repeats to save time...

        # bind to data and train:
        mach_tuned_pipe = machine(tuned_pipe, X, y)

        println("Starting training...")
        fit!(mach_tuned_pipe)

        println("Generating plots...")
        ŷ = MLJ.predict(mach_tuned_pipe, X)
        ŷtest = MLJ.predict(mach_tuned_pipe, Xtest)

        p5 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
        savefig(p5, joinpath(outpath_hpo, "scatter_result.png"))
        savefig(p5, joinpath(outpath_hpo, "scatter_result.svg"))
        savefig(p5, joinpath(outpath_hpo, "scatter_result.pdf"))

        p6 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
        savefig(p6, joinpath(outpath_hpo, "quantile_quantile.png"))
        savefig(p6, joinpath(outpath_hpo, "quantile_quantile.svg"))
        savefig(p6, joinpath(outpath_hpo, "quantile_quantile.pdf"))

        println("saving model")
        MLJ.save(joinpath(outpath_hpo, "$(model_name)__hpo.jls"), mach_tuned_pipe)

    end

end







function train_vanilla(model_name,
                       model_loader,
                       target,
                       longname,
                       units,
                       X,
                       y,
                       Xtest,
                       ytest,
                       outpath;
                       featuresreduced=false
                       )

    target_name = String(target)

    println("\tSetting up save directories...")
    outpathtarget = joinpath(outpath, target_name)
    if !isdir(outpathtarget)
        mkdir(outpathtarget)
    end

    outpathmodel = joinpath(outpathtarget, model_name)
    if !isdir(outpathmodel)
        mkdir(outpathmodel)
    end

    outpathdefault = joinpath(outpathmodel, "default")
    if !isdir(outpathdefault)
        mkdir(outpathdefault)
    end

    outpath_featuresreduced = joinpath(outpathmodel, "important_only")
    if !isdir(outpath_featuresreduced)
        mkdir(outpath_featuresreduced)
    end

    if featuresreduced
        path_to_use = outpath_featuresreduce
    else
        path_to_use = outpathdefault
    end


    # instantiate the model
    println("\tInstantiating model: $(model_name)...")
    mdl = model_loader()

    # verify that the schema (i.e.the scientific types) of our features and targets are compatible with the model
    scitype(y) <: target_scitype(mdl)
    scitype(X) <: input_scitype(mdl)


    # define machine with generic hyperparameters
    println("\ttraining...")
    mach = machine(mdl, X, y)
    fit!(mach)

    # evaluate the result and save to output files in default directory
    println("\tevaluating...")
    ŷ = MLJ.predict(mach, X)  # generate predictions on training set
    ŷtest = MLJ.predict(mach, Xtest)  # generate predictions on testing set


    p1 = scatterresult(y, ŷ,
                       ytest, ŷtest;
                       plot_title="Fit for $(model_name)",
                       xlabel="True $(longname) [$(units)]",
                       ylabel="Predicted $(longname) [$(units)]",
                       )

    savefig(p1, joinpath(path_to_use, "scatterplt.png"))
    savefig(p1, joinpath(path_to_use, "scatterplt.svg"))
    savefig(p1, joinpath(path_to_use, "scatterplt.pdf"))


    p2 = quantilequantile(y, ŷ,
                          ytest, ŷtest;
                          title="Fit for $(model_name)",
                          xlabel="True $(longname) [$(units)]",
                          ylabel="Predicted $(longname) [$(units)]"
                          )

    savefig(p2, joinpath(path_to_use, "quantile_quantile.png"))
    savefig(p2, joinpath(path_to_use, "quantile_quantile.svg"))
    savefig(p2, joinpath(path_to_use, "quantile_quantile.pdf"))

    # save the model
    println("\tsaving output...")
    MLJ.save(joinpath(path_to_use, "$(model_name)__vanilla.jls"), mach)


    println("\tdone!")
    # return r²(ŷ, y), r²(ŷtest, ytest)
    return rsq(ŷ, y), rsq(ŷtest, ytest), rmse(ŷtest, ytest), mae(ŷtest, ytest)
end





"""
    function explore_via_rfr(target, logname, units, X, y, Xtest, ytest, outpath; min_allowed=0.1,)

"""
function explore_via_rfr(target,
                         longname,
                         units,
                         X,
                         y,
                         Xtest,
                         ytest,
                         outpath;
                         nfeatures=200,
                         name_replacements=nothing
                          )


    model_name = "RandomForestRegressor"
    model_loader = @load RandomForestRegressor pkg=DecisionTree


    target_name = String(target)

    println("Setting up save directories...")
    outpathtarget = joinpath(outpath, target_name)
    if !isdir(outpathtarget)
        mkpath(outpathtarget)
    end

    outpathmodel = joinpath(outpathtarget, model_name)
    if !isdir(outpathmodel)
        mkpath(outpathmodel)
    end

    outpathdefault = joinpath(outpathmodel, "default")
    if !isdir(outpathdefault)
        mkpath(outpathdefault)
    end

    outpath_featuresreduced = joinpath(outpathmodel, "important_only")
    if !isdir(outpath_featuresreduced)
        mkpath(outpath_featuresreduced)
    end

    outpath_hpo = joinpath(outpathmodel, "hyperparameter_optimized")
    if !isdir(outpath_hpo)
        mkpath(outpath_hpo)
    end

    println(joinpath(outpath, String(target)))
    # set up directories for saving output


    # instantiate the model
    println("Instantiating model: $(model_name)...")
    # mdl = model_loader(max_depth=10, n_trees=100, n_subfeatures=0, rng=42)
    mdl = model_loader(n_subfeatures=0, rng=42)

    # verify that the schema (i.e.the scientific types) of our features and targets are compatible with the model
    scitype(y) <: target_scitype(mdl)
    scitype(X) <: input_scitype(mdl)

    println("Data sci-types are compatible with the $(model_name)...")


    #######################################################################################
    #                        1. Train vanilla RFR model                                  #
    #######################################################################################


    # define machine with generic hyperparameters and attach to data with full set of features.
    mach = machine(mdl, X, y)
    println("Training $(model_name) with default parameters on full feature set...")
    res = fit!(mach)

    # evaluate the result and save to output files in default directory

    ŷ = MLJ.predict(mach, X)  # generate predictions on training set
    ŷtest = MLJ.predict(mach, Xtest)  # generate predictions on testing set


    p1 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    savefig(p1, joinpath(outpathdefault, "scatter_result.png"))
    savefig(p1, joinpath(outpathdefault, "scatter_result.svg"))
    savefig(p1, joinpath(outpathdefault, "scatter_result.pdf"))

    p2 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    savefig(p2, joinpath(outpathdefault, "quantile_quantile.png"))
    savefig(p2, joinpath(outpathdefault, "quantile_quantile.svg"))
    savefig(p2, joinpath(outpathdefault, "quantile_quantile.pdf"))

    # save the model
    MLJ.save(joinpath(outpathdefault, "$(model_name)__default.jls"), mach)


    #######################################################################################
    #                        2. Reduce features using MDI                                 #
    #######################################################################################
    features = propertynames(X)
    fi = impurity_importance(mach.fitresult)
    fi_pairs = collect(Dict(zip(features, fi)))
    # sort descending
    sort!(fi_pairs, by= x->-x[2])

    fi_df = DataFrame()
    fi_df.feature_name = [x[1] for x ∈ fi_pairs]
    fi_df.rel_importance = [x[2] for x ∈ fi_pairs]
    fi_df.rel_importance .= fi_df.rel_importance ./ maximum(fi_df.rel_importance)

    fi_df = fi_df[1:nfeatures, :]

    CSV.write(joinpath(outpath_featuresreduced, "importance_ranking.csv"), fi_df)


    # create varnames for printing
    fi_df.plot_name = String.(fi_df.feature_name)

    if name_replacements != nothing
        for row ∈ eachrow(fi_df)
            fname = String(row.feature_name)
            if fname ∈ keys(name_replacements)
                row.plot_name = name_replacements[fname]
            end
        end
    end


    if nfeatures > 25
        pfi = rankimportances(fi_df.plot_name[1:25], fi_df.rel_importance[1:25]; ylabel="Normalized Mean Decrease in Impurity", title="Ranked Feature Importance", xtickfontsize=7)
    else
        pfi = rankimportances(fi_df.plot_name, fi_df.rel_importance; ylabel="Normalized Mean Decrease in Impurity", title="Ranked Feature Importance", xtickfontsize=7)
    end

    savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.png"))
    savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.svg"))
    savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.pdf"))

    # mdl = model_loader(max_depth=10, n_trees=100, n_subfeatures=0, rng=42)
    mdl = model_loader(n_subfeatures=0, rng=42)
    println("Training $(model_name) on reduced feature list...")
    pipe =  Pipeline(
        selector=FeatureSelector(features=fi_df.feature_name),
        model=mdl
    )

    mach_important = machine(pipe, X, y)  # we bind to all data not in test set
    fit!(mach_important)

    ŷ = MLJ.predict(mach_important, X)
    ŷtest = MLJ.predict(mach_important, Xtest)

    p3 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.png"))
    savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.svg"))
    savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.pdf"))

    p4 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.png"))
    savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.svg"))
    savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.pdf"))

    MLJ.save(joinpath(outpath_featuresreduced, "$(model_name)__featuresreduced.jls"), mach_important)


    #######################################################################################
    #        3. Train Seperate Vanilla model w/ standardized inputs         #
    #######################################################################################

    Standardizer = @load Standardizer pkg=MLJModels
    # pipe = Standardizer() |> model_loader(max_depth=10, n_trees=100, n_subfeatures=0, rng=42)
    pipe = Standardizer() |> model_loader(n_subfeatures=0, rng=42)
    mach = machine(pipe, X, y)  # we bind to all data not in test set
    fit!(mach)

    ŷ = MLJ.predict(mach, X)  # generate predictions on training set
    ŷtest = MLJ.predict(mach, Xtest)  # generate predictions on testing set

    p1 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    savefig(p1, joinpath(outpathdefault, "scatter_result__standardized.png"))
    savefig(p1, joinpath(outpathdefault, "scatter_result__standardized.svg"))
    savefig(p1, joinpath(outpathdefault, "scatter_result__standardized.pdf"))

    p2 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    savefig(p2, joinpath(outpathdefault, "quantile_quantile__standardized.png"))
    savefig(p2, joinpath(outpathdefault, "quantile_quantile__standardized.svg"))
    savefig(p2, joinpath(outpathdefault, "quantile_quantile__standardized.pdf"))

    # save the model
    MLJ.save(joinpath(outpathdefault, "$(model_name)__default_standardized.jls"), mach)


    #######################################################################################
    #                        4. Reduce features using MDI and Standardization             #
    #######################################################################################
    #mdl = model_loader(rng=42)
    # pipe2 =  Pipeline(
    #     selector=FeatureSelector(features=fi_df.feature_name),
    #     standardizer=Standardizer(),
    #     model=mdl,
    # )
    # pipe2 = FeatureSelector(features=fi_df.feature_name) |> Standardizer() |> model_loader(max_depth=10, n_trees=100, n_subfeatures=0, rng=42)
    pipe2 = FeatureSelector(features=fi_df.feature_name) |> Standardizer() |> model_loader(n_subfeatures=0, rng=42)
    println(pipe2)

    mach_important2 = machine(pipe2, X, y)  # we bind to all data not in test set
    fit!(mach_important2)

    ŷ = MLJ.predict(mach_important2, X)
    ŷtest = MLJ.predict(mach_important2, Xtest)

    p3 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    savefig(p3, joinpath(outpath_featuresreduced, "scatter_result__standardized.png"))
    savefig(p3, joinpath(outpath_featuresreduced, "scatter_result__standardized.svg"))
    savefig(p3, joinpath(outpath_featuresreduced, "scatter_result__standardized.pdf"))

    p4 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile__standardized.png"))
    savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile__standardized.svg"))
    savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile__standardized.pdf"))

    MLJ.save(joinpath(outpath_featuresreduced, "$(model_name)__featuresreduced_standardized.jls"), mach_important2)

    # CSV.write(joinpath(outpath_featuresreduced, "importance_ranking.csv"), fi_df)

    # if nfeatures > 25
    #     pfi = rankimportances(String.(fi_df.feature_name[1:25]), fi_df.rel_importance[1:25]; ylabel="Normalized Mean Decrease in Impurity", title="Ranked Feature Importance", xtickfontsize=7)
    # else
    #     pfi = rankimportances(String.(fi_df.feature_name), fi_df.rel_importance; ylabel="Normalized Mean Decrease in Impurity", title="Ranked Feature Importance", xtickfontsize=7)
    # end

    # savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.png"))
    # savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.svg"))
    # savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.pdf"))

    # mdl = model_loader(rng=42)
    # println("Training $(model_name) on reduced feature list...")
    # pipe =  Pipeline(
    #     selector=FeatureSelector(features=fi_df.feature_name),
    #     model=mdl
    # )

    # mach_important = machine(pipe, X, y)  # we bind to all data not in test set
    # fit!(mach_important)

    # ŷ = MLJ.predict(mach_important, X)
    # ŷtest = MLJ.predict(mach_important, Xtest)

    # p3 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    # savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.png"))
    # savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.svg"))
    # savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.pdf"))

    # p4 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    # savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.png"))
    # savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.svg"))
    # savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.pdf"))

    # MLJ.save(joinpath(outpath_featuresreduced, "$(model_name)__featuresreduced.jls"), mach_important)












    # ŷ = MLJ.predict(mach_important, X)
    # ŷtest = MLJ.predict(mach_important, Xtest)

    # p3 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    # savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.png"))
    # savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.svg"))
    # savefig(p3, joinpath(outpath_featuresreduced, "scatter_result.pdf"))

    # p4 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    # savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.png"))
    # savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.svg"))
    # savefig(p4, joinpath(outpath_featuresreduced, "quantile_quantile.pdf"))

    #MLJ.save(joinpath(outpath_featuresreduced, "$(model_name)__featuresreduced.jls"), mach_important)


    # pipe =  Pipeline(
    #     selector=FeatureSelector(features=fi_df.feature_name),
    #     model=mdl
    # )

    # PCA = @load PCA pkg=MultivariateStats
    # ICA = @load ICA pkg=MultivariateStats
    # pipe = 


    # # now let us perform the feature selection via shap values
    # println("Generating feature importance ranking")
    # res_df = getFeatureImportances(Xtest, mach, 50, 0.20)
    # CSV.write(joinpath(outpath_featuresreduced, "importance_ranking.csv"), res_df)

    # # plot the results
    # pfi = rankimportances(res_df.feature_name, res_df.rel_importance; title="Ranked Feature Importance", xtickfontsize=7)
    # savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.png"))
    # savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.svg"))
    # savefig(pfi, joinpath(outpath_featuresreduced, "importanc_ranking.pdf"))

    # # train a new model on the reduced feature set
    # println("Training $(model_name) on reduced feature list...")
    # pipe =  Pipeline(
    #     selector=FeatureSelector(features=Symbol.(res_df.feature_name)),
    #     model=mdl
    # )





    # if hpo
    #     println("Performing hyperparameter optimization on reduced feature set...")
    #     rs = make_ranges(pipe, model_name)

    #     tuning = RandomSearch(rng=42)

    #     tuned_pipe = TunedModel(
    #         model=pipe,
    #         range=rs,
    #         tuning=tuning,
    #         measures=[rsq, rms, mae],  #first is used for the optimization but all are stored
    #         resampling=CV(nfolds=6, rng=42), # this does ~ 85:15 split 6 times
    #         #resampling=Holdout(fraction_train=0.85, shuffle=true),
    #         acceleration=CPUThreads(),
    #         n=200 # define the total number of models to try
    #     )
    #     # we are skipping repeats to save time...

    #     # bind to data and train:
    #     mach_tuned_pipe = machine(tuned_pipe, X, y)

    #     println("Starting training...")
    #     fit!(mach_tuned_pipe)

    #     println("Generating plots...")
    #     ŷ = MLJ.predict(mach_tuned_pipe, X)
    #     ŷtest = MLJ.predict(mach_tuned_pipe, Xtest)

    #     p5 = scatterresult(y, ŷ, ytest, ŷtest; plot_title="$(model_name) for $(longname)")
    #     savefig(p5, joinpath(outpath_hpo, "scatter_result.png"))
    #     savefig(p5, joinpath(outpath_hpo, "scatter_result.svg"))
    #     savefig(p5, joinpath(outpath_hpo, "scatter_result.pdf"))

    #     p6 = quantilequantile(y, ŷ, ytest, ŷtest; title="$(model_name) plot for $(longname)")
    #     savefig(p6, joinpath(outpath_hpo, "quantile_quantile.png"))
    #     savefig(p6, joinpath(outpath_hpo, "quantile_quantile.svg"))
    #     savefig(p6, joinpath(outpath_hpo, "quantile_quantile.pdf"))

    #     println("saving model")
    #     MLJ.save(joinpath(outpath_hpo, "$(model_name)__hpo.jls"), mach_tuned_pipe)

    # end

end














export make_ranges
export hpo_model
export explore_model
export train_vanilla
export explore_via_rfr


end
