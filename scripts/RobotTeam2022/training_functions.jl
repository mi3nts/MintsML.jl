function train_basic(y, X̃,
                     ytest, X̃test,
                     longname, savename, mdl,
                     target_name, units, target_long,
                     outpath;
                     suffix="",
                      )

    #######################################################################################
    #                        0. Set up outgoing paths                                     #
    #######################################################################################
    outpathtarget = joinpath(outpath, target_name)
    outpathmodel = joinpath(outpathtarget, savename)
    outpathdefault = joinpath(outpathmodel, "default")
    outpath_featuresreduced = joinpath(outpathmodel, "important_only")
#    outpath_hpo = joinpath(outpathmodel, "hyperparameter_optimized")

    for path ∈ [outpathtarget, outpathmodel, outpathdefault, outpath_featuresreduced] # , outpath_hpo]
        if !isdir(path)
            mkpath(path)
        end
    end

    scitype(y) <: target_scitype(mdl)
    scitype(X̃) <: input_scitype(mdl)

    #######################################################################################
    #   1. Train vanilla model on only reflectance data
    #######################################################################################
    mach_refs_only = machine(mdl, X̃, y)
    fit!(mach_refs_only)


    ŷ = MLJ.predict(mach_refs_only, X̃)  # generate predictions on training set
    ŷtest = MLJ.predict(mach_refs_only, X̃test)  # generate predictions on testing set


    p1 = scatterresult(y, ŷ,
                      ytest, ŷtest;
                      xlabel="True $(target_long) [$(units)]",
                      ylabel="Predicted $(target_long) [$(units)]",
                      plot_title="Fit for $(longname)",)

    savefig(p1, joinpath(outpathdefault, "scatterplt__$(suffix).png"))
    savefig(p1, joinpath(outpathdefault, "scatterplt__$(suffix).svg"))
    savefig(p1, joinpath(outpathdefault, "scatterplt__$(suffix).pdf"))

    p2 = quantilequantile(y, ŷ,
                          ytest, ŷtest;
                          xlabel="True $(target_long) [$(units)]",
                          ylabel="Predicted $(target_long) [$(units)]",
                          title="Fit for $(longname)",)

    savefig(p2, joinpath(outpathdefault, "qq__$(suffix).png"))
    savefig(p2, joinpath(outpathdefault, "qq__$(suffix).svg"))
    savefig(p2, joinpath(outpathdefault, "qq__$(suffix).pdf"))


    # save the model
    MLJ.save(joinpath(outpathdefault, "$(savename)__$(suffix).jls"), mach_refs_only)

    #######################################################################################
    #   2. Compute feature importances
    #######################################################################################
    if reports_feature_importances(mdl)
        fi_pairs = feature_importances(mach_refs_only)  # `:impurity` feature importances
        fi_df = DataFrame()
        fi_df.feature_name = [x[1] for x ∈ fi_pairs]
        fi_df.rel_importance = [x[2] for x ∈ fi_pairs]
        fi_df.rel_importance .= fi_df.rel_importance ./ maximum(fi_df.rel_importance)
        fi_df.plot_name = String.(fi_df.feature_name)


        if name_replacements != nothing
            for row ∈ eachrow(fi_df)
                fname = String(row.feature_name)
                if fname ∈ keys(name_replacements)
                    row.plot_name = name_replacements[fname]
                end
            end
        end


        pfi = rankimportances(fi_df.plot_name[1:25], fi_df.rel_importance[1:25]; ylabel="Normalized Impurity Importance", title="Ranked Feature Importance", xtickfontsize=7)

        savefig(pfi, joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).png"))
        savefig(pfi, joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).svg"))
        savefig(pfi, joinpath(outpath_featuresreduced, "importance_ranking__$(suffix).pdf"))
    end

    return  (; :model=>savename, :features=>suffix, :rsquared=>rsq(ŷtest, ytest), :rmse=>rmse(ŷtest, ytest), :mae=>mae(ŷtest, ytest),)
end

