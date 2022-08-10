using mintsML
using MLJ
using DataFrames, CSV
using ProgressMeter


include("./config.jl")
include("./utils.jl")

outpath = "/media/john/HSDATA/datasets"
ispath(outpath)

function make_datasets(datapath, outpath)
    @showprogress for (target, info) ∈ targetsDict
        # make output
        target_name = String(target)
        date = split(datapath, "/")[end]
        outpath_base = joinpath(outpath, date, target_name)

        # make sure outpath exists
        if !isdir(outpath_base)
            println("making folder $(outpath_base)")
            mkpath(outpath_base)
        end

        #(y, X), (yval, Xval), (ytest, Xtest) = makeDatasets(datapath, target)
        (y, X), (ytest, Xtest) = makeDatasets(datapath, target)

        # now we want to save the data.
        CSV.write(joinpath(outpath_base, "X.csv"), X)
        CSV.write(joinpath(outpath_base, "y.csv"), DataFrame(Dict(target => y)))

        # CSV.write(joinpath(outpath_base, "Xval.csv"), Xval)
        # CSV.write(joinpath(outpath_base, "yval.csv"), DataFrame(Dict(target => yval)))

        CSV.write(joinpath(outpath_base, "Xtest.csv"), Xtest)
        CSV.write(joinpath(outpath_base, "ytest.csv"), DataFrame(Dict(target => ytest)))

    end
end

# do it for 11-23
datapath = "/media/john/HSDATA/processed/11-23"
datapath2 = "/media/john/HSDATA/processed/12-09"
datapath3 = "/media/john/HSDATA/processed/12-10"
ispath(datapath)

#make_datasets(datapath, outpath)
make_datasets(datapath2, outpath)
make_datasets(datapath3, outpath)
