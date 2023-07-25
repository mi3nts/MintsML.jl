using MintsML
using Documenter

DocMeta.setdocmeta!(MintsML, :DocTestSetup, :(using MintsML); recursive=true)

makedocs(;
    modules=[MintsML],
    authors="John Waczak",
    repo="https://github.com/mi3nts/MintsML.jl/blob/{commit}{path}#{line}",
    sitename="MintsML.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mi3nts.github.io/MintsML.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mi3nts/MintsML.jl",
    devbranch="main",
)
