
function _modelregistry()
    return Registry((;
        id = Field(String, name = "ID", formatfn=FeatureRegistries.code_format),
        backbone = Field(Bool, name = "Is backbone"),
        input = Field(Any, name = "Input block", optional = true),
        output = Field(Any, name = "Output block", optional = true),
        description = Field(String, name = "Description",
            formatfn=FeatureRegistries.md_format, optional=true),
        weightsloader = Field(FastAI.Datasets.DatasetLoader, optional=true),
        weightsdownloaded = Field(Bool, name = "Weights downloaded",
            computefn = (row, k) -> !ismissing(row.weightsloader) && FastAI.Datasets.isavailable(row.weightsloader)),
        backend = Field(Symbol, name = "Backend", default = :flux),
        package = Field(Module, name = "Package"),
        constructor = Field(Function, name = "Constructor"),
        tags = Field(Vector{String}, name = "Tags", default = String[])
    ), name = "Models", loadfn=(row; kwargs...) -> row.constructor(row; kwargs...))
end


const MODELS = _modelregistry()
const MODEL_CONFIGS = Set{ModelConfig}()
models() = MODELS


# `ModelConfig` makes it easier to add model definitions to a registry and register their
# weights as a `DataDep`. `register` is called on it in `__init__()`.

Base.@kwdef struct ModelConfig
    id::String
    config::String
    checkpoint = nothing
    input = Any
    output = Any
    description = missing
    package = @__MODULE__
    backend = :pytorch
    tags = String[]
    backbone = false
    size = missing
    checksum = ""
    constructor = _load_from_registry(config)
end

function _load_from_registry(config::String, row; kwargs...)
    if isnothing(row.weightsloader)
        return load_mmpose_model(config)
    else
        return load_mmpose_model(config,
            only(readdir(FastAI.Datasets.loaddata(row.weightsloader), join=true)))
    end
end
_load_from_registry(config::String) = Base.Fix1(_load_from_registry, config)



function register(registry::Registry, config::ModelConfig)
    (; id, config, checkpoint, input, output, description, package,
        backend, backbone, constructor, size, tags, checksum) = config
    datadep = "mmpose_" * replace(id, "/" => "_")
    DataDeps.register(DataDeps.DataDep(
        datadep,
        """
        Weight checkpoint for MMPose (https://github.com/open-mmlab/mmpose) model

            Configuration: "$id"
            Size: $size

        Description:

        $description
        """,
        checkpoint,
        checksum,
    ))
    haskey(registry, id) || push!(registry, (;
        id, backbone, input, output, description, backend, package, constructor, tags,
        weightsloader = FastAI.Datasets.DataDepLoader(datadep)
    ))
end
