module MMPose

import DataDeps
using FilePathsBase
using PyCall
using FastAI
using FastAI.Datasets: DataDepLoader
using FeatureRegistries
using FeatureRegistries: Registry, Field
using PyCallChainRules
using PyCallChainRules.Torch: TorchModuleWrapper

torch() = pyimport("torch")::PyObject
torchvision() = pyimport("torchvision")::PyObject

mmpose() = pyimport("mmpose")
apis() = pyimport("mmpose.apis")
pipelines() = pyimport("mmpose.datasets.pipelines")

mmposedir() = parent(parent(Path(mmpose().__file__)))

include("models/registry.jl")
include("models/models.jl")


function __init__()
    foreach(config -> register(models(), config), MODEL_CONFIGS)
    pyimport("xtcocotools")
end

end
