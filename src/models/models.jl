



"""
    load_mmpose_model(config, [checkpoint])

Load a mmpose model from a configuration file. Return
a `PyCallChainRules.TorchModuleWrapper` that can be used
like any Flux.jl model.
"""
function load_mmpose_model(config, checkpoint = nothing)
    py_model = mmpose_to_sequential(cd(mmposedir()) do
        apis().init_pose_model(config, checkpoint)
    end)
    return TorchModuleWrapper(py_model)
end

"""
    mmpose_to_sequential(m::PyObject) -> PyObject{nn.Sequential}

Turn a Python mmpose model into a simple PyTorch chain.
"""
mmpose_to_sequential(m) = pyimport("torch.nn").Sequential(
    m.backbone,
    m.keypoint_head
)

##

push!(MODEL_CONFIGS, ModelConfig(
    id = "2d_kpt_sview_rgb_img/topdown_heatmap/res50_coco_256x192",
    config = "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py",
    checkpoint = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth",
    description = """
    Single-view human body pose estimation from a 2D, RGB image. Trained
    on the COCO dataset with images of size `(192, 256)`.
    """,
    input = Vision.ImageTensor{2},
    checksum = "ec54d7f3a287169022530c15dc5b5e6d0224e3c7a3881c26231c63f63686c9b3",
))



push!(MODEL_CONFIGS, ModelConfig(
    id = "2d_kpt_sview_rgb_img/topdown_heatmap/mpii/res50_mpii_256x256.py",
    config = "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/res50_mpii_256x256.py",
    checkpoint = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256-418ffc88_20200812.pth",
    description = """
    Single-view human body pose estimation from a 2D, RGB image. Trained
    on the MPII dataset with images of size `(256, 256)`.
    """,
    input = Vision.ImageTensor{2},
    checksum = "ec54d7f3a287169022530c15dc5b5e6d0224e3c7a3881c26231c63f63686c9b3",
))
