import rerun as rr

rr.init("gigapose_rerun_test", spawn=True)


rr.log(
    "asset_1",
    rr.Asset3D(path="/home/luthov/hoi/gigapose_integration/gigapose/gigaPose_datasets/datasets/hope_val_000001/_rerun_glb_cache/obj_000001.glb"),
    static=True,
)

rr.log(
    "asset_2",
    rr.Asset3D(path="/home/luthov/hoi/gigapose_integration/gigapose/gigaPose_datasets/datasets/hope_val_000001/_rerun_glb_cache/obj_000002.glb"),
    static=True,
)

rr.log(
    "asset_1",
    rr.Transform3D(
        translation=[0.0, 0.0, 0.0],
        mat3x3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ),
    static=True,
)

rr.log(
    "asset_2",
    rr.Transform3D(
        translation=[1.0, 0.0, 0.0],
        mat3x3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ),
    static=True,
)