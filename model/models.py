
def get_models(config):
    if "Ours" in config.train.choose_model:
        from .entry import build_model
        model = build_model(config)
        print("Our model for segmentation")
    if "SwinUNETR" in config.train.choose_model:
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            in_channels=len(config.data.use_modalities),
            out_channels=len(config.data.use_modalities),
            img_size=config.data.target_size,
            feature_size=96,
        )
        print("SwinUNETR for segmentation")

    return model
