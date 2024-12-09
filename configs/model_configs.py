from configs.path_configs import MODELS_DIR

FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"  # noqa: E501
LPIPS_WEIGHTS_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"

FID_WEIGHTS_PATH = MODELS_DIR / "pt_inception-2015-12-05-6726825d.pth"
# LPIPS_WEIGHTS_PATH = MODELS_DIR / "alexnet-owt-7be5be79.pth"

CLIP_MODEL_DIR_PATH = MODELS_DIR / "clip"
