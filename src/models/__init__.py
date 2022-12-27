from .baseline import BaselineModel
from .spectrogram import SpectrogramModel
from .largekernel import LargeKernelModel


def get_model_class(model_name):
    if model_name == "baseline" or model_name == "spectrogram":
        return BaselineModel
    elif model_name == "spectrogram_v2":
        return SpectrogramModel
    elif model_name == "large-kernel":
        return LargeKernelModel
    else:
        raise NotImplementedError("Model is not implemented")
