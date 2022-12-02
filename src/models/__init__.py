from .baseline import BaselineModel
from .largekernel import LargeKernelModel


def get_model_class(model_name):
    if model_name == "baseline":
        return BaselineModel
    elif model_name == "spectrogram":
        return BaselineModel
    elif model_name == "large-kernel":
        return LargeKernelModel
    else:
        raise NotImplementedError("Model is not implemented")
