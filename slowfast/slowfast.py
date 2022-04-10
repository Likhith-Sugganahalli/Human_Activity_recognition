from typing import List
import numpy as np
import json
from pytorchvideo.models import create_slowfast

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
import torch
from slowfast.env import checkpoint_pathmgr as pathmgr



class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self,config):
        self.config = config
        super().__init__()

    def forward(self, frames: torch.Tensor) -> List[torch.Tensor]:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.config["alpha"]
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list



def load_video(video_path: str,config: json) -> EncodedVideo:
    # The duration of the input clip is also specific to the model.
    clip_duration = (config["num_frames"] * config["sampling_rate"])/config["frames_per_second"]


    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration
    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)
    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    return video_data

def apply_transform(input_video: dict,config: json) -> dict:
    
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(config["num_frames"]),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(config["mean"], config['std']),
                ShortSideScale(
                    size=config["side_size"]
                ),
                CenterCropVideo(config["crop_size"]),
                PackPathway(config)
            ]
        ),
    )
    
    video_data = transform(input_video)
    
    return video_data

def normal_to_sub_bn(checkpoint_sd, model_sd):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        checkpoint_sd (OrderedDict): source dict of parameters.
        model_sd (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    for key in model_sd:
        if key not in checkpoint_sd:
            if "bn.split_bn." in key:
                load_key = key.replace("bn.split_bn.", "bn.")
                bn_key = key.replace("bn.split_bn.", "bn.bn.")
                checkpoint_sd[key] = checkpoint_sd.pop(load_key)
                checkpoint_sd[bn_key] = checkpoint_sd[key]

    for key in model_sd:
        if key in checkpoint_sd:
            model_blob_shape = model_sd[key].shape
            c2_blob_shape = checkpoint_sd[key].shape

            if (
                len(model_blob_shape) == 1
                and len(c2_blob_shape) == 1
                and model_blob_shape[0] > c2_blob_shape[0]
                and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                before_shape = checkpoint_sd[key].shape
                checkpoint_sd[key] = torch.cat(
                    [checkpoint_sd[key]]
                    * (model_blob_shape[0] // c2_blob_shape[0])
                )
    return checkpoint_sd

def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key

def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    scaler=None,
    inflation=False,
    convert_from_caffe2=False,
    epoch_reset=False,
    clear_name_pattern=(),
) -> int:
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        scaler (GradScaler): GradScaler to load the mixed precision scale.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
        epoch_reset (bool): if True, reset #train iterations from the checkpoint.
        clear_name_pattern (string): if given, this (sub)string will be cleared
            from a layer name if it can be matched.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    # Account for the DDP wrapper in the multi-gpu setting.
    ms =  model
    if convert_from_caffe2:
        with pathmgr.open(path_to_checkpoint, "rb") as f:
            caffe2_checkpoint = pickle.load(f, encoding="latin1")
        state_dict = OrderedDict()
        name_convert_func = get_name_convert_func()
        for key in caffe2_checkpoint["blobs"].keys():
            converted_key = name_convert_func(key)
            converted_key = c2_normal_to_sub_bn(converted_key, ms.state_dict())
            if converted_key in ms.state_dict():
                c2_blob_shape = caffe2_checkpoint["blobs"][key].shape
                model_blob_shape = ms.state_dict()[converted_key].shape

                # expand shape dims if they differ (eg for converting linear to conv params)
                if len(c2_blob_shape) < len(model_blob_shape):
                    c2_blob_shape += (1,) * (
                        len(model_blob_shape) - len(c2_blob_shape)
                    )
                    caffe2_checkpoint["blobs"][key] = np.reshape(
                        caffe2_checkpoint["blobs"][key], c2_blob_shape
                    )
                # Load BN stats to Sub-BN.
                if (
                    len(model_blob_shape) == 1
                    and len(c2_blob_shape) == 1
                    and model_blob_shape[0] > c2_blob_shape[0]
                    and model_blob_shape[0] % c2_blob_shape[0] == 0
                ):
                    caffe2_checkpoint["blobs"][key] = np.concatenate(
                        [caffe2_checkpoint["blobs"][key]]
                        * (model_blob_shape[0] // c2_blob_shape[0])
                    )
                    c2_blob_shape = caffe2_checkpoint["blobs"][key].shape

                if c2_blob_shape == tuple(model_blob_shape):
                    state_dict[converted_key] = torch.tensor(
                        caffe2_checkpoint["blobs"][key]
                    ).clone()
                    print(
                        "{}: {} => {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
                else:
                    print(
                        "!! {}: {} does not match {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
            else:
                if not any(
                    prefix in key for prefix in ["momentum", "lr", "model_iter"]
                ):
                    print(
                        "!! {}: can not be converted, got {}".format(
                            key, converted_key
                        )
                    )
        diff = set(ms.state_dict()) - set(state_dict)
        diff = {d for d in diff if "num_batches_tracked" not in d}
        if len(diff) > 0:
            print("Not loaded {}".format(diff))
        ms.load_state_dict(state_dict, strict=False)
        epoch = -1
    else:
        # Load the checkpoint on CPU to avoid GPU mem spike.
        with pathmgr.open(path_to_checkpoint, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        model_state_dict_3d = (
            model.state_dict()
        )
        checkpoint["model_state"] = normal_to_sub_bn(
            checkpoint["model_state"], model_state_dict_3d
        )
        if inflation:
            # Try to inflate the model.
            inflated_model_dict = inflate_weight(
                checkpoint["model_state"], model_state_dict_3d
            )
            ms.load_state_dict(inflated_model_dict, strict=False)
        else:
            if clear_name_pattern:
                for item in clear_name_pattern:
                    model_state_dict_new = OrderedDict()
                    for k in checkpoint["model_state"]:
                        if item in k:
                            k_re = k.replace(
                                item, "", 1
                            )  # only repace first occurence of pattern
                            model_state_dict_new[k_re] = checkpoint[
                                "model_state"
                            ][k]
                            print("renaming: {} -> {}".format(k, k_re))
                        else:
                            model_state_dict_new[k] = checkpoint["model_state"][
                                k
                            ]
                    checkpoint["model_state"] = model_state_dict_new

            pre_train_dict = checkpoint["model_state"]
            model_dict = ms.state_dict()
            # Match pre-trained weights that have same shape as current model.
            pre_train_dict_match = {
                k: v
                for k, v in pre_train_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
            # Weights that do not have match from the pre-trained model.
            not_load_layers = [
                k
                for k in model_dict.keys()
                if k not in pre_train_dict_match.keys()
            ]
            # Log weights that are not loaded with the pre-trained weights.
            if not_load_layers:
                for k in not_load_layers:
                    print("Network weights {} not loaded.".format(k))
            # Load pre-trained weights.
            ms.load_state_dict(pre_train_dict_match, strict=False)
            epoch = -1

            # Load the optimizer state (commonly not done when fine-tuning)
        if "epoch" in checkpoint.keys() and not epoch_reset:
            epoch = checkpoint["epoch"]
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scaler:
                scaler.load_state_dict(checkpoint["scaler_state"])
        else:
            epoch = -1

    return epoch


def load_model(filepath_to_checkpoint: str):
    model = create_slowfast(model_num_class=400)
    load_checkpoint(filepath_to_checkpoint,model)
    return model