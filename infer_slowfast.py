import torch
import os
import argparse
from typing import Dict
from typing import List
import json
from slowfast.slowfast import load_model,apply_transform,load_video

from utils import get_project_root,get_json,load_filepaths_from_folder


def main(model_filepath: str,videos_filepath: str) -> None:
    """ main function serves as the entry point to the script, creates model, processes input and calls prediction on it
    args: 
        (model_filepath)path to savedModel folder
        (video_filepath)path to folder containing input video files
    
    returns:
        None
    """
    root_dir_path = get_project_root()

    config = get_json(os.path.join(root_dir_path,"configs","slowfast.json"))
    
    # videos_filepath = os.path.join(root_dir_path,"gifs")
    list_gifs = list(load_filepaths_from_folder(videos_filepath))
    kinetics_classnames = get_json(os.path.join(root_dir_path,"configs","kinetics_classnames.json"))

    # Set to GPU or CPU
    # model_name = config["model_name"]
    device = "cpu"

    # model_filepath = os.path.join(root_dir_path,"models",model_name)
    
    model = load_model(model_filepath)
    model = model.eval()
    model = model.to(device)

    

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for key, value in kinetics_classnames.items():
        kinetics_id_to_classname[value] = str(key).replace('"', "")

    for video_path in list_gifs:


        video_data = load_video(video_path,config)
        # Apply a transform to normalize the video input
        video_data = apply_transform(video_data,config)

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = [i.to(device)[None, ...] for i in inputs]

        # Pass the input clip through the model
        preds = model(inputs)

        # Get the predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=5).indices[0]

        # Map the predicted classes to the label names
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
        print(video_path)
        print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))
        print("##########")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="run slowfast model inferences on input gif files .")
    parser.add_argument(
        "--model_path",
        required=True,
        metavar="/path/to/saved_model_folder/",
        help="path to pyth of slowfast",
    )
    parser.add_argument(
        "--input_gif_files",
        required=True,
        default="./logs",
        metavar="/path/to/videos_folder/",
        help=" path to input gif files",
    )
    args = parser.parse_args()
    main(
        model_filepath=args.model_path,videos_filepath=args.input_gif_files
    )