import os
import sys
import argparse
from typing import List
import json
import tqdm
import time

import tensorflow as tf

from utils import get_json,get_project_root,load_filepaths_from_folder,load_tf_labels,load_gif_generator
from movinet.movinet import load_movinet_from_local_path,get_top_k_streaming_labels,process_video_input




def generate_result(model: tf.keras.Model,input_video: tf.Tensor,label_map: tf.Tensor,resolution=224) -> List[str]:

    #process the incoming video, normalize it and resize it
    video = process_video_input(input_video)

    # Create initial states for the stream model
    init_states_fn = model.layers[-1].resolved_object.signatures['init_states']
    init_states = init_states_fn(tf.shape(video[tf.newaxis]))

    # split video into clips to feed into the model
    clips = tf.split(video[tf.newaxis], video.shape[0], axis=1)

    all_logits = []

    print('Running the model on the video...')

    # To run on a video, pass in one frame at a time
    states = init_states
    for clip in tqdm.tqdm(clips):
        # Input shape: [1, 1, 172, 172, 3]
        logits, states = model.predict({**states, 'image': clip}, verbose=0)
        all_logits.append(logits)

    logits = tf.concat(all_logits, 0)
    probs = tf.nn.softmax(logits)
    top_probs, top_labels, top_probs_idx = get_top_k_streaming_labels(probs,label_map)

    return top_labels


def main(model_filepath, videos_filepath) -> None:
    dir_root_path = get_project_root()

    config = get_json(json_path=os.path.join(dir_root_path,"configs","movinet.json"))
    kinetics_600_labels = load_tf_labels(txt_file_path=os.path.join(dir_root_path,"configs","labels.txt"))

    # path_to_gifs = os.path.join(dir_root_path,"gifs")
    # path_model = os.path.join(dir_root_path,"models",movinet_model_name)
    list_gifs = list(load_filepaths_from_folder(videos_filepath))
    generated_videos = load_gif_generator(list_gifs)

    movinet_model_id = config["model_id"]
    movinet_model_name = config["model_name"]
    model = load_movinet_from_local_path(model_filepath)

    #markdown The base input resolution to the model. A good value is 224, but can change based on model size.
    resolution = config["resolution"]
    #markdown The fps of the input video.
    video_fps = config["video_fps"]
    #markdown The fps to display the output plot. Depending on the duration of the input video, it may help to use a lower fps.
    display_fps = config["display_fps"] 

    for video,filename in generated_videos:
        labels  = generate_result(model,video,kinetics_600_labels,resolution)
        print(filename)
        print(labels)
        print("############")
    sys.exit()


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="run movinet model inferences on input gif files .")
    parser.add_argument(
        "--model_path",
        required=True,
        metavar="/path/to/saved_model_folder/",
        help="path to saved model of movinet",
    )
    parser.add_argument(
        "--input_gif_files",
        required=True,
        default="./logs",
        metavar="/path/to/videos_folder/",
        help="path to input gif files",
    )
    args = parser.parse_args()
    main(
        model_filepath=args.model_path,videos_filepath=args.input_gif_files
    )
