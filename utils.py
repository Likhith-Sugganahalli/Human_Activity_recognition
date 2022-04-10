from pathlib import Path
import json
import os
import tensorflow as tf
from typing import List


def get_project_root() -> str:
    """ function to determine root directory path of the project, given the utils.py is in the root directory"""
    return str(Path(__file__).parent)


def get_json(json_path: str):
        """
            Get config file from json file
        """
        cf = open(json_path)
        json_config = json.load(cf)
        return json_config


def load_tf_labels(txt_file_path: str) -> tf.Tensor:
    """ loads class labels as tensorflow tensors"""
    with tf.io.gfile.GFile(txt_file_path) as f:
        lines = f.readlines()
        kinetics_600_labels_list = [line.strip() for line in lines]
        kinetics_600_labels = tf.constant(kinetics_600_labels_list)
    return kinetics_600_labels

def load_filepaths_from_folder(folder: str):
    """ generator funtion to load files from a given folder"""
    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)
        if filepath is not None:
            yield (filepath)

def load_gif_generator(file_paths: List[str], image_size=(224, 224)):
	"""Loads a gif file into a TF tensor."""

	for file_path in file_paths:
		with tf.io.gfile.GFile(file_path, 'rb') as f:
			video = tf.io.decode_gif(f.read())
			yield video,file_path