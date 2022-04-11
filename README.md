# Human_Activity_recognition

## Steps to setup environment to run the code
 - clone the repository `git clone git@github.com:Likhith-Sugganahalli/Human_Activity_recognition.git`
 - move into the root directory `cd Human_Activity_recognition/`
 - create a pyenv `python3 -m venv env`
 - source the env `source env/bin/activate`
 - install requirements `python3 -m pip install -r requirements.txt` if this gives any issue with version compatibility, run `python3 -m pip install torchvision torch pytorchvideo tensorflow tensorflow_hub`
 - download MoviNet model files `./download_movinet.sh`
 - download SlowFast model file `./download_slowfast.sh` 

## Running inference
 - on MoviNet `python3 infer_movinet.py --model_path=./movinet_a2_stream_kinetics-600_classification_3/ --input_gif_files=./testing`
 - on SlowFast `python3 infer_slowfast.py --model_path=./SLOWFAST_8x8_R50.pyth --input_gif_files=./testing/`
