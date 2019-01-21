Detect MacBook Air
==================
Use an object detection neural network to detect a Mac in images.

# Instructions
Get this code:

    git clone --recursive https://github.com/WSU-RAS/detect-laptop
    cd detect-laptop

## Labeling images
Then, to label them in Sloth (see my Arch
[PKGBUILD](https://github.com/floft/PKGBUILDs/tree/master/python-sloth)):

    ./gen_sloth.sh
    ./annotate.sh # 'f' to label as laptop, space for next, Ctrl+S to save

Convert TensorFlow {tftrain,tfvalid,tftest}.record files:

    ./sloth2tf.py

## Get pre-trained TensorFlow networks

    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz
    for i in *.tar.gz; do tar xaf $i; done

## Fix bug
To fix [a bug](https://github.com/tensorflow/models/issues/4996#issuecomment-410640308), in *models/research/object_detection/metrics/coco_tools.py* change

    results.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

to

    results.dataset['categories'] = self.dataset['categories']

## Training
Uncomment the model you wish to train in *config.sh*.

Install dependencies:

    sudo pacman -S cython
    pip install --user pycocotools

Then run training and monitor the results:

    ./train.sh
    tensorboard --logdir float:object_detection_models.float,quantized:object_detection_models.quantized,ppn:object_detection_models.ppn

## Export model
Export the model for TensorFlow Lite:

    ./export.sh
