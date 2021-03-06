# Video Segmentation


<img src="https://github.com/bijustin/video-segmentation/blob/master/example-result.gif" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="800" height="600" />

The mask is the consensus mask generated by our algorithm. The img maps the mask to original video.

Many of the videos used come from the DAVIS dataset:  [DAVIS dataset](https://davischallenge.org/)



Here is the link to our project website: [Project Website Link](https://sites.google.com/umich.edu/video-segmentation/home)

## Setup

The development environment we use is Ubuntu Linux 18.04

The python environment we use is Anaconda. [Link to Anaconda installation](https://docs.anaconda.com/anaconda/install/), remember to set `conda` command globally in bashrc.

First download the whole repo:

```
git clone git@github.com:bijustin/video-segmentation.git
```

Create and activate Anaconda environment:

```
conda create -n video-segmentation python=3.7
conda activate video-segmentation
```

Install the required python package:

```
cd video-segmentation/src
pip install -r requirements.txt
```

### Setup for packages required for NLC algorithm

Our NLC algorithm is adapted from [video-seg](https://github.com/pathak22/videoseg). So our setup steps are similar to them.

Here is our setup:

- Install optical flow

```
cd video-segmentation/src/uNLC/lib/
git clone https://github.com/pathak22/pyflow.git
cd pyflow/
python setup.py build_ext -i
python demo.py    # -viz option to visualize output
```

- Install Dense CRF code

```
cd video-segmentation/src/uNLC/lib/
git clone https://github.com/lucasb-eyer/pydensecrf.git
cd pydensecrf/
python setup.py build_ext -i
```

- Install appearance saliency

```
cd video-segmentation/src/uNLC/lib/
git clone https://github.com/ruanxiang/mr_saliency.git
```

- Install kernel temporal segmentation code

```
# cd video-segmentation/src/uNLC/lib/
# wget http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz
# tar -zxvf kts_ver1.1.tar.gz && mv kts_ver1.1 kts
# rm -f kts_ver1.1.tar.gz

# Edit kts/cpd_nonlin.py to remove weave dependecy. Due to this change, we are shipping the library.
# Included invideo-segmentation/src/uNLC/lib/kts/ . However, it is not a required change if you already have weave installed
# (which is mostly present by default).
```

- Convert them to modules

```
cd video-segmentation/src/uNLC/lib/
cp __init__.py mr_saliency/
cp __init__.py kts/
```

## Run the program

After the setups, we could run the main program `runner.py`

By running

```
cd video-segmentation/src/
python runner.py -h
```
You can see what flags we have and their usages by running:


You should put all the video files under directory `video-segmentation/videos/`

Here are examples of how to run the `runner.py`

### Run without NLC

You can run the all of the algorithms (not including NLC) by:
```
python runner.py --filename [Your video name, e.g. bus.mp4]
```
You do not need to prerun NLC in this case


### Prerun NLC algorithm

You can prerun NLC algorithm by

```
python runner.py --filename [Your video name, e.g. bus.mp4] --prerunNLC True
```

and the numpy array will be saved under `video-segmentation/src/uNLC/maskSeq`

You can also change the batch size of the NLC algorithm by running

```
python runner.py --filename [Your video name, e.g. bus.mp4] --prerunNLC True --NLCbatch [Batch size you want (should be an integer)]
```


### Run the algorithm with NLC

This can only happen after you have prerun NLC algorithm first. After that, you can run the algorithm with NLC by

```
python runner.py --filename [Your video name, e.g. bus.mp4] --prerunNLC True --NLCon True
```
