# PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations
[[Project page]](https://pianomime.github.io/)
[[Paper]](https://arxiv.org/pdf/2407.18178)
[[Arxiv]](https://arxiv.org/abs/2407.18178)
[[Colab]](https://colab.research.google.com/drive/1Rv1XGPA0a4x3a_M6yXc7uiwKnmmIu95o?usp=sharing)

**Cheng Qian**<sup>1</sup>, **Julen Urain**<sup>2</sup>, **Kevin Zakka**<sup>3</sup>, **Jan Peters**<sup>2</sup>

<sup>1</sup>TU Munich, 
<sup>2</sup>TU Darmsadt, 
<sup>3</sup>UC Berkeley

TLDR:
We train a generalist policy for controlling dexterous robot hands to play any songs,
using human pianist demonstration videos from internet. We use residual reinforcement learning to learn song-specific policies from demonstrations, and a two-stage diffusion policy to generalize to new songs.

[![Video](https://i.ytimg.com/vi/LW0AiBIcnL0/hqdefault.jpg)](https://youtu.be/LW0AiBIcnL0)
## 🚨 News: Dataset Preparation Tutorial Released!

We're thrilled to announce that we've just published a **Tutorial** that walks you through the entire process of preparing your dataset from videos and MIDI files! 🎹🎥

### 📍 Where to find it:
[📓 `tutorial/data_preprocessing.ipynb`](tutorial/data_preprocessing.ipynb)

Inside the notebook, you'll learn how to:
- Estimate homography matrix from video coordinates to real piano coordinates
- Extract fingering and human fingertip trajectories from videos
- Format your data for training

## Getting Started

We have a tutorial on Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Rv1XGPA0a4x3a_M6yXc7uiwKnmmIu95o?usp=sharing)

## Installation and Setup

Follow the steps below to set up the PianoMime.

### Step 1: Clone the Repository
Start by cloning the repository:
    
```sh
git clone https://github.com/sNiper-Qian/pianomime.git
```

### Step 2: Install Dependencies

1. Open a terminal and run the following command to install the necessary libraries:

    ```sh
    sudo apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
    ```

2. Run the following script to install additional dependencies for RoboPianist:

    ```sh
    bash pianomime/scripts/install_deps.sh
    ```

3. Install the Python dependencies by running:

    ```sh
    pip install -r pianomime/requirements.txt
    ```

4. (Optional) Sometimes it is needed to install JAX with the required version:

    ```sh
    pip install --upgrade "jax==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install -U "jaxlib==0.4.23+cuda12.cudnn89" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

### Step 3: Download the Dataset and Checkpoints

1. Download the dataset from the following link:
   https://drive.google.com/file/d/1X8q-PvqyqL2X15wCZevTfAtSDfiHpYAa/view?usp=sharing

2. Download the checkpoints from the following link:
   https://drive.google.com/file/d/1-wa1UAn_mbPN87D6GIi4PS0VNDE5mbQh/view?usp=sharing

## Dataset Preparation
We also provide a tutorial for generate dataset from videos and MIDI files.

You can find the step-by-step guide here:
[Data Preparation Tutorial](tutorial/data_preprocessing.ipynb)

This notebook will walk you through the process of converting your video and MIDI data into a structured dataset, ready for training.
## Citation

Please use the following citation:

```bibtex
@misc{qian2024pianomimelearninggeneralistdexterous,
      title={PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations}, 
      author={Cheng Qian and Julen Urain and Kevin Zakka and Jan Peters},
      year={2024},
      eprint={2407.18178},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.18178}, 
}
```

## Acknowledgements

The simulation environment is based on RoboPianist [RoboPianist](https://github.com/google-research/robopianist)  

The diffusion policy is adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)

The inverse-kinematics controller is adapted from [Pink](https://github.com/stephane-caron/pink)

The human demonstration videos are downloaded from YouTube channel [PianoX](https://www.youtube.com/channel/UCsR6ZEA0AbBhrF-NCeET6vQ)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
