# PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations
We train a generalist policy for controlling dexterous robot hands to play any songs,
using human pianist demonstration videos from internet. We use residual reinforcement learning to learn song-specific policies from demonstrations, and a two-stage diffusion policy to generalize to new songs.
[![Video](https://i.ytimg.com/vi/LW0AiBIcnL0/hqdefault.jpg)](https://youtu.be/LW0AiBIcnL0)

## Installation and Setup

## Getting Started

We have a tutorial on Google Colab:
[Colab Notebook](https://drive.google.com/file/d/15WXesKKqKEQTMUPSJBYyT-PVknIRsPPZ/view?usp=sharing)

## Installation and Setup

Follow the steps below to set up the PianoMime.

### Step 1: Install Dependencies

1. Open a terminal and run the following command to install the necessary libraries:

    ```sh
    sudo apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
    ```

2. Run the following script to install additional dependencies:

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

### Step 2: Download the Dataset and Checkpoints

1. Download the dataset from the following link:
   https://drive.google.com/file/d/15WXesKKqKEQTMUPSJBYyT-PVknIRsPPZ/view?usp=sharing

2. Download the checkpoints from the following link:
   https://drive.google.com/file/d/1-wa1UAn_mbPN87D6GIi4PS0VNDE5mbQh/view?usp=sharing

## Acknowledgements

The simulation environment is based on RoboPianist ([https://github.com/google-research/robopianist](https://github.com/google-research/robopianist))  
The diffusion policy is adapted from Diffusion Policy ([https://github.com/real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy))  
The inverse-kinematics controller is adapted from Pink ([https://github.com/stephane-caron/pink](https://github.com/stephane-caron/pink))

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
