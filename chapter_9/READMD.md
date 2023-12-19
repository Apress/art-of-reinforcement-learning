# Chapter 9 - Policy Gradient Methods

This folder includes the following module:
* `reinforce.py` implements the REINFORCE policy gradient algorithm and Monte Carlo method to solve classic control tasks
* `reinforce_baseline.py` implements the REINFORCE with baseline policy gradient algorithm and Monte Carlo method to solve classic control tasks
* `actor_critic.py` implements the Actor-Critic algorithm, an online learning algorithm, to solve classic control tasks
* `actor_critic_with_entropy.py` implements the Actor-Critic algorithm with entropy to encourage exploration in solving classic control tasks
* `actor_critic_atari.py` implements the Actor-Critic algorithm with entropy to solve classic Atari video games
* `gym_env_processor.py` contains functions for environment pre-processing, such as frame resizing, frame stacking, and frame skipping, specifically designed for Atari video games
* `trackers.py` contains code for tracking statistics during training and evaluation


## How to run the code
Please note that the code requires Python 3.10.6 or a higher version to run.


### Execute the code on local machine
Before you get started, ensure that you have the latest version of pip installed on your machine by executing the following command:
```
python3 -m pip install --upgrade pip setuptools
```

To run the code, follow these steps:

1. Open the terminal application in your operating system.
2. Navigate to the specific chapter where you want to execute the code.
3. Install the required packages by using pip and referencing the `requirements.txt` file.
4. Once the packages are installed, you can proceed to execute the individual modules.


Here's an example of how to execute the module for using REINFORCE with baseline to solve the Cart Pole problem in chapter 9.
```
cd <repo_path_on_your_computer>/chapter_9

pip3 install -r requirements.txt

python3 -m reinforce_baseline
```

**Using PyTorch with GPUs:**
If you are utilizing Nvidia GPUs, it is highly recommended to install PyTorch with CUDA by following the instructions provided at https://pytorch.org/get-started/locally/.

It is crucial to acknowledge that training a Actor-Critic agent on Atari games can often be a time-consuming process, requiring hours or even days, particularly when utilizing GPUs. The duration of training depends on various factors, such as the complexity of the problem and the specifications of the computer hardware being used.


### Execute the code on Jupyter notebooks or Google Colab
Here's an example of how to execute the module on Jupyter notebooks or Google Colab. Please ensure that you have uploaded all the necessary module files to the server before proceeding. And you may need to restart the runtime after installing the required packages.
```
!pip3 install -r requirements.txt

!python3 -m reinforce_baseline
```

**Important Note:**
To avoid potential conflicts with Python modules like absl.flags, it's highly recommended that you do not executing multiple modules in the same session when using Jupyter notebooks or Colab.


## Reference Code
* [Deep RL Zoo](https://github.com/michaelnny/deep_rl_zoo)
* [Baselines](https://github.com/openai/baselines)