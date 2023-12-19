# Chapter 7 - Nonlinear Value Function Approximation

This folder includes the following module:
* `naive_deep_q.py` implements fully-connected neural network to approximate the state-action value function for solving classic control tasks such as cart pole or mountain car
* `deep_q_with_replay.py` implements fully-connected neural network to approximate the state-action value function, with experience replay to solve classic control tasks like cart pole or mountain car
* `deep_q_with_targetnet.py` implements fully-connected neural network to approximate the state-action value function, with target Q network to solve classic control tasks like cart pole or mountain car
* `dqn_classic.py` implements the DQN agent with a fully-connected neural network, experience replay, and target Q network to solve classic control tasks like cart pole or mountain car
* `dqn_atari.py` implements the DQN agent with a convolutional neural network architecture, experience replay, and target Q network to solve classic Atari video games
* `gym_env_processor.py` contains functions for environment pre-processing, such as frame resizing, frame stacking, and frame skipping, specifically designed for Atari video games
* `replay.py` contains code for implementing experience replay
* `trackers.py` contains code for tracking statistics during training and evaluation
* `example_of_training_nn.py` provides minimum code to show how to train a neural network using PyTorch
* `visualize_atari_env_perprocessing.py` provides code for visualizing the environment pre-processing steps performed on Atari games
* `visualize_activation_functions.py` provides code for visualizing come common activation functions


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


Here's an example of how to execute the module for using DQN to solve the Atari video game Pong in chapter 7.
```
cd <repo_path_on_your_computer>/chapter_7

pip3 install -r requirements.txt

python3 -m dqn_atari --environment_name=Pong
```

**Using PyTorch with GPUs:**
If you are utilizing Nvidia GPUs, it is highly recommended to install PyTorch with CUDA by following the instructions provided at https://pytorch.org/get-started/locally/.

It is crucial to acknowledge that training a DQN agent on Atari games can often be a time-consuming process, requiring hours or even days, particularly when utilizing GPUs. The duration of training depends on various factors, such as the complexity of the problem and the specifications of the computer hardware being used.



### Execute the code on Jupyter notebooks or Google Colab
Here's an example of how to execute the module on Jupyter notebooks or Google Colab. Please ensure that you have uploaded all the necessary module files to the server before proceeding. And you may need to restart the runtime after installing the required packages.
```
!pip3 install -r requirements.txt

!python3 -m dqn_atari --environment_name=Pong
```


**Important Note:**
To avoid potential conflicts with Python modules like absl.flags, it's highly recommended that you do not executing multiple modules in the same session when using Jupyter notebooks or Colab.


## Reference Code
* [DQN Zoo](https://github.com/deepmind/dqn_zoo)
* [Deep RL Zoo](https://github.com/michaelnny/deep_rl_zoo)
* [Baselines](https://github.com/openai/baselines)