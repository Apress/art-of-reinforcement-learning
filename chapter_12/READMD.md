# Chapter 12 - Distributed Reinforcement Learning

This folder includes the following module:
* `ppo.py` implements the code for the distributed PPO algorithm
* `dist_ppo_continuous.py` a driver program which uses the distributed PPO algorithm to solve classic robotic control tasks
* `gym_env_processor.py` contains the functions for environment pre-processing like observation normalization and reward normalization for the robotic control tasks
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


Here's an example of how to execute the module for using distributed PPO algorithm to solve the Ant robotic control tasks in chapter 12.
```
cd <repo_path_on_your_computer>/chapter_12

pip3 install -r requirements.txt

python3 -m dist_ppo_continuous --environment_name=Ant-v4
```

**Using PyTorch with GPUs:**
If you are utilizing Nvidia GPUs, it is highly recommended to install PyTorch with CUDA by following the instructions provided at https://pytorch.org/get-started/locally/.


### Execute the code on Jupyter notebooks or Google Colab
Here's an example of how to execute the module on Jupyter notebooks or Google Colab. Please ensure that you have uploaded all the necessary module files to the server before proceeding. And you may need to restart the runtime after installing the required packages.
```
!pip3 install -r requirements.txt

!python3 -m dist_ppo_continuous --environment_name=Ant-v4
```

**Important Note:**
* To avoid potential conflicts with Python modules like absl.flags, it's highly recommended that you do not executing multiple modules in the same session when using Jupyter notebooks or Colab.
* Running distributed RL on Colab is not recommended due to its resource usage limitations, particularly in terms of CPU allocation.


## Reference Code
* [Deep RL Zoo](https://github.com/michaelnny/deep_rl_zoo)
* [Baselines](https://github.com/openai/baselines)