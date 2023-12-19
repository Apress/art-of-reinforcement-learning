# Chapter 3 - Dynamic Programming

This folder includes the following module:
* `envs` contains the classes for MRP (Markov Reward Process) and MDP (Markov Decision Process) environments
* `algos.py` implements the Dynamic Programming (DP) algorithms to solve MRP and MDP problems
* `service_dog_mrp.py` a driver program which uses DP algorithm to solve the service dog MRP problem
* `service_dog_mdp.py` a driver program which uses DP algorithm to solve the service dog MDP problem


### Service Dog MRP
<img src="./images/dog_mrp.png" width="600" >

### Service Dog MDP
<img src="./images/dog_mdp.png" width="600" >


## Additional task
We have added supplementary modules that incorporate DP algorithms to address the student MRP and MDP problems. These particular problems were initially presented in UCL's RL course, which was instructed by Professor David Silver.

For further details on MDP, you can refer to the following resource:
https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf

* `student_mrp.py` a driver program which uses DP algorithm to solve the student MRP problem
* `student_mdp.py` a driver program which uses DP algorithm to solve the student MDP problem


### Student MRP
<img src="./images/student_mrp.png" width="600" >

### Student MDP
<img src="./images/student_mdp.png" width="600" >


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


Here's an example of how to execute the module for using DP to solve Service Dog MDP in chapter 3.
```
cd <repo_path_on_your_computer>/chapter_3

pip3 install -r requirements.txt

python3 -m dp_service_dog_mdp
```

### Execute the code on Jupyter notebooks or Google Colab
Here's an example of how to execute the module on Jupyter notebooks or Google Colab. Please ensure that you have uploaded all the necessary module files to the server before proceeding. And you may need to restart the runtime after installing the required packages.
```
!pip3 install -r requirements.txt

!python3 -m dp_service_dog_mdp
```

**Important Note:**
To avoid potential conflicts with Python modules like absl.flags, it's highly recommended that you do not executing multiple modules in the same session when using Jupyter notebooks or Colab.