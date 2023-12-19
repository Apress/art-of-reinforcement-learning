# Chapter 5 - Temporal Difference Learning

This folder includes the following module:
* `envs` contains the classes for MDP environments
* `algos.py` implements TD algorithms for solving MDP problems
* `td0_policy_evaluation_service_dog_mdp.py` a driver program utilizes the TD0 method to evaluate a policy for the service dog MDP problem
* `sarsa_service_dog_mdp.py` a driver program applies the SARSA algorithm to solve the service dog MDP problem
* `q_learning_service_dog_mdp.py` a driver program utilizes the Q-learning algorithm to solve the service dog MDP problem
* `double_q_casino_mdp.py` a module shows the comparison of double Q-learning vs. Q-learning on the casino MDP problem
* `n_step_sarsa_service_dog_mdp.py` a driver program applies the SARSA algorithm with N-step return to solve the service dog MDP problem

### Service Dog MDP
<img src="./images/dog_mdp.png" width="600" >


## Additional task
We have added supplementary modules that incorporate DP algorithms to address the student MRP and MDP problems. These particular problems were initially presented in UCL's RL course, which was instructed by Professor David Silver.

For further details on MDP, you can refer to the following resource:
https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf

Furthermore, we have included the code for the Maximization Bias MDP example, depicted in Figure 6.5 of the book "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto.

To learn more about the book, you can visit the following link: 
http://incompleteideas.net/book/the-book-2nd.html

* `sarsa_student_mdp.py` a driver program which uses SARSA algorithm to solve the student MDP problem
* `q_learning_student_mdp.py` a driver program which uses Q-learning algorithm to solve the student MDP problem
* `double_q_max_bias_mdp.py` a module shows the comparison of double Q-learning vs. Q-learning on the Maximization Bias MDP

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


Here's an example of how to execute the module for using Q-learning to solve the Service Dog MDP in chapter 5.
```
cd <repo_path_on_your_computer>/chapter_5

pip3 install -r requirements.txt

python3 -m q_learning_service_dog_mdp
```

### Execute the code on Jupyter notebooks or Google Colab
Here's an example of how to execute the module on Jupyter notebooks or Google Colab. Please ensure that you have uploaded all the necessary module files to the server before proceeding. And you may need to restart the runtime after installing the required packages.
```
!pip3 install -r requirements.txt

!python3 -m q_learning_service_dog_mdp
```

**Important Note:**
To avoid potential conflicts with Python modules like absl.flags, it's highly recommended that you do not executing multiple modules in the same session when using Jupyter notebooks or Colab.