# A-Barrier-Lyapunov-Actor-Critic-Reinforcement-Learning-Approach-for-Safe-and-Stable-Control

Repository containing the code for the paper ["A Barrier-Lyapunov Actor-Critic Reinforcement Learning Approach for
Safe and Stable Control"](https://arxiv.org/abs/2304.04066), and this code is developed based on the code: https://github.com/yemam3/SAC-RCBF

This repository only contains the code for the algorithms ***Barrier-Lyapunov Actor-Critic (BLAC)*** and ***Barrier ActorCritic (BAC)***, 
for other algorithms, please refer to:

***LAC***: https://github.com/hithmh/Actor-critic-with-stability-guarantee

***CPO, PPO-Lagrangian and TRPO-Lagrangian***: https://github.com/openai/safety-starter-agents


## Installation Requirement
The experiments are run with Pytorch, and wandb (https://wandb.ai/site) is used to save the data and draw the graphs. 
To run the experiments, some packages are listed below with their versions (in my conda environment).
```bash
python: 3.6.13
pytorch: 1.10.2 
numpy: 1.17.5
wandb: 0.12.11
cvxpy: 1.1.20
gym: 0.15.7
gpytorch 1.6.0
```

## Running the Experiments

The two environments are `Unicycle` and `SimulatedCars`. In `Unicycle`, a unicycle is required to arrive at the
desired location, i.e., destination, while avoiding collisions with obstacles. `SimulatedCars` involves a chain of five cars following each other on a straight road. The goal is to control the velocity of the 4th car to keep
a desired distance from the 3rd car while avoiding collisions with other cars.


### `Unicycle` Env: 
* BLAC algorithm: First navigate to the corresponding directory `A-Barrier-Lyapunov-Actor-Critic-Reinforcement-Learning-Approach-for-Safe-and-Stable-Control/BLAC/Unicycle/`, and then use the command:
`python main.py --env Unicycle --gamma_b 50 --max_episodes 50  --cuda --updates_per_step 2 --batch_size 128 --seed 20 --no_diff_qp --start_steps 1000`

* BAC algorithm: First navigate to the corresponding directory `A-Barrier-Lyapunov-Actor-Critic-Reinforcement-Learning-Approach-for-Safe-and-Stable-Control/BAC/Unicycle/`, and then use the command:
`python main.py --env Unicycle --gamma_b 50 --max_episodes 50  --cuda --updates_per_step 2 --batch_size 128 --seed 20 --no_diff_qp --start_steps 1000`

### `SimulatedCars` Env: 
* BLAC algorithm: First navigate to the corresponding directory `A-Barrier-Lyapunov-Actor-Critic-Reinforcement-Learning-Approach-for-Safe-and-Stable-Control/BLAC/Simulated Car Following/`, and then use the command:
`python main.py --env SimulatedCars --gamma_b 50 --max_episodes 50 --cuda --updates_per_step 2 --batch_size 128 --seed 20 --no_diff_qp --start_steps 200`

* BAC algorithm: First navigate to the corresponding directory `A-Barrier-Lyapunov-Actor-Critic-Reinforcement-Learning-Approach-for-Safe-and-Stable-Control/BAC/Simulated Car Following/`, and then use the command:
`python main.py --env SimulatedCars --gamma_b 50 --max_episodes 50 --cuda --updates_per_step 2 --batch_size 128 --seed 20 --no_diff_qp --start_steps 200`

For the meanings of the parameters, please refer to the `main.py` file.
## Others 
If you have some questions regarding the code or the paper, please do not hesitate to contact me by email. My email address is `liqun.zhao@eng.ox.ac.uk`.
