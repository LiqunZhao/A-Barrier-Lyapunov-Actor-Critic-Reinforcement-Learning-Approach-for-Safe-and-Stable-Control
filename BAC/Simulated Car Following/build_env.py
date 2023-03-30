from envs.simulated_cars_env import SimulatedCarsEnv


"""
This file includes a function that simply returns one of the two supported environments. 
"""

def build_env(args):
    """Build our custom gym environment."""

    if args.env_name == 'SimulatedCars':
        return SimulatedCarsEnv(args.seed)

    else:
        raise Exception('Env {} not supported!'.format(args.env_name))
