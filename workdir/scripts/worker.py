@ray.remote
class Worker(object):
    def __init__(self, config, policy_params, env_name, noise):
        self.env = # Initialize environment.
        self.policy = # Construct policy.
        # Details omitted.

    def do_rollouts(self, params):
        perturbation = # Generate a random perturbation to the policy.

        self.policy.set_weights(params + perturbation)
        # Do rollout with the perturbed policy.

        self.policy.set_weights(params - perturbation)
        # Do rollout with the perturbed policy.

        # Return the rewards.