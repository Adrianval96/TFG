import ray
import ray.tune as tune

ray.init()
tune.run_experiments({
    "my_experiment": {
        "run": "A3C",
        "env": "Breakout-v0",
        "stop": {"time_total_s": 3600},
        "config": {
            "num_gpus": 1,
            "num_workers": 4,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        },
    },
})