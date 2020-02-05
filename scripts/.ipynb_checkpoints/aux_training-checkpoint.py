

def on_episode_start(info):
    print(info.keys())  # -> "env", 'episode"
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))
    episode.user_data["pole_angles"] = []

def on_episode_step(info):
    episode = info["episode"]
    pole_angle = abs(episode.last_observation_for()[2])
    episode.user_data["pole_angles"].append(pole_angle)

def on_episode_end(info):
    episode = info["episode"]
    pole_angle = np.mean(episode.user_data["pole_angles"])
    print("episode {} ended with length {} and pole angles {}".format(
        episode.episode_id, episode.length, pole_angle))
    episode.custom_metrics["pole_angle"] = pole_angle

def on_train_result(info):
    print("trainer.train() result: {} -> {} episodes".format(
        info["trainer"].__name__, info["result"]["episodes_this_iter"]))
