from env.environment import make_default_env
import numpy as np

# Initialize a demo environment with a fixed random seed for reproducibility.
env = make_default_env(max_steps=5, rng_seed=42)
obs = env.reset()
print("Initial obs:", obs)

# Randomly sample an action at each step and execute one environment transition.
for t in range(5):
    action = np.random.randint(env.num_actions)
    obs, reward, done, info = env.step(action)
    print(f"t={t}, action(qid)={info['question_id']}, "
          f"correct={info['correct']}, reward={reward}, next_obs={obs}")
    if done:
        break
