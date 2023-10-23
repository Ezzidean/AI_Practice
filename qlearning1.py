import gym
import numpy as np

env = gym.make('MountainCar-v0', render_mode="human")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 2000

"""
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
"""

DISCRETE_OS_SIZE = [20, 20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(discrete_os_win_size)

q_table = np.random.uniform(Low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
"""
print(q_table.shape)
print((q_table))
"""
def get_discrete_state(state):
    discrete_state = (state -env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for epiode in range(EPISODES):
    #checkes render state ever 2k episodes
    if epiode % SHOW_EVERY ==0:
        render = True
    else:
        render = False
    discrete_state = get_discrete_state((env.reset()))
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward,state, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if not done:
            max_future_q =  np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            #formula to calc all Q values
            #values begin changes as agent completes route once
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward+ DISCOUNT * max_future_q)
            #new q based on result of current_q
            q_table[discrete_state+(action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"Completed on ep: {epiode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    #print(new_state, new_state)
    if render:
        gym.env.render()

env.close()