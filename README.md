# Hindsight Experience Replay
Implementation of the bit flip experiment in the [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf) paper using Double DQN with a Dueling architecture in PyTorch.

### Introduction to HER
Hindsight Experience Replay (HER) is a technique for dealing with sparse rewards in environments with clearly defined goal states. In these environments, it is easy to tell if the goal is reached, but might be hard to get there. The idea is to use an off-policy algorithm that uses a replay memory and fill it with two kinds of experiences acquired during training. First of all experiences conditioned on the original goal states, but also experiences conditioned on "hindsight goals" that have actually been reached (e.g. the final state that was reached during an episode). Introducing these artificially reached goals means that both positive and negative feedback will be provided so that learning can occur.


### Results
Below are the success rates during training for 50 bits (there are more than 10^15 different states/goals in this case :scream:). Exploration starts at 20% and is decayed linearly to 0% during half of the training epochs. Each epoch consist of 50 cycles, where in each cycle 16 episodes are used to fill the replay memory followed by 40 update steps of the DQN. The success rates are taken as an avarage success over each epoch (i.e. avarged over 50 cycles * 16 episodes runs).

![50 bits](/50_bits.png)


### Todo
- [x] Run experiment with 50 bits
- [ ] Implement DDPG to use for continous actions
- [ ] Implement custom environment to test DDPG 
