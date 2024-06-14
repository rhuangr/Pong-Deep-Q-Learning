# Pong with Deep-Q Learning
## Project Description
This program teaches an agent to play pong using Deep-Q learning. The repository contains a file for the agent class and a file for the pong game. The neural network was coded using **only** numpy. The code for pong game was sourced from [Geeks for Geeks](https://www.geeksforgeeks.org/create-a-pong-game-in-python-pygame/) and was modified to provide agents with the necessary information to learn. 
![image](https://github.com/rhuangr/Pong-Deep-Q-Learning/assets/170949635/566f4e94-e3b1-4863-ae8f-1f6ea1c84dd6)
The agents folder contains two files with the agents that I have trained. If you wish to see how those perform, simply move the `leftAgent.npz` and `rightAgent.npz` in the same directory as pongGame.py and follow the installation steps below 😸. These agents were not trained for a long time, so they do occasionally miss the ball (for some reason, the left Agent also performs significantly better than its right counterpart). I am certain that if you kept training them, they would perform improve to never miss the ball. 
## Installation
1.  Clone the repository.
2.  Install the numpy library if not yet installed.
3.  Run pongGame.py.  _You can paste the following commands while in the cloned repository directory 👍  `python pongGame.py`_
## Designing appropriate reward signals
The training process to get above average results was surpisingly 
A natural approach would simply be to reward the agent for hitting the ball and punish it for missing. However, since the agent begins training by taking random moves, its chances of hitting the ball are very low. The reward received by the agent is sparse, leaving it with little feedback to converge to the optimal policy.

Therefore, to create an environment with more rewards, I created two extra subgoals for the agent to achieve, with each subgoal having its own reward system.

1. Moving the board in the direction of an *incoming* ball.
2. Following an *incoming* ball with the board.

## Features (Input Neuron Values)
Every game state is represented with 5 features. Furthermore, every value within this feature vector will be normalized to be in the range of -1 to 1.  The five features are described below:

1. The Y position of the ball relative to the game window.
2. The X position of the ball.
3. The direction in which the ball is heading.
4. The Y position of the board
5. The distance between the ball and the board.
	(Looking back, this could most likely be ommited and still achieve the same results)
## Bugs
RL agents are very effective at finding bugs in programs. Initally, after training the agents exclusively on bouncing the ball. Agents found a bug in the initial pong game, which allowed the ball to be bounced against the goal wall for a short period of time. 

https://github.com/rhuangr/Pong-Deep-Q-Learning/assets/170949635/245d0f56-b1ec-4cf4-b2e7-29255023aa7d

The bug was simple to fix: set a number of frames for which the agent cannot earn further rewards from bouncing the ball. 
