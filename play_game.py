from tetris_env import Tetris
from utils import *
from data import *
import numpy as np
import time
#[10,9,1,1,3,3,2,2,1,4]
#[0.745, 0.523, 0.183, 0.252, 0.200, 0.200, 0.114, 0.204, 0.035, 0.438, 0.377, 0.061, 1]
def play_game(seed = False, weights = [0.745, 0.523, 0.183, 0.252, 0.200, 0.200, 0.114, 0.204, 0.035, 0.438, 0.377, 0.061, 1], shared_score = None, render = False):
	env = Tetris(GENETIC, seed, start_level = 18, weights = weights, render = render)
	next_states = env._get_states()
	done = False
	score = 0

	while not done:
		best_score = None
		best_action = None

		placements, next_placements = next_states

		for _, (action, placement) in enumerate(placements.items()):
			for _, (_, next_placement) in enumerate(next_placements[action].items()):
				state_score = next_placement['board_score'] + placement['board_score']
				if best_score == None or state_score > best_score:
					best_score = state_score
					best_action = action

		next_states, reward, done, info = env.step(best_action)
		score = info['score']
		
	env.close()

	if shared_score:
		shared_score.value += int(score)
	return score

#play_game(render = True)