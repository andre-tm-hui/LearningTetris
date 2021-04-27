from utils import *
from data import *
from dqn import DeepQNetwork
from tetris_env import Tetris

import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def play(fname, mode, seed = (random.randint(0,255), random.randint(0,255)), features = None, save = False, render = False):
	if mode == GENETIC:
		player_data = np.load(fname)
		weights = player_data[0][:-1]

	elif mode == MIX_DQN:
		model = DeepQNetwork(MIX_DQN, (10,20), len(features)).to(device)
		model.load_state_dict(torch.load(fname))
		model.eval()
		weights = np.zeros(15)

	env = Tetris(mode, seed, start_level = 18, weights = weights, render = render, save = save, feature_select = features)
	states = env._get_states()
	done = False
	score = 0

	with torch.no_grad():
		while not done:
			best_score = None
			best_action = None

			if mode == GENETIC:
				placements, next_placements = states
				for _, (action, placement) in enumerate(placements.items()):
					for _, (_, next_placement) in enumerate(next_placements[action].items()):
						state_score = next_placement['board_score'] + placement['board_score']
						if best_score == None or state_score > best_score:
							best_score = state_score
							best_action = action

			elif mode == MIX_DQN:
				next_actions, next_states = zip(*states.items())
				board_states = torch.tensor([obj[0] for obj in next_states], device=device, dtype=torch.float)
				board_states = board_states.view(board_states.shape[0], 1, board_states.shape[1], board_states.shape[2])
				next_states = (board_states, torch.tensor([obj[1] for obj in next_states], device=device, dtype=torch.float))

				predictions = model(*next_states)
				index = torch.argmax(predictions).item()
				next_state = (next_states[0][index, :], next_states[1][index, :])
				best_action = next_actions[index]

			states, reward, done, info = env.step(best_action)
			score = info['score']
			
		env.close()
	
	try:
		del model
		torch.cuda.empty_cache()
	except:
		pass
	return score

def score_config(fname, mode, seeds = list(range(10)), features = None, save = False, render = False):
	score = []
	for seed in seeds:
		score += [play(fname, mode, seed = (0,seed), features = features, save = save, render = render)]

	return score


if __name__ == '__main__':
	plt.plot([362424, 369540, 369992, 487876, 562752, 520696, 543740, 476056, 518952, 454976])
	plt.xlabel('Generation')
	plt.xticks(range(10))
	plt.ylabel('Average Score')
	plt.savefig('ga_graph.png')

	evaluation_dir = './evaluate/'
	fnames = os.listdir(evaluation_dir)

	dqn_fnames = []
	genetic_fnames = []
	for fname in fnames:
		if 'pth' == fname[-3:]:
			dqn_fnames += [fname]
		elif 'npy' == fname[-3:]:
			genetic_fnames += [evaluation_dir + fname]

	'''next_plus = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
	lr = {1e-2: [], 1e-3: [], 1e-4: []}
	full, next_only = [], []

	for fname in dqn_fnames:
		if 'full' in fname:
			full += [evaluation_dir + fname]
		elif 'min' in fname:
			next_only += [evaluation_dir + fname]
		elif 'nextplus' in fname:
			next_plus[int(fname[8])] += [evaluation_dir + fname]
		elif 'lr' in fname:
			lr[float(fname[5:13])] += [evaluation_dir + fname]

	plt.clf()
	plt.xlabel('Game')
	plt.xticks(range(10))
	plt.ylabel('Score')

	score = []
	for fname in full:
		score += [score_config(fname, MIX_DQN, features = [0,1,2,3,4,5,6,7])]

	print('Avg Score when All Features Used: %f' % np.mean(score))
	print('Score SD when All Features Used: %f' % np.std(score))
	plt.plot(np.mean(score, axis=0), label='All Features')

	score = []
	for fname in next_only:
		score += [score_config(fname, MIX_DQN, features = [0])]

	print('Avg Score when Only Next Piece Used: %f' % np.mean(score))
	print('Score STD when Only Next Piece Used: %f' % np.std(score))
	plt.plot(np.mean(score, axis=0), label='Next Piece Only')

	for feat, fnames in next_plus.items():
		if len(fnames) > 0:
			score = []
			for fname in fnames:
				score += [score_config(fname, MIX_DQN, features = [0, feat])]

			print('Avg Score when Next Piece and Feature %d Used: %f' % (feat, np.mean(score)))
			print('Score SD when Next Piece and Feature %d Used: %f' % (feat, np.std(score)))
		plt.plot(np.mean(score, axis=0), label='Next Piece and Feature %d' % feat)

	plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
	plt.savefig('features.png')
	plt.clf()
	plt.xlabel('Game')
	plt.xticks(range(10))
	plt.ylabel('Score')

	for lr, fnames in lr.items():
		if len(fnames) > 0:
			score = []
			for fname in fnames:
				score += [score_config(fname, MIX_DQN, features = [0])]

			print('Avg Score when LR = %f Used: %f' % (lr, np.mean(score)))
			print('Score SD when LR = %f Used: %f' % (lr, np.std(score)))
		plt.plot(np.mean(score, axis=0), label='LR = %f' % lr)

	plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
	plt.savefig('lr.png')'''
	plt.clf()
	plt.xlabel('Game')
	plt.xticks(range(10))
	plt.ylabel('Score')

	'''for fname in dqn_fnames:
		if 'best' in fname:
			score = score_config(evaluation_dir + fname, MIX_DQN, features = [0,1,2,3,4,5,6,7], save = True, render = True)
			print('Avg Score when DQN Used: %f' % np.mean(score))
			print('Score SD when DQN Used: %f' % np.std(score))
			break

		plt.plot(score)
		plt.savefig('compare.png')'''

	for fname in genetic_fnames:
		score = score_config(fname, GENETIC, features = [0], save = True, render = 6000, seeds = [0])
		print('Avg Score when GA Used: %f' % np.mean(score))
		print('Score SD when GA Used: %f' % np.std(score))
		break

	plt.plot(score)

	



