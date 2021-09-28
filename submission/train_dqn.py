from tetris_env import Tetris
from collections import namedtuple
import torch
import numpy as np
import random, sys, os, time
from data import *
from dqn import DeepQNetwork as DQN
from statistics import mean
import argparse
import pathlib

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-m', help='Model name', default='model', type=str)
parser.add_argument('-epochs', help='Number of epochs to train for', default=10000, type=int)
parser.add_argument('-feats', help='The list of selected features used for training', default=[0,1,2,3,4,5,6,7], type=int)
parser.add_argument('-lr', help='Learning Rate', default=0.001, type=float)
parser.add_argument('-e_start', help='Epsilon decay start value', default=1.0, type=float)
parser.add_argument('-e_end', help='Epsilon decay end value', default=0.001, type=float)
parser.add_argument('-batch_size', help='Batch Size', default=512, type=int)
parser.add_argument('-train_all', help='Train all models for evaluation', default=False, type=bool)


Transition = namedtuple('Transition',
                        ('state', 'done', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train(epoch = 0, epochs = 20000, load_model = None, model_name = 'model', mode = MIX_DQN, feature_select = None, lr = 0.001, e_start = 0.999, e_end = 0.001, batch_size = 512):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	for p in ['graphs/data', 'graphs/plots', 'scores', 'checkpoints', 'models/dqn/datasets']:
		pathlib.Path(p).mkdir(parents=True, exist_ok=True)

	if os.path.isfile('models/dqn/datasets/train_dataset_%d.npy' % epochs):
		train_dataset = np.load('models/dqn/datasets/train_dataset_%d.npy' % epochs)
	else:
		train_dataset = np.random.randint(0, 255, (int(1.5 * epochs), 2))
		np.save('models/dqn/datasets/train_dataset_%d.npy' % epochs, train_dataset)

	
	
	e_decay = int(epochs * 0.5)
	g = 0.999

	game_history = []
	graph_data = {'reward':[], 'score':[], 'lines':[]}
	high_score = 0
		
	if mode == BOARD_DQN:
		input_size = [10,21]
	else:
		input_size = [10,20]
	feature_size = len(feature_select)

	model = DQN(mode, input_size, feature_size).to(device)
	target = DQN(mode, input_size, feature_size).to(device)

	if load_model == None:
		load_model = model_name
	if os.path.isfile('models/dqn/%s.pth' % load_model):
		model.load_state_dict(torch.load('models/dqn/%s.pth' % load_model))
		graph_data = np.load('graphs/data/%s.npy' % load_model)
		graph_data = {'reward':graph_data[0], 'score':graph_data[1], 'lines':graph_data[2]}
		with open('checkpoints/%s.txt' % load_model) as f:
			epoch = int(f.readline())
		with open('scores/%s.txt' % load_model, 'r') as f:
			high_score = int(f.readline())
		info = {'number_of_lines':230}
	target.load_state_dict(model.state_dict())
	target.eval()

	memory = ReplayMemory(10000)
	
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
	criterion = torch.nn.MSELoss()

	game_index = 0

	while epoch < epochs:
		if feature_select != None:
			env = Tetris(mode, tuple(train_dataset[epoch]), start_level = 18, render = False, feature_select = feature_select)
		else:
			env = Tetris(mode, tuple(train_dataset[epoch]), start_level = 18, render = False)

		state = env._get_state()
		if mode == MIX_DQN:
			state = (torch.tensor(state[0], device=device, dtype=torch.float), torch.tensor(state[1], device=device, dtype=torch.float))
		else:
			state = torch.tensor(state, device=device, dtype=torch.float)
		states = env._get_states()
		done = False
		score = 0
		t = time.time()

		while not done:
			next_actions, next_states = zip(*states.items())

			e = e_end + (max(e_decay - epoch, 0) * (e_start - e_end) / e_decay)
			random_action = random.random() <= e

			if mode == MIX_DQN:
				board_states = torch.tensor([obj[0] for obj in next_states], device=device, dtype=torch.float)
				board_states = board_states.view(board_states.shape[0], 1, board_states.shape[1], board_states.shape[2])
				next_states = (board_states, torch.tensor([obj[1] for obj in next_states], device=device, dtype=torch.float))
			else:
				next_states = torch.tensor(next_states, device=device, dtype=torch.float)
			
				if mode == BOARD_DQN:
					next_states = next_states.view(next_states.shape[0], 1, next_states.shape[1], next_states.shape[2])
				next_states = [next_states]
			model.eval()
			with torch.no_grad():
				predictions = model(*next_states)
			model.train()

			if random_action:
				size = len(next_states[0]) if mode == MIX_DQN else len(next_states)
				index = random.randint(0, size-1)
			else:
				index = torch.argmax(predictions).item()

			if mode == MIX_DQN:
				next_state = (next_states[0][index, :], next_states[1][index, :])
			else:
				next_state = next_states[0][index, :]
			action = next_actions[index]

			states, reward, done, info = env.step(action)
			score += reward

			memory.push(state, done, next_state, torch.tensor([[reward]], device=device, dtype=torch.float))

			if done and epoch > 0:
				t = time.time() - t
				if info['score'] > high_score:
					high_score = info['score']

				sys.stdout.write('\rGame {:4.0f} | Reward Score: {:5.0f} | Game Score: {:6.0f} | Lines Cleared: {:3.0f} | Highest Score: {:6.0f} | Playtime: {:2.0f}:{:02.0f}'.format(
					epoch, 
					score, 
					info['score'], 
					info['number_of_lines'], 
					high_score, 
					t // 60, t % 60 
				))

				if len(memory) > 2000:
					game_history.append([score, info['score'], info['number_of_lines']])
					if len(game_history) > 100:
						del game_history[0]

					graph_data['reward'] = np.append(graph_data['reward'], [mean([g[0] for g in game_history])])
					graph_data['score'] = np.append(graph_data['score'], [mean([g[1] for g in game_history])])
					graph_data['lines'] = np.append(graph_data['lines'], [mean([g[2] for g in game_history])])

					plt.close()
					
					for i, (k, graph) in enumerate(graph_data.items()):
						plt.clf()
						plt.plot(graph)
						plt.xlabel('epoch')
						plt.ylabel('mean %s over the last 100 games' % k)
						plt.title('%s vs. Games Played' % k)
						plt.savefig('graphs/plots/%s_%s.png' % (model_name, k))
					np.save('graphs/data/%s.npy' % model_name, np.array([graph_data['reward'], graph_data['score'], graph_data['lines']]))

				break
			else:
				state = next_state
		env.close()


		if len(memory) > batch_size and len(memory) > 2000:
			epoch += 1
			transitions = memory.sample(batch_size)
			batch = Transition(*zip(*transitions))

			if mode == FEATURE_DQN:
				state_batch = [torch.cat(batch.state).view(-1,feature_size)]
			elif mode == MIX_DQN:
				board_states = torch.cat([obj[0] for obj in batch.state]).view(-1,1,20,10)
				feature_states = torch.cat([obj[1] for obj in batch.state]).view(-1,feature_size)
				state_batch = (board_states, feature_states)
			else:
				state_batch = [torch.cat(batch.state).view(-1,1,21,10)]
			reward_batch = torch.cat(batch.reward)
			done_batch = list(batch.done)
			if mode == FEATURE_DQN:
				next_state_batch = [torch.cat(batch.next_state).view(-1,feature_size)]
			elif mode == MIX_DQN:
				board_states = torch.cat([obj[0] for obj in batch.next_state]).view(-1,1,20,10)
				feature_states = torch.cat([obj[1] for obj in batch.next_state]).view(-1,feature_size)
				next_state_batch = (board_states, feature_states)				
			else:
				next_state_batch = [torch.cat(batch.next_state).view(-1,1,21,10)]

			q_values = model(*state_batch)
			next_prediction_batch = target(*next_state_batch)

			g_batch = torch.cat(tuple(reward if done else reward + g * prediction for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

			optimizer.zero_grad()
			loss = criterion(q_values, g_batch)
			loss.backward()
			for param in model.parameters():
				param.grad.data.clamp_(-1,1)
			optimizer.step()

			torch.save(model.state_dict(), 'models/dqn/%s.pth' % model_name)
			with open('checkpoints/%s.txt' % model_name, 'w') as f:
				f.write(str(epoch))
			if epoch % 15 == 0:
				target.load_state_dict(model.state_dict())
			if epoch == epochs // 2:
				torch.save(model.state_dict(), 'models/dqn/%s_%d.pth' % (model_name, epochs // 2))
				np.save('graphs/data/%s_%d.npy' % (model_name, epochs // 2), np.array([graph_data['reward'], graph_data['score'], graph_data['lines']]))
				with open('scores/%s_%d.txt' % (model_name, epochs // 2), 'w') as f:
					f.write(str(high_score))
				with open('checkpoints/%s_%d.txt' % (model_name, epochs // 2), 'w') as f:
					f.write(str(epoch))
		else:
			info = {'number_of_lines':230}

		with open('scores/%s.txt' % model_name, 'w') as f:
			f.write(str(high_score))
	return high_score


if __name__ == '__main__':
	args = parser.parse_args()
	
	if not args.train_all:
		train(0, args.epochs, None, args.m, MIX_DQN, args.feats, args.lr, args.e_start, args.e_end, args.batch_size)
	else:
		train(0, 30000, None, 'best', MIX_DQN, [0,1,2,3,4,5,6,7], 0.002)

		sum_full, sum_min, sum_board = 0, 0, 0
		for i in range(5):
			sum_full += train(0, 5000, None, 'full_%d' % i, MIX_DQN, [0,1,2,3,4,5,6,7], 0.001)
			sum_min += train(0, 5000, None, 'min_%d' % i, MIX_DQN, [0], 0.001)

		print(sum_full, sum_min, sum_board)

		print('\nAverage Scores for:',
			'\nMix (All Features): %f' % (sum_full / 5),
			'\nMix (Next Only): %f' % (sum_min / 5),
			'\nBoard: %f' % (sum_board /5)
		)

		lrs = [0.01, 0.001, 0.0001]
		lr_scores = []
		for i, lr in enumerate(lrs):
			sum_scores = 0
			for j in range(4):
				if lr != 0.001:
					sum_scores += train(0, 5000, None, 'next_%d_%f' % (j,lr), MIX_DQN, [0], lr)
				else:
					sum_scores += train(0, 5000, None, 'min_%d' % j, MIX_DQN, [0], lr)

			lr_scores += [sum_scores]

		lr = lrs[lr_scores.index(max(lr_scores))]

		print('\nAverage Scores with changing learning rates:',
			'\n0.01: %f' % (lr_scores[0] / 3),
			'\n0.001: %f' % (lr_scores[1] / 3),
			'\n0.0001: %f' % (lr_scores[2] / 3))

		configs = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]]
		heur_scores = []
		for i, c in enumerate(configs):
			sum_scores = 0
			for j in range(4):
				sum_scores += train(0, 5000, None, 'nextplus%d_%d' % (c[1], j), MIX_DQN, c, lr)

			heur_scores += [sum_scores]


		print('\nAverage Scores with different heuristics, added individually:')
		for c, s in zip(configs, heur_scores):
			print('\n%d: %f' % (c[1], s / 3))