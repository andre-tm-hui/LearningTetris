from tetris_env import Tetris
from collections import namedtuple
import torch
import numpy as np
import random, sys, os, time
from data import *
from dqn import DeepQNetwork as DQN
from statistics import mean

import matplotlib.pyplot as plt

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


def train(epoch = 0, epochs = 20000, load_model = None, model_name = 'model', mode = FEATURE_DQN, n_features = 8, max_replays = 5, feature_select = None):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	for p in ['graphs', 'graphs/data', 'graphs/plots', 'scores', 'models', 'checkpoints']:
		if not os.path.isdir(p):
			os.mkdir(p)
	
	e_start = 1
	e_end = 0.001
	e_decay = int(epochs * 0.5)
	g = 0.999
	batch_size = 512

	line_start = 0
	line_end = 200
	line_growth = int(epochs * 0.25)

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
	if os.path.isfile('models/%s.pth' % load_model):
		model.load_state_dict(torch.load('models/%s.pth' % load_model))
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
	
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
	criterion = torch.nn.MSELoss()

	while epoch < epochs:
		# replay a certain seed up to 3 times, starting from epoch 10000, if the score is below a threshold
		# as a result, the threshold increases over time, starting from 20 and ending at ~190
		if epoch > e_decay and epoch < e_decay + line_growth and info['number_of_lines'] < (line_start + (max(0, (epoch - e_decay) / line_growth)) * (line_end - line_start)) and replayed < max_replays:
			replayed += 1
		else:
			replayed = 0
			seed = random.randint(0,255), random.randint(0,255)
		if feature_select != None:
			env = Tetris(mode, seed, start_level = 18, render = False, feature_select = feature_select)
		else:
			env = Tetris(mode, seed, start_level = 18, render = False)

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
				feature_states = torch.cat([obj[1] for obj in batch.state]).view(-1,8)
				state_batch = (board_states, feature_states)
			else:
				state_batch = [torch.cat(batch.state).view(-1,1,21,10)]
			reward_batch = torch.cat(batch.reward)
			done_batch = list(batch.done)
			if mode == FEATURE_DQN:
				next_state_batch = [torch.cat(batch.next_state).view(-1,feature_size)]
			elif mode == MIX_DQN:
				board_states = torch.cat([obj[0] for obj in batch.next_state]).view(-1,1,20,10)
				feature_states = torch.cat([obj[1] for obj in batch.next_state]).view(-1,8)
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

			torch.save(model.state_dict(), 'models/%s.pth' % model_name)
			with open('checkpoints/%s.txt' % model_name, 'w') as f:
				f.write(str(epoch))
			if epoch % 15 == 0:
				target.load_state_dict(model.state_dict())
			if epoch == epochs // 2:
				torch.save(model.state_dict(), 'models/%s_%d.pth' % (model_name, epochs // 2))
				np.save('graphs/data/%s_%d.npy' % (model_name, epochs // 2), np.array([graph_data['reward'], graph_data['score'], graph_data['lines']]))
				with open('scores/%s_%d.txt' % (model_name, epochs // 2), 'w') as f:
					f.write(str(high_score))
				with open('checkpoints/%s_%d.txt' % (model_name, epochs // 2), 'w') as f:
					f.write(str(epoch))
		else:
			info = {'number_of_lines':230}

		with open('scores/%s.txt' % model_name, 'w') as f:
			f.write(str(high_score))


if __name__ == '__main__':
	#train(epochs = 10000, load_model = 'feature_model_short_5000', model_name = 'feature_model_short', mode = FEATURE_DQN, max_replays = 0)
	#train(epochs = 10000, model_name = 'board_model_short', mode = BOARD_DQN, max_replays = 0)
	#train(epochs = 10000, model_name = 'feature_model_replays_short', mode = FEATURE_DQN)
	#train(epochs = 10000, load_model = 'board_model_short_5000', model_name = 'board_model_replays_short', mode = BOARD_DQN)

	#train(epochs = 5000, model_name = 'model_5000e', mode = BOARD_DQN)
	#train(epochs = 15000, model_name = 'model_15000e', mode = BOARD_DQN)
	#train(epochs = 20000, model_name = 'mix_model_20000e', mode = MIX_DQN)
	train(epochs = 20000, model_name = 'board_model_20000e', mode = MIX_DQN, feature_select = [0])
	