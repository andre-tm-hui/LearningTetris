from play_game import *
import random
from multiprocessing import Process, Value
import numpy as np
import argparse
import os
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-m', help='Model name', required=True, type=str)
parser.add_argument('-players', help='Number of players per generation', default=100, type=int)
parser.add_argument('-initmult', help='Population multiplier for the 0th generation', default=1.5, type=float)
parser.add_argument('-ngames', help='Number of games played per player', default=5, type = int)
parser.add_argument('-mutp', help='Probability of mutations', default = 0.03, type = float)
parser.add_argument('-gens', help='Number of generations', default = 10, type = int)
parser.add_argument('-crs', help='Crossover percentage', default = 0.2, tyipe = float)


if __name__ == '__main__':
	args = parser.parse_args()

	players = args.players
	player_data = np.array([[random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(), 0]])
	generation = 0
	initial_population_multiplier = args.initmult
	generation_crossover_p = args.crs
	mutation_p = args.mutp
	games_played = args.ngames
	generations = args.gens

	pathlib.Path('models/ga/%s' % args.m).mkdir(parents=True, exist_ok=True)
	

	try:
		player_data = np.load('models/ga/%s/players.npy' % args.m)
		print('Highest Score:', player_data[0][-1])
		generation = np.load('models/ga/%s/generation.npy' % args.m)[0]
	except:
		print('No previous data found.')
		for _ in range(1, int(players*initial_population_multiplier)):
			player = np.array([random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(), 0])
			player_data = np.append(player_data, [player], axis = 0)

	while generation < generations:
		seeds = []
		for _ in range(games_played):
			seeds.append((random.randint(0,255), random.randint(0,255)))

		for p in range(len(player_data)):
			scores = Value('i', 0)
			games = []
			for i in range(games_played):
				#def __init__(self, mode, seed = False, start_level = 18, weights = False, render = False)
				proc = Process(target=play_game, args=(seeds[i], player_data[p][:-1], scores, False))
				games.append(proc)
				proc.start()
			for g in games:
				g.join()
			player_data[p][-1] = scores.value
			#print(scores.value)


		player_data = sorted(player_data, key = lambda x: x[-1], reverse = True)
		next_generation = np.copy(player_data[:int(generation_crossover_p * players)])

		while len(next_generation) < players:
			p1, p2 = random.randint(0,int(generation_crossover_p * players)-1), random.randint(0,int(generation_crossover_p * players)-1)
			while p1 == p2:
				p2 = random.randint(0,int(generation_crossover_p * players)-1)
			p1, p2 = next_generation[p1], next_generation[p2]
			child = [0,0,0,0,0,0,0,0,0,0,0,0,0]
			for i in range(len(child)):
				if random.random() < mutation_p:
					child[i] = random.random()
				else:
					if random.random() < 0.5:
						child[i] = p1[i]
					else:
						child[i] = p2[i]
			child.append(0)
			next_generation = np.append(next_generation, [np.array(child)], axis = 0)

		player_data = numpy.copy(next_generation)
		print("Generation", str(generation), ":", player_data[0][-1])
		generation += 1
		np.save('models/ga/%s/players.npy' % args.m, player_data)
		np.save('models/ga/%s/generation.npy' % args.m, np.array([generation]))

	print('Highest Score:', player_data[0][-1])
	print('Best Player:', player_data[0])