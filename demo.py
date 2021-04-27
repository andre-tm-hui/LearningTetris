import evaluate
import argparse
from data import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', help='Model name: name and location of the model to be loaded', required=True, type=str)
parser.add_argument('-feats', help='Define the features used in the model if using a Deep Q-Learning model (all features enabled by default)', default=[0,1,2,3,4,5,6,7], type=int)
parser.add_argument('-fps', help='Set the render FPS (-1 to turn off rendering)', default=600, type=int)
parser.add_argument('-s', help='Save the game to the given filename (do not include extensions)', default=None)


if __name__ == '__main__':
	args = parser.parse_args()
	if args.m[-3:] == 'npy':
		mode = GENETIC
	elif args.m[-3:] == 'pth':
		mode = MIX_DQN
	evaluate.play(args.m, mode, features=args.feats, render=args.fps, save=args.s)