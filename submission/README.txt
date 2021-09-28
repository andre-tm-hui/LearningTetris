USAGE INSTRUCTIONS:
Ensure that the following packages are installed:
	numpy, pytorch, nes-py, opencv-python, matplotlib
You may use the 'requirements.txt' file to install the above packages.

Ensure that the directory is as follows:
	- evaluate
		- best_dqn.pth
		- best_ga.npy
	- data.py
	- demo.py
	- dqn.py
	- evaluate.py
	- heuristics.py
	- play_game.py
	- tetris_env.py
	- train_dqn.py
	- train_genetic.py
	- utils.py
	- tetris.nes
	- README.txt
	- requirements.txt

Training:
To train a Genetic Algorithm agent, run the following:
	python train_genetic.py

To train a Deep Q-Learning agent, run the following:
	python train_dqn.py

Add '-h' to the end of the commands to see optional arguments.

Running a Demonstration:
To run a demonstration, use the following command:
	python demo.py -m *Model path and name*
More arguments can be seen with more detail by running:
	python demo.py -h
A selection of pre-trained models are provided in the 'evaluate' directory.


Tested on Windows 10.


