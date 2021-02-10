from data import *
from heuristics import score
import math
import numpy
import random

def get_column_heights(board):
	heights = [20,20,20,20,20,20,20,20,20,20]
	for x in range(10):
		for y in range(19, -1, -1):
			if board[y][x] == 1:
				heights[x] = y
	return heights

def valid_position(pos, rel, board):
	for r in rel:
		x = int(pos[0] + r[0])
		y = int(pos[1] - r[1])
		if y < 0:
			y = 0
		if x < 0 or x > 9 or y > 19 or board[y][x] == 1:
			return False
	return True

def placeable(pos, rel, board):
	if valid_position(pos, rel, board) and not valid_position([pos[0], pos[1]+1], rel, board):
		return True
	return False

def reachable(pos, piece, board, speed):
	if not valid_position([5,0], piece['rel'], board) or not valid_position([5,0], piece_data[piece['default']]['rel'], board):
		return False
	delta_x = pos[0] - 5
	delta_y = pos[1]
	if delta_x != 0:
		direction = delta_x / abs(delta_x)
	else:
		return True
	curr_x = 5
	curr_y = 0
	max_frames = speed * delta_y

	curr_x = curr_x + direction
	lock_delay = 0
	frame = 1
	while frame < max_frames:
		frame += 1
		new_x = int(curr_x + direction * math.floor((frame-1) / tap_speed))
		new_y = int(curr_y + math.floor(frame / speed))

		if not valid_position([new_x, new_y], piece['rel'], board):
			return False
		if placeable([new_x, new_y], piece['rel'], board):
			lock_delay += 1
			if lock_delay > speed:
				return False
		else:
			lock_delay = 0
		if new_x == pos[0] and new_y <= pos[1]:
			return True
	return False


def find_path(pos, piece, board, board_highest, speed, spin = False, tuck = False):
	if pos[1] <= 0:
		return False
	breakout_height = min(list(dict.fromkeys([board_highest[el[0] + pos[0]] for i, el in enumerate(piece['rel'])])))
	if pos[1] < breakout_height:
		if reachable(pos, piece, board, speed):
			return [[pos, True]]
		else:
			return False

	if valid_position([pos[0], pos[1]-1], piece['rel'], board):
		path = find_path([pos[0], pos[1]-1], piece, board, board_highest, speed, spin, tuck)
		if path:
			path.append([[pos[0], pos[1]-1], 'drop'])
			return path

	if valid_position([pos[0]-1, pos[1]], piece['rel'], board) and not tuck:
		if speed <= 2:
			path = find_path([pos[0]-1, pos[1]], piece, board, board_highest, speed, True, True)
		else:
			path = find_path([pos[0]-1, pos[1]], piece, board, board_highest, speed, spin, True)
		if path:
			path.append([[pos[0]-1, pos[1]], 'right'])
			return path

	if valid_position([pos[0]+1, pos[1]], piece['rel'], board) and not tuck:
		if speed <= 2:
			path = find_path([pos[0]+1, pos[1]], piece, board, board_highest, speed, True, True)
		else:
			path = find_path([pos[0]+1, pos[1]], piece, board, board_highest, speed, spin, True)
		if path:
			path.append([[pos[0]+1, pos[1]], 'left'])
			return path

	if valid_position(pos, piece_data[piece['rotations'][0]]['rel'], board) and not spin:
		if speed <= 2:
			path = find_path(pos, piece_data[piece['rotations'][0]], board, board_highest, speed, True, True)
		else:
			path = find_path(pos, piece_data[piece['rotations'][0]], board, board_highest, speed, True, tuck)

		if path:
			path.append([pos, 'A'])
			return path

	if valid_position(pos, piece_data[piece['rotations'][-1]]['rel'], board) and not spin:
		if speed <= 2:
			path = find_path(pos, piece_data[piece['rotations'][-1]], board, board_highest, speed, True, True)
		else:
			path = find_path(pos, piece_data[piece['rotations'][-1]], board, board_highest, speed, True, tuck)
		if path:
			path.append([pos, 'B'])
			return path

	return False

def place_piece(board, piece, pos):
	for r in piece['rel']:
		board[pos[1]-r[1]][pos[0]+r[0]] = 1
	to_del = []
	for y in range(20):
		if 0 not in board[y]:
			to_del.append(y)
	board = numpy.delete(board, to_del, axis=0)
	for key in to_del:
		board = numpy.insert(board, 0, [0,0,0,0,0,0,0,0,0,0], axis = 0)
	return board, len(to_del)


def generate_placements(piece_name, board, speed, weights = [0.745, 0.523, 0.183, 0.252, 0.200, 0.200, 0.114, 0.204, 0.035, 0.438, 0.377, 0.061, 1]):
	piece_ids = [el for el in piece_data.keys() if el[0] == piece_name[0]]
	placements = []
	board_highest = get_column_heights(board)

	for piece_id in piece_ids:
		piece = piece_data[piece_id]
		adj_offset = list(set([x[0] for x in piece['rel']]))

		for x in range(10):
			max_height = min([board_highest[a] for a in [x + offset for offset in adj_offset if x + offset >= 0 and x + offset < 10]])
			for y in range(19, max(max_height - 2 - piece['p_offset'], 0), -1):
				if placeable([x,y], piece['rel'], board):
					path = find_path([x,y], piece, board, board_highest, speed)
					if path:
						new_piece = piece
						for p in path:
							if p[1] == "A":
								new_piece = piece_data[piece['rotations'][0]]
							elif p[1] == "B":
								new_piece = piece_data[piece['rotations'][1]]

						new_board, lines_cleared = place_piece(board.copy(), piece, [x,y])
						entry = {
							'board': new_board,
							'placement_pos': [x,y],
							'path': path,
							'placed_piece': piece,
							'init_piece': new_piece,
							'board_score': score(new_board, get_column_heights(new_board), lines_cleared, y - piece['h_offset'], weights),
							'lines_cleared': lines_cleared
						}

						placements.append(entry)
	if len(placements) == 0:
		new_board, lines_cleared = place_piece(board.copy(), piece_data[piece_name], [5,0])
		placements.append({
			'board': new_board,
			'placement_pos': [5,0],
			'path': [[[5,0], True]],
			'placed_piece': piece_data[piece_name],
			'init_piece': piece_data[piece_name],
			'board_score': score(new_board, get_column_heights(new_board), lines_cleared, 0 - piece_data[piece_name]['h_offset'], weights),
			'lines_cleared': lines_cleared
		})
	return placements

def are_equal(board1, board2):
	for x in range(10):
		for y in range(20):
			if board1[y][x] != board2[y][x]:
				return False
	return True

def get_action(inputs):
	action = 0
	for i in inputs:
		if i == 'A':
			action += 1
		elif i == 'B':
			action += 2
		elif i == 'select':
			action += 4
		elif i == 'start':
			action += 8
		elif i == 'up':
			action += 16
		elif i == 'down':
			action += 32
		elif i == 'left':
			action += 64
		elif i == 'right':
			action += 128
	return action
