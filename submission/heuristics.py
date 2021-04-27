
def score(board, board_highest, lines_cleared, placement_height, weights = [0.745, 0.523, 0.183, 0.252, 0.200, 0.200, 0.114, 0.204, 0.035, 0.438, 0.377, 0.061, 1]):
	if len(weights) == 1:
		return 0

	holes, overhangs, hole_depth = score_holes(board, board_highest)
	jagged, slope = score_bumps(board_highest)

	if lines_cleared == 4:
		line_clear_score = 10000
	elif lines_cleared > 0:
		line_clear_score = 1
	else:
		line_clear_score = 0

	heuristic_score = (
		- weights[0] * holes
		- weights[1] * overhangs
		- weights[2] * hole_depth
		- weights[3] * (14 - placement_height)
		- weights[4] * jagged
		+ weights[5] * slope
		- weights[6] * wells(board_highest)
		- weights[7] * right_well(board)
		- weights[8] * parity(board)
		- weights[9] * line_clear_score * (1 if lines_cleared == 1 else 0)
		- weights[10] * line_clear_score * (1 if lines_cleared == 2 else 0)
		- weights[11] * line_clear_score * (1 if lines_cleared == 3 else 0)
		+ weights[12] * line_clear_score * (1 if lines_cleared == 4 else 0)
		)

	return heuristic_score


def closed_hole(board, board_highest, pos, direction = 0):
	if pos[1] - 1 >= 0 and board[pos[1]-1][pos[0]] == 0:
		if pos[1] - 1 > board_highest[pos[0]]:
			return closed_hole(board, board_highest, [pos[0], pos[1]-1], 0)
		else:
			return False

	elif pos[0] + 1 <= 9 and board[pos[1]][pos[0]+1] == 0 and direction >= 0:
		if pos[1] > board_highest[pos[0]+1]:
			return closed_hole(board, board_highest, [pos[0]+1, pos[1]], 1)
		else:
			return False

	elif pos[0] - 1 >= 0 and board[pos[1]][pos[0]-1] == 0 and direction <= 0:
		if pos[1] > board_highest[pos[0]-1]:
			return closed_hole(board, board_highest, [pos[0]-1, pos[1]], -1)
		else:
			return False

	return True

def score_holes(board, board_highest):
	empty = []
	holes = []
	depth_score = 0
	overhangs = []

	for x in range(10):
		for y in range(19, board_highest[x]-1, -1):
			if board[y][x] == 0:
				empty.append([x,y])

	for cell in empty:
		depth_score += cell[1] - board_highest[cell[0]]
		if closed_hole(board, board_highest, cell):
			holes.append(cell)
		else:
			overhangs.append(cell)

	return len(holes), len(overhangs), depth_score

def score_bumps(board_highest):
	jagged_score = 0
	slope_score = 0
	for x in range(8):
		diff = board_highest[x] - board_highest[x+1]
		jagged_score += abs(diff)
		slope_score -= diff

	return jagged_score, slope_score

def right_well(board):
	right_well_score = 0
	for y in range(20):
		if board[y][9] != 0:
			right_well_score += 1
	return right_well_score

def wells(board_highest):
	well_score = 0
	for x in range(9):
		left_diff = 20
		right_diff = 20
		if x > 0:
			left_diff = board_highest[x] - board_highest[x-1]
		if x < 8:
			right_diff = board_highest[x] - board_highest[x+1]
		if left_diff > 1 and right_diff > 1:
			well_score += min(left_diff, right_diff)

	return well_score

def parity(board):
	parity_score = 0
	for x in range(10):
		for y in range(20):
			if board[y][x] != 0:
				if y % 2 == 0:
					if x % 2 == 0:
						parity_score += 1
					else:
						parity_score -= 1
				else:
					if x % 2 == 0:
						parity_score -= 1
					else:
						parity_score += 1
	return abs(parity_score)


