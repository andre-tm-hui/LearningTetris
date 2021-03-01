GENETIC = 0
BOARD_DQN = 1
FEATURE_DQN = 2
MIX_DQN = 3

JYPD_A = 1
JYPD_B = 2
JYPD_START = 8
JYPD_LEFT = 64
JYPD_RIGHT = 128

_PIECE_ORIENTATION_TABLE = [
    'Tu',
    'Tr',
    'Td',
    'Tl',
    'Ju',
    'Jr',
    'Jd',
    'Jl',
    'Zh',
    'Zv',
    'O',
    'Sh',
    'Sv',
    'Lu',
    'Lr',
    'Ld',
    'Ll',
    'Iv',
    'Ih',
]

piece_data = {
	'Tu': {
		'rel': [[-1,0], [0,0], [1,0], [0,1]],
		'rotations': ['Tl', 'Tr'],
		'p_offset': 0,		# number of cells filled by the piece below the piece's y-position
		'h_offset': 1,		# number of cells filled by the piece above the piece's y-position
		'id': 0,
		'orientation': 2,
		'default': 'Td',
	},
    'Tr': {
		'rel': [[1,0], [0,0], [0,1], [0,-1]],
		'rotations': ['Tu', 'Td'],
		'p_offset': 1,
		'h_offset': 1,
		'id': 0,
		'orientation': 3,
		'default': 'Td',
	},
    'Td': {
		'rel': [[-1,0], [0,0], [1,0], [0,-1]],
		'rotations': ['Tr', 'Tl'],
		'p_offset': 1,
		'h_offset': 0,
		'id': 0,
		'orientation': 0,
		'default': 'Td',
	},
    'Tl': {
		'rel': [[-1,0], [0,0], [0,1], [0,-1]],
		'rotations': ['Td', 'Tu'],
		'p_offset': 1,
		'h_offset': 1,
		'id': 0,
		'orientation': 1,
		'default': 'Td',
	},
    'Ju':{
		'rel': [[0,1], [0,0], [0,-1], [-1,-1]],
		'rotations': ['Jl', 'Jr'],
		'p_offset': 1,
		'h_offset': 1,
		'id': 1,
		'orientation': 1,
		'default': 'Jl',
	},
    'Jr':{
		'rel': [[-1,1], [-1,0], [0,0], [1,0]],
		'rotations': ['Ju', 'Jd'],
		'p_offset': 0,
		'h_offset': 1,
		'id': 1,
		'orientation': 2,
		'default': 'Jl',
	},
    'Jd':{
		'rel': [[0,1], [0,0], [0,-1], [1,1]],
		'rotations': ['Jr', 'Jl'],
		'p_offset': 1,
		'h_offset': 1,
		'id': 1,
		'orientation': 3,
		'default': 'Jl',
	},
    'Jl':{
		'rel': [[1,-1], [-1,0], [0,0], [1,0]],
		'rotations': ['Jd', 'Ju'],
		'p_offset': 1,
		'h_offset': 0,
		'id': 1,
		'orientation': 0,
		'default': 'Jl',
	},
    'Zh':{
		'rel': [[-1,0], [0,0], [0,-1], [1,-1]],
		'rotations': ['Zv'],
		'p_offset': 1,
		'h_offset': 0,
		'id': 2,
		'orientation': 0,
		'default': 'Zh',
	},
    'Zv':{
		'rel': [[0,-1], [0,0], [1,0], [1,1]],
		'rotations': ['Zh'],
		'p_offset': 1,
		'h_offset': 1,
		'id': 2,
		'orientation': 1,
		'default': 'Zh',
	},
    'O':{
		'rel': [[-1,-1], [-1,0], [0,-1], [0,0]],
		'rotations': ['O'],
		'p_offset': 1,
		'h_offset': 0,
		'id': 3,
		'orientation': 0,
		'default': 'O',
	},
    'Sh':{
		'rel': [[-1,-1], [0,-1], [0,0], [1,0]],
		'rotations': ['Sv'],
		'p_offset': 1,
		'h_offset': 0,
		'id': 4,
		'orientation': 0,
		'default': 'Sh',
	},
    'Sv':{
		'rel': [[0,1], [0,0], [1,0], [1,-1]],
		'rotations': ['Sh'],
		'p_offset': 1,
		'h_offset': 1,
		'id': 4,
		'orientation': 1,
		'default': 'Sh',
	},
    'Lu':{
		'rel': [[0,1], [0,0], [0,-1], [1,-1]],
		'rotations': ['Ll', 'Lr'],
		'p_offset': 1,
		'h_offset': 1,
		'id': 5,
		'orientation': 3,
		'default': 'Lr',
	},
    'Lr':{
		'rel': [[-1,-1], [-1,0], [0,0], [1,0]],
		'rotations': ['Lu', 'Ld'],
		'p_offset': 1,
		'h_offset': 0,
		'id': 5,
		'orientation': 0,
		'default': 'Lr',
	},
    'Ld':{
		'rel': [[-1,1], [0,1], [0,0], [0,-1]],
		'rotations': ['Lr', 'Ll'],
		'p_offset': 1,
		'h_offset': 1,
		'id': 5,
		'orientation': 1,
		'default': 'Lr',
	},
    'Ll':{
		'rel': [[-1,0], [0,0], [1,0], [1,1]],
		'rotations': ['Ld', 'Lu'],
		'p_offset': 0,
		'h_offset': 1,
		'id': 5,
		'orientation': 2,
		'default': 'Lr',
	},
    'Iv':{
		'rel': [[0,-1], [0,0], [0,1], [0,2]],
		'rotations': ['Ih'],
		'p_offset': 1,
		'h_offset': 2,
		'id': 6,
		'orientation': 1,
		'default': 'Ih',
	},
    'Ih':{
		'rel': [[-2,0], [-1,0], [0,0], [1,0]],
		'rotations': ['Iv'],
		'p_offset': 0,
		'h_offset': 0,
		'id': 6,
		'orientation': 0,
		'default': 'Ih',
	},
}

level_speeds = [48, 42, 38, 33, 28, 23, 18, 13, 8, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

line_clear_scores = [1, 10, 25, 75, 300]

tap_speed = 6
#list(dict.fromkeys([el[0] for i, el in enumerate(pieceData[piece][rel])]))