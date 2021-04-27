"""An OpenAI Gym interface to the NES game Tetris"""
from nes_py import NESEnv
from heuristics import *
import random
import numpy as np
from utils import *
from data import *
import time
import cv2

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

class Tetris(NESEnv):
    # mode can be: GENETIC, BOARD_DQN or FEATURE_DQN
    def __init__(self, mode, seed = False, start_level = 18, weights = [0], render = False, feature_select = [0,1,2,3,4,5,6,7], save = False):
        super(Tetris, self).__init__('tetris.nes')
        self._start_level = start_level
        self._current_score = 0
        self._current_lines = 0
        self._mode = mode
        self._render = render
        self._save = save
        self._placements = dict()
        self._weights = weights
        self._last_piece = None
        self._last_next_piece = None
        self._feature_select = feature_select
        if self._mode == GENETIC:
            self._next_placements = {}

        if self._save != False:
            self._cap = cv2.VideoCapture(0)
            self._fourcc = cv2.VideoWriter_fourcc(*'XVID')
            if type(self._save) == str:
                self._out = cv2.VideoWriter('%s.avi' % self._save, self._fourcc, 60, (256,240))
            else:
                self._out = cv2.VideoWriter('output.avi', self._fourcc, 60, (256,240))

        self.reset()
        self._set_seed(seed)

    def _next_frame(self, action):
        if self._save != False:
            self._out.write(np.array(self.screen)[:,:,::-1])
        self._frame_advance(action)
        if self._is_game_over and self._save:
             self._out.write(np.array(self.screen)[:,:,::-1])
             self._out.release()

    def _set_seed(self, seed):
        # set the seed
        if not seed:
            seed = random.randint(0,255), random.randint(0,255)
        while self.ram[0x00C0] in [0,1,2]:
            self.ram[0x0017:0x0019] = seed
            self._next_frame(JYPD_START)
            self._next_frame(0)
        # wait until level-select menu is loaded
        for _ in range(5):
            self._next_frame(0)
        # move to appropriate starting level
        for _ in range(self._start_level % 10):
            self._next_frame(JYPD_RIGHT)
            self._next_frame(0)
        # if the starting level is greater than 10, press 'A+Start', otherwise, just press 'Start'
        if self._start_level >= 10:
            self._next_frame(JYPD_A)
            self._next_frame(JYPD_A + JYPD_START)
        else:
            self._next_frame(JYPD_START)
        # skip one frame to exit level-select screen
        self._next_frame(0)
        # skip frames until the game is playable
        while self.ram[0x0048] != 1:
            self._next_frame(0)
        # skip frames until the piece position is at the spawning point [5,0]
        while self._piece_position != [5,0]:
            self._next_frame(0)

    def _read_bcd(self, address, length, little_endian=True):
        """
        Read a range of bytes where each nibble is a 10's place figure.
        Args:
            address: the address to read from as a 16 bit integer
            length: the number of sequential bytes to read
            little_endian: whether the bytes are in little endian order
        Returns:
            the integer value of the BCD representation
        """
        if little_endian:
            iterator = range(address, address + length)
        else:
            iterator = reversed(range(address, address + length))
        # iterate over the addresses to accumulate
        value = 0
        for idx, ram_idx in enumerate(iterator):
            value += 10**(2 * idx + 1) * (self.ram[ram_idx] >> 4)
            value += 10**(2 * idx) * (0x0F & self.ram[ram_idx])

        return value

    @property
    def _current_piece(self):
        try:
            return _PIECE_ORIENTATION_TABLE[self.ram[0x0042]]
        except IndexError:
            return None
    @property
    def _next_piece(self):
        """Return the current piece."""
        try:
            return _PIECE_ORIENTATION_TABLE[self.ram[0x00BF]]
        except IndexError:
            return None

    @property
    def _number_of_lines(self):
        return self._read_bcd(0x0050, 2)

    @property
    def _lines_being_cleared(self):
        return self.ram[0x0056]
    
    @property
    def _score(self):
        return self._read_bcd(0x0053, 3)
    
    @property
    def _is_game_over(self):
        """Return True if the game is over, False otherwise."""
        if self._level == 29:
            return True
        return bool(self.ram[0x0058])

    @property
    def _board(self):
        """Return the Tetris board from NES RAM."""
        board = self.ram[0x0400:0x04C8].reshape((20, 10)).copy()
        for x in range(10):
            for y in range(20):
                if board[y][x] == 239:
                    board[y][x] = 0
                else:
                    board[y][x] = 1
        return board

    @property
    def _game_phase(self):
        return self.ram[0x0048]    

    @property
    def _piece_position(self):
        return [self.ram[0x0060], self.ram[0x0061]]

    @property
    def _level(self):
        return self.ram[0x0064]

    @property
    def _column_heights(self):
        column_heights = [20,20,20,20,20,20,20,20,20,20]
        for x in range(10):
            for y in range(19,-1,-1):
                if self._board[y][x] == 1:
                    column_heights[x] = y
        return column_heights


    def _get_reward(self):
        """Return the reward after a step occurs."""
        # reward the change in score
        reward = line_clear_scores[self._number_of_lines - self._current_lines]
        # add weight to moves played later in the game
        #reward = reward * (1 + 0.05 * self._number_of_lines)
        # update the locals
        self._current_score += reward
        self._current_lines = self._number_of_lines

        return reward

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return self._is_game_over

    def _get_info(self):
        """Return the info after a step occurs."""
        return dict(
            current_piece=self._current_piece,
            number_of_lines=self._number_of_lines,
            score=self._score,
            next_piece=self._next_piece,
            board=self._board,
            game_phase=self._game_phase,
            piece_position=self._piece_position,
            level=self._level,
        )

    def _get_state(self):
        if self._mode == FEATURE_DQN or self._mode == MIX_DQN:
            # create an array of values corresponding to different features
            board = self._board
            holes, overhangs, hole_depth = score_holes(board, get_column_heights(board))
            jagged, slope = score_bumps(get_column_heights(board))
            ret_state = [piece_data[self._next_piece]['id'], holes, overhangs, hole_depth, jagged, slope, wells(get_column_heights(board)), parity(board)]
        else:
            # append the next piece to the board before returning it
            ret_state = [np.append(self._board,[[piece_data[self._current_piece]['id'] for i in range(10)]], axis=0)]
        if self._mode == FEATURE_DQN or self._mode == BOARD_DQN:
            return ret_state
        else:
            return [self._board], [ret_state[i] for i in self._feature_select]

    def _get_states(self):
        while self._current_piece == None and not self._is_game_over:
            self._next_frame(0)
        if self._is_game_over:
            return {}
        # generate all possible placements of the current piece
        placements = generate_placements(self._current_piece, self._board, level_speeds[self._level], self._weights)
        # (re)set all dictionaries
        states = dict()
        self._placements = dict()
        if self._mode == GENETIC:
            self._next_placements = dict()
        # convert each placement into an action and their resulting state
        for p in placements:
            action = (p['placement_pos'][0],p['placement_pos'][1],p['placed_piece']['orientation'])
            self._placements[action] = p

            if self._mode == FEATURE_DQN or self._mode == MIX_DQN:
                board = p['board']
                holes, overhangs, hole_depth = score_holes(board, get_column_heights(board))
                jagged, slope = score_bumps(get_column_heights(board))
                if self._mode == MIX_DQN:
                    features = [piece_data[self._next_piece]['id'], holes, overhangs, hole_depth, jagged, slope, wells(get_column_heights(board)), parity(board)]
                    states[action] = (p['board'], [features[i] for i in self._feature_select])
                else:
                    states[action] = [piece_data[self._next_piece]['id'], holes, overhangs, hole_depth, jagged, slope, wells(get_column_heights(board)), parity(board)]

            elif self._mode == BOARD_DQN:
                states[action] = np.append(p['board'],[[piece_data[self._next_piece]['id'] for i in range(10)]], axis=0)
            
            elif self._mode == GENETIC:
                # generate the next placements in the case of the Genetic Algorithm
                next_placements = generate_placements(self._next_piece, p['board'], level_speeds[self._level], self._weights)
                temp = {}
                for next_p in next_placements:
                    temp[(next_p['placement_pos'][0],next_p['placement_pos'][1],next_p['placed_piece']['orientation'])] = next_p
                self._next_placements[action] = temp

        if self._mode == GENETIC:
            return self._placements, self._next_placements
        else:
            return states

    def step(self, action):
        placement = self._placements[action]
        # get the current/next pieces
        init_piece = self._current_piece
        next_piece = self._next_piece

        while True:
            # break the loop if the game phase becomes unplayable
            if self._game_phase != 1 or self._is_game_over:
                break

            curr_pos = self._piece_position.copy()
            curr_piece = piece_data[self._current_piece]
            path_used = False

            # move the piece above where it should fall
            if curr_pos[1] < placement['path'][0][0][1]:
                action = 0
                # check if the piece needs to be rotated
                if curr_piece != placement['init_piece']:
                    if piece_data[curr_piece['rotations'][0]] == placement['init_piece']:
                        action += JYPD_B
                    else:
                        action += JYPD_A
                # determine the direction the piece needs to move
                if curr_pos[0] - placement['path'][0][0][0] > 0:
                    action += JYPD_LEFT
                elif curr_pos[0] - placement['path'][0][0][0] < 0:
                    action += JYPD_RIGHT
                if action > 0:
                    self._next_frame(action)
                    if self._render:
                        self.render()
                        time.sleep(1/self._render if self._render > 0 else 0)
                    if self._save:
                        self._cap

            # if it's already above where it should be, and there is a move (tuck/spin) to be done
            if len(placement['path']) > 1 and curr_pos[1] >= placement['path'][0][0][1]:
                subpath = [p for p in placement['path'][1:] if p[0][1] == curr_pos[1] and p[1] != 'drop']
                action = 0
                n_actions = 0
                for s in subpath:
                    action = get_action([s[1]])
                    if action > 0:
                        self._next_frame(action)
                        if self._render:
                            self.render()
                            time.sleep(1/self._render if self._render > 0 else 0)
                    placement['path'].remove(s)
                    path_used = True

            # do not skip a frame if the path has been used, or if the piece has changed in height
            if not (curr_pos[1] != self._piece_position[1] or path_used):
                self._next_frame(0)
                if self._render:
                    self.render()
                    time.sleep(1/self._render if self._render > 0 else 0)

        # skip frames until the game is at a playable phase
        while self._game_phase != 1:
            self._next_frame(0)
            if self._render:
                self.render()
                time.sleep(1/self._render if self._render > 0 else 0)
            if self._is_game_over:
                break

        # DEBUGGING: check if the current board is equal to the next board, i.e. check for missdrops
        if not are_equal(self._board, placement['board']) and curr_pos[1] > 3 and False:
            print('missdrop', placement['placement_pos'], placement['init_piece']['rotations'])
            print(self._board)
            print(placement['board'])
            #self.render()
            #time.sleep(100000)

        # skip frames until the piece position is at the spawning point [5,0]
        while self._piece_position != [5,0]:
            self._next_frame(0)
            if self._render:
                self.render()
                time.sleep(1/self._render if self._render > 0 else 0)
            if self._is_game_over:
                break

        self._last_piece = init_piece
        self._last_next_piece = next_piece

        return self._get_states(), self._get_reward(), self._get_done(), self._get_info()

# explicitly define the outward facing API for the module
__all__ = [Tetris.__name__]