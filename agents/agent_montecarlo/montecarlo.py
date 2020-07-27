import numpy as np
from typing import Tuple, Optional
from agents.common import check_end_state, GameState, available_columns, apply_player_action
from agents.common import PLAYER1, PLAYER2, BoardPiece, PlayerAction, SavedState
from random import choice
import time
import copy
from math import log

determined_C = 0.5 #Global variable for exploration term

class node_montecarlo:
    def __init__(self, action:int=None, C = None, parent=None, board_state =None, player=None):
        self.action = action
        self.visits = 0  # Number of time the node has been visited
        self.reward = 0  # Accumulated reward value or the number of times for win
        self.C = C
        self.parent = parent  # Null if node is a root node
        self.children = np.array([], dtype=np.int8)  # Array of children
        self.board_state = board_state  # Current board state
        self.player = player
        self.opponent = PLAYER2 if player==PLAYER1 else PLAYER1
        self.available_actions = self.free_columns()


    def child_addition(self, action:PlayerAction) -> object:
        '''
        Function to add child node
        :param action:
        :return: Node of the child
        '''
        #Board state is the state after the player has put a piece
        im_board, pre_board = apply_player_action(self.board_state.copy(), action, self.opponent, copy=True)
        #Get a new child node
        child_node = node_montecarlo(action=action, parent=self, board_state=im_board, player=self.opponent)
        self.children = np.append(self.children, child_node)
        #Keep only the untried actions
        self.available_actions = self.available_actions[self.available_actions != action]
        return child_node

    def free_columns(self)->np.ndarray:
        '''
        Function to get the array of available actions
        :return: array of free columns (array)
        '''
        if check_end_state(self.board_state, self.opponent) != GameState.STILL_PLAYING:
            #If the end state is win or draw then the list of possible actions is empty
            return np.array([])
        else:
            return available_columns(self.board_state)


def best_child(node: node_montecarlo, c_value: float) -> node_montecarlo:
    '''
    Function to determine the most urgent node with the highest Upper Confidence bounds (UCB)
    UCB = Xj + C * sqrt(2 * ln n / nj)
        Xj -> reward of choice j (Exploitation =rewards / visits of the child j)
        C -> Exploration term
        n -> Number of times the parent has been tried (node visits)
        nj -> Number of times the choice j has been tried (child visit)
    :param node: The root node from where the children are evaluated
    :param c_value: Exploration term
    :return: best_child node, the most urgent action
    '''
    node.C = c_value
    # print(c_value)
    best_child_node = None
    child_node_UCB = -100000
    for child in node.children:
        UCB = child.reward / child.visits + node.C * np.sqrt(2 * log(node.visits) / child.visits)
        if UCB > child_node_UCB:
            best_child_node = child
            child_node_UCB = UCB
    return best_child_node

def expansion(node:node_montecarlo)->node_montecarlo:
    if len(node.available_actions) != 0:
        #Choose a random action from untried actions
        action = choice(node.available_actions)
        node = node.child_addition(action)

    return node

def selection(node:node_montecarlo)->node_montecarlo:
    while len(node.available_actions) == 0 and len(node.children) != 0:
        node = best_child(node, c_value=determined_C)
    return node

def tree_policy(node:node_montecarlo) -> node_montecarlo:
    node = selection(node)

    node = expansion(node)

    # print(f'Current player tree {node.player}')
    # print(f'Current player tree {node.board_state}')
    # print(f'Actions {node.available_actions}')
    return node


'''def tree_policy(node:node_montecarlo) -> node_montecarlo:

    Function to obtain the best node child from a node
    :param node: The root node frm where the random expansion will be evaluated before selecting the
    best node with the higher UCB value
    :return: Best child node

    current_node = node
        # if the node is not fully expanded
    print(f'Current player tree {current_node.player}')
    print(f'Current player tree {current_node.board_state}')
    actions = len(current_node.available_actions)

    print(f'Actions {actions}')
    print(f'Children {current_node.children}')
    if actions == 0 and len(current_node.children) != 0:
        current_node = best_child(current_node, determined_C)
        return current_node
    else:
        current_node = expansion(current_node)

    if actions != 0:
        current_node = expansion(current_node)
    return current_node

    return current_node
'''

def default_policy(node: node_montecarlo) -> int:
    '''
    Simulation of the game from a node
    :param node: Node to get the board state
    :return: end_game = True and the last player
    '''
    board = node.board_state.copy()
    player = node.player
    end_game = False
    # While the board is not a terminal board:
    while not end_game and len(available_columns(board)) != 0:
        # print(f'Current player default {player}')
        action = choice(available_columns(board))  # Random action to simulate
        board, _ = apply_player_action(board, action, player, copy=True)
        # Check if the board is an end_game
        if check_end_state(board, player, action) == GameState.IS_DRAW:
            return 0
        if check_end_state(board, player, action) == GameState.IS_WIN:
            end_game = True
        else:
            if player == PLAYER1:
                player = PLAYER2
            else:
                player = PLAYER1
    if player == node.player:
        return 1
    if player == node.opponent:
        return -1


def back_propagation(node: node_montecarlo, result: int):
    '''
    Function to back propagate the values of the rewards and update them
    Also update visits
    :param node: Node to start back propagating
    :param result: The result for each node is changing from 1 to -1
    '''
    while node is not None:
        node.visits += 1
        node.reward += result
        node = node.parent
        result = -result

def montecarlo_search(board:np.ndarray) -> PlayerAction:
    '''
    Function to create a node
    :param board_state:
    :return:
    '''
    #Create a root node
    root_node = node_montecarlo(board_state=board.copy(), player=active_player)
    search_time = 1
    pre_time = time.time()
    while time.time() < (search_time + pre_time):
        node = root_node
        #Selection and expansion (Tree policy)
        node = tree_policy(node)
        #Simulation (Default policy)
        result = default_policy(node)
        #Back propagation
        back_propagation(node, result)
    return best_child(root_node, 0).action

def generate_move_mcts(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
        #Tuple[PlayerAction, Optional[SavedState]]:
    global active_player
    global active_opponent

    if player == PLAYER1:
        active_player = PLAYER1
        active_opponent = PLAYER2
    else:
        active_player = PLAYER2
        active_opponent = PLAYER1

    columns = board.shape[1]
    if board[0, columns // 2] == 0:
        best_action = columns // 2
    else:
        best_action = montecarlo_search(board.copy())
    return best_action, saved_state
