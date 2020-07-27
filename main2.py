import numpy as np
from typing import Optional, Callable
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from agents.agent_random import generate_move_ran
from agents.agent_score import generate_move_sc
from agents.agent_minimax_aa import generate_move_min
from agents.agent_minimax_abeta import generate_move_min_ab
from agents.agent_montecarlo import generate_move_mc
import pstats
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
import cProfile

'''Main 2 to run tests for the C parameter'''


def user_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]):
    action = PlayerAction(-1)
    while not 0 <= action < board.shape[1]:
        try:
            action = PlayerAction(input("Column? "))
        except:
            pass
    return action, saved_state


def human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Montecarlo",
    player_2: str = "Minimax_ab",
    draw: int = 0,
    win_1: int = 0,
    win_2: int = 0,
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):
    import time
    from agents.common import PLAYER1, PLAYER2, GameState
    from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                # print(pretty_print_board(board))
                # print(
                #    f'{player_name} you are playing with {"X" if player == PLAYER1 else "O"}'
                # )
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                # print(f"Move time: {time.time() - t0:.3f}s")
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player, action)
                if end_state != GameState.STILL_PLAYING:
                    # print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                        draw +=1
                    else:
                        print(f'{player_name} won playing {"X" if player == PLAYER1 else "O"}')
                        if player_name == 'Montecarlo':
                            win_1 += 1
                        if player_name == "Minimax_ab":
                            win_2 += 1
                    playing = False
                    break
        return draw, win_1, win_2




# cProfile.run('human_vs_agent(generate_move_mc, generate_move_min_ab)', 'minimax_abeta')


if __name__ == "__main2__":
    #cProfile.run('human_vs_agent(generate_move_mc, generate_move_min_ab)', 'MCTS_vs_minimax_abeta_5sec')
    human_vs_agent(generate_move_mc, generate_move_min_ab, 'Montecarlo', 'Minimax_ab')


draw = 0
win_1 = 0
win_2 = 0

for c in range(10):

    draw, win_1, win_2 = human_vs_agent(generate_move_mc, generate_move_min_ab,draw=draw, win_1=win_1, win_2=win_2)
print(f'Draw {draw}, win montecarlo {win_1}, win minimax {win_2}')