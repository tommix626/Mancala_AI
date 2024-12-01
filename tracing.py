import torch
from mancala import MancalaGame, RLAgent, RandomPlayer, evaluate_win_rate  # Import your Mancala components
from mancala import *
# NUMBER_OF_PITS = 3
# TOTAL_PITS = 2*NUMBER_OF_PITS + 2
# STONE_PER_PIT = 1
# LEARNING_RATE= 0.00001
def play_game(agent1, agent2, verbose=True):
    """
    Simulates a game between two agents and logs the gameplay trace.

    Parameters:
    - agent1: The first player (e.g., RLAgent).
    - agent2: The second player (e.g., RandomPlayer or RLAgent).
    - verbose: If True, prints the board state after each move.

    Returns:
    - winner: The winner of the game (1, 2, or 0 for draw).
    """
    game = MancalaGame()
    move_log = []

    if verbose:
        print("\nInitial Board State:")
        print_board(game)

    while not game.game_over:
        current_agent = agent1 if game.current_player == 1 else agent2
        action = current_agent.choose_move(game)
        move_log.append((game.current_player, action))

        if verbose:
            print(f"\nPlayer {game.current_player} chooses pit {action}.")

        game.make_move(action)

        if verbose:
            print_board(game)

    winner = game.check_winner()
    if verbose:
        print("\nGame Over!")
        print(f"Winner: {'Player ' + str(winner) if winner != 0 else 'Draw'}")

    return move_log, winner


def print_board(game):
    """
    Pretty-prints the Mancala board state.
    """
    print("  Opponent Side")
    print(" | ", " | ".join(map(str, game.board[TOTAL_PITS-2:NUMBER_OF_PITS:-1])), " | ")
    print(f"{game.board[TOTAL_PITS-1]}{' ' * 28}{game.board[NUMBER_OF_PITS]}")
    print(" | ", " | ".join(map(str, game.board[0:NUMBER_OF_PITS])), " | ")
    print("  Player Side")


if __name__ == "__main__":
    # Load the trained RLAgent model
    rl_agent = RLAgent(action_size=NUMBER_OF_PITS)
    # rl_agent.load_model(filepath="saved_models/1delta-RLAgent_hidden64_epsilon0.05_gamma0.8_epoch14999_winrate0.477.pth") #This is a model with loss rate of 0.5% (3pit of 1)
    rl_agent.load_model(filepath="saved_models/1delta-RLAgent_hidden64_epsilon0.05_gamma0.8_epoch29999_winrate0.95569.pth") #This is a model with loss rate of 2.3% (3pit of 2)
    # rl_agent.load_model(filepath="saved_models/1delta-RLAgent_hidden64_epsilon0.05_gamma0.8_epoch24999_winrate0.02021.pth") #This is a model with loss rate of 0.3% (3pit of 1) but always tie
    rl_agent.epsilon = 0
    # Choose an opponent (RandomPlayer or another RLAgent)
    random_opponent = RandomPlayer()
    # Alternatively, you can have self-play:
    # random_opponent = RLAgent(action_size=6)
    # random_opponent.load_model(filepath="rl_agent.pth")
    evaluate_win_rate(rl_agent, num_games=10000)
    print("Simulating Game Between RL Agent and Random Player...\n")
    trace, winner = play_game(rl_agent, random_opponent, verbose=True)
    # without verbose, do it 10000 times and calculate the win rate
    win = 0
    lose = 0
    tie = 0
    for i in range(10000):
        trace, winner = play_game(rl_agent, random_opponent, verbose=False)
        if winner == 1:
            win += 1
        elif winner == 2:
            lose += 1
        else:
            tie += 1
    print("Win rate: ", win/10000)

    print("\nTrace of Moves:")
    for player, action in trace:
        print(f"Player {player} chose pit {action}")