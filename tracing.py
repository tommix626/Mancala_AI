import torch
from mancala import MancalaGame, RLAgent, RandomPlayer  # Import your Mancala components


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
    print(" | ", " | ".join(map(str, game.board[12:6:-1])), " | ")
    print(f"{game.board[13]}{' ' * 28}{game.board[6]}")
    print(" | ", " | ".join(map(str, game.board[0:6])), " | ")
    print("  Player Side")


if __name__ == "__main__":
    # Load the trained RLAgent model
    rl_agent = RLAgent(action_size=6)
    rl_agent.load_model(filepath="saved_models/RLAgent_hidden64_epsilon0.05_gamma0.8_winrate0.56387.pth")

    # Choose an opponent (RandomPlayer or another RLAgent)
    random_opponent = RandomPlayer()
    # Alternatively, you can have self-play:
    # random_opponent = RLAgent(action_size=6)
    # random_opponent.load_model(filepath="rl_agent.pth")

    print("Simulating Game Between RL Agent and Random Player...\n")
    trace, winner = play_game(rl_agent, random_opponent, verbose=True)

    print("\nTrace of Moves:")
    for player, action in trace:
        print(f"Player {player} chose pit {action}")