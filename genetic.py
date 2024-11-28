import random
import numpy as np
import torch
import torch.nn as nn
import pygad.torchga
import pygad
from mancala import MancalaGame, evaluate_win_rate


# -----------------------------
# Neural Network Model
# -----------------------------

class MancalaNet(nn.Module):
    def __init__(self):
        super(MancalaNet, self).__init__()
        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# -----------------------------
# Genetic Algorithm Components
# -----------------------------

def fitness_function(ga_instance, solution, sol_idx):
    global model, opponent

    # Create a copy of the model and set its weights from the solution
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)

    # Evaluate the model's performance
    wins = 0
    num_games = 5
    for _ in range(num_games):
        game = MancalaGame()
        while not game.game_over:
            state = torch.FloatTensor(game.get_state())
            if game.current_player == 1:
                with torch.no_grad():
                    q_values = model(state)
                legal_moves = game.get_legal_moves()
                q_values_np = q_values.detach().numpy()
                q_values_np = [q_values_np[i % 6] if i in legal_moves else -float('inf') for i in range(6)]
                action = legal_moves[np.argmax([q_values_np[i % 6] for i in legal_moves])]
            else:
                action = opponent.choose_move(game)
            game.make_move(action)
        if game.check_winner() == 1:
            wins += 1
    return wins / num_games


class RandomPlayer:
    def choose_move(self, game):
        legal_moves = game.get_legal_moves()
        return random.choice(legal_moves)


def evaluate_win_rate_with_weights(agent, num_games=10000):
    """Evaluates the win rate of the agent against a RandomPlayer."""
    wins = 0
    draws = 0
    losses = 0
    for _ in range(num_games):
        game = MancalaGame()
        random_player = RandomPlayer()
        while not game.game_over:
            if game.current_player == 1:
                action = agent.choose_move(game)
            else:
                action = random_player.choose_move(game)
            game.make_move(action)
        if game.check_winner() == 1:
            wins += 1
        elif game.check_winner() == 2:
            losses += 1
        else:
            draws += 1
    win_rate = wins / num_games
    print(f"Win Rate: {win_rate:.2%}, Draws: {draws/num_games:.2%}, Losses: {losses/num_games:.2%}")
    return win_rate

# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    # Initialize the neural network model
    model = MancalaNet()

    # Initialize the opponent
    opponent = RandomPlayer()

    # Prepare the genetic algorithm
    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=10)

    ga_instance = pygad.GA(
        num_generations=10,
        num_parents_mating=5,
        initial_population=torch_ga.population_weights,
        fitness_func=fitness_function,
        mutation_percent_genes=10,
        mutation_type="random",
        mutation_probability=0.1,
        crossover_type="single_point",
        parent_selection_type="tournament",
        keep_parents=2
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    # Evaluate the best GA agent
    print("Evaluating the best GA Agent...")
    evaluate_win_rate(solution, num_games=1000)