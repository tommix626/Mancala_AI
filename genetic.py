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
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.out(x)

    def choose_move(self, game):
        state = torch.FloatTensor(game.get_state())
        with torch.no_grad():
            q_values = self(state)
        legal_moves = game.get_legal_moves()
        q_values_np = q_values.detach().numpy()
        q_values_np = [q_values_np[i % 6] if i in legal_moves else -float('inf') for i in range(6)]
        return legal_moves[np.argmax([q_values_np[i % 6] for i in legal_moves])]

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

# -----------------------------
# Genetic Algorithm Components
# -----------------------------

def fitness_function(ga_instance, solution, sol_idx):
    global model, opponent, num_games

    # Set model weights from the solution
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)

    # Evaluate the model's performance
    wins = 0
    num_games_int = int(num_games)
    for _ in range(num_games_int):
        game = MancalaGame()
        while not game.game_over:
            if game.current_player == 1:
                action = model.choose_move(game)
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

# Callback function to be called after each generation
def on_generation(ga_instance):
    print("=====================================")
    global model, opponent, num_games
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Generation {ga_instance.generations_completed}:")
    print(f"Best Fitness: {solution_fitness:.4f}, model_idx: {solution_idx}")
    # Load the best solution's weights into the model
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    # Evaluate the model's performance
    win_rate = evaluate_win_rate(model, num_games=1000)
    print(f"Win Rate: {win_rate:.2%}")
    # Save the model
    model.save(f"best_model_gen_{ga_instance.generations_completed}_win_{win_rate:.4%}.pth")
    num_games += 0.1
    num_games = min(num_games, 50)
    print(f"incrementing num_games to {num_games:.1f}")
    print("=====================================\n\n")

# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    # Initialize the neural network model
    model = MancalaNet()

    # Initialize the opponent
    opponent = RandomPlayer()
    num_games = 20 # start with a small number, increase it as we evolve the population.

    # Prepare the genetic algorithm
    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=100)

    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=5,
        initial_population=torch_ga.population_weights,
        fitness_func=fitness_function,
        mutation_percent_genes=10,
        mutation_type="random",
        mutation_probability=0.1,
        crossover_type="single_point",
        parent_selection_type="tournament",
        keep_parents=2,
        on_generation=on_generation,  # The callback function
        parallel_processing=3
    )
    ga_instance.run()

    # Final evaluation of the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Final Best Solution Parameters:", solution)
    print("Final Best Solution Fitness:", solution_fitness)
    # Load the best solution's weights into the model
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    # Save the final best model
    model.save("best_ga_agent_final.pth")
    # Evaluate the final best model
    print("Evaluating the final best GA agent...")
    evaluate_win_rate(model, num_games=1000)