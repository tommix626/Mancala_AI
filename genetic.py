import random
import numpy as np
import torch
import torch.nn as nn
import pygad.torchga
import pygad

from GA_model import MancalaNet
from fitness import fitness_function
from mancala import MancalaGame, evaluate_win_rate


def on_generation(ga_instance, context):
    """
    Callback function to be called after each generation.
    Uses the context object for accessing shared variables.
    """
    print("=====================================")

    # Retrieve the best solution from the current generation
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Generation {ga_instance.generations_completed}:")
    print(f"Best Fitness: {solution_fitness:.4f}, model_idx: {solution_idx}")

    # Load the best solution's weights into the model
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=context.model, weights_vector=solution)
    context.model.load_state_dict(model_weights_dict)

    # Evaluate the model's performance
    win_rate = evaluate_win_rate(context.model, num_games=1000)
    print(f"Win Rate: {win_rate:.2%}")

    # Save the model
    filename = f"best_model_gen_{ga_instance.generations_completed}_win_{win_rate:.4%}.pth"
    context.model.save(filename)
    print(f"Model saved as: {filename}")

    # Increment the number of games, with an upper limit
    context.num_games += 0.1
    context.num_games = min(context.num_games, 50)
    print(f"Incrementing num_games to {context.num_games:.1f}")

    print("=====================================\n\n")