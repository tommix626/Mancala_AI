# -----------------------------
# Main Execution of the Genetic Algorithm
# -----------------------------
from pygad import pygad
import pygad
from GA_model import MancalaNet, RandomPlayer
from context import GAContext
from fitness import fitness_function
from genetic import on_generation
from mancala import evaluate_win_rate

if __name__ == "__main__":
    # Initialize the neural network model
    model = MancalaNet(position=1)

    # Initialize the opponent
    opponent = RandomPlayer()
    num_games = 20 # start with a small number, increase it as we evolve the population.

    # Create a context object
    context = GAContext(model=model, opponent=opponent, num_games=num_games)
    context.TIME_PENALTY_ENABLED = True
    context.TIME_PENALTY_WEIGHT = 0.001

    # In population competition parameters
    context.POPULATION_COMPETITION_ENABLED = True # TODO: Enable after implementing AI vs AI version of the game. (AI can be player 1 or player 2) add get statte by player perspective, our last feature gives AI change to modify its strategy by first/second hand.
    context.COMPETITION_WEIGHT = 0.1

    context.INITIAL_PERCENT = 0.0  # Initial percent of top solutions to consider
    context.MAX_PERCENT = 0.05  # Maximum percent of top solutions to consider
    context.PERCENT_INCREASE_RATE = 0.001  # Increment per generation for the top-percent size
    context.SUBSAMPLE_RATIO = 0.25  # Fraction of subsamples taken from the top candidates

    # Prepare the genetic algorithm
    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=20)

    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=5,
        initial_population=torch_ga.population_weights,
        fitness_func=lambda ga_instance, solution, sol_idx: fitness_function(ga_instance, solution, sol_idx, context),
        mutation_percent_genes=10,
        mutation_type="random",
        mutation_probability=0.1,
        crossover_type="single_point",
        parent_selection_type="tournament",
        keep_parents=2,
        on_generation=lambda ga_instance: on_generation(ga_instance, context),
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
    evaluate_win_rate(model, num_games=10000)
    evaluate_win_rate(model, num_games=100000)