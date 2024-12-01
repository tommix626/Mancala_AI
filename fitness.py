# -----------------------------
# Configurable Constants
# -----------------------------
import numpy as np
import pygad
import pygad.torchga

from GA_model import MancalaNet
from mancala import MancalaGame

TIME_PENALTY_ENABLED = True  # Enable or disable time penalty
TIME_PENALTY_WEIGHT = 0.001  # Weight of the time penalty (penalize per round)

POPULATION_COMPETITION_ENABLED = True  # Enable or disable population competition
COMPETITION_WEIGHT = 0.1  # Weight of the competition reward
NUM_COMPETITION_OPPONENTS = 10  # Number of opponents to compete with in the population

# -----------------------------
# Fitness Function
# -----------------------------
def fitness_function(ga_instance, solution, sol_idx, context):
    """
    Fitness function to evaluate the performance of a solution.
    """

    # Load weights from the solution vector into the model
    model_weights_dict = pygad.torchga.model_weights_as_dict(model=context.model, weights_vector=solution)
    context.model.load_state_dict(model_weights_dict)

    # Main fitness metrics
    total_wins = 0
    total_rounds = 0
    num_games_int = int(context.num_games)

    # Play games against the baseline opponent
    for _ in range(num_games_int):
        game = MancalaGame()
        rounds = 0
        while not game.game_over:
            rounds += 1
            if game.current_player == 1:
                action = context.model.choose_move(game)
                game.make_move(action, player_perspective=1)
            else:
                action = context.opponent.choose_move(game)
                game.make_move(action, player_perspective=2)

        if game.check_winner() == 1:
            total_wins += 1
        total_rounds += rounds

    # Calculate win rate as primary fitness score
    win_rate = total_wins / context.num_games

    # Apply time penalty (optional)
    if context.TIME_PENALTY_ENABLED:
        avg_rounds = total_rounds / context.num_games
        time_penalty = avg_rounds * context.TIME_PENALTY_WEIGHT
        win_rate -= time_penalty

    # Population competition (optional)
    if context.POPULATION_COMPETITION_ENABLED:
        competitors = dynamic_top_k_population(ga_instance, context)  # Get dynamic top-K competitors
        competition_score = 0
        for competitor_weights in competitors:
            # Load competitor's weights
            competitor_model = MancalaNet()
            competitor_weights_dict = pygad.torchga.model_weights_as_dict(
                model=competitor_model, weights_vector=competitor_weights
            )
            competitor_model.load_state_dict(competitor_weights_dict)

            # Simulate a game between the models
            game = MancalaGame()
            while not game.game_over:
                if game.current_player == 1:
                    action = context.model.choose_move(game)
                    game.make_move(action, player_perspective=1)
                else:
                    action = competitor_model.choose_move(game)
                    game.make_move(action, player_perspective=2)

            # Reward based on the outcome
            if game.check_winner() == 1:
                competition_score += 1

        # Normalize and add to fitness
        win_rate += (competition_score / len(competitors)) * context.COMPETITION_WEIGHT

    return win_rate

def dynamic_top_k_population(ga_instance, context):
    """
    Select a dynamic top-K population, gradually increasing the threshold and introducing randomness.
    """
    # Update the percentage to consider based on the generation
    current_percent = min(
        context.INITIAL_PERCENT + ga_instance.generations_completed * context.PERCENT_INCREASE_RATE,
        context.MAX_PERCENT
    )

    # Calculate the top-K candidates based on fitness
    num_candidates = int(current_percent * len(ga_instance.population))
    num_candidates = max(1, num_candidates)  # Ensure at least one candidate

    # Subsample randomly from a broader range (e.g., top 20%)
    broader_range = int(num_candidates / context.SUBSAMPLE_RATIO)
    broader_range = min(broader_range, len(ga_instance.population))  # Limit to the total population
    candidate_indices = np.argsort(ga_instance.last_generation_fitness)[-broader_range:]  # Top indices in fitness
    print(f"debug:: broader_range: {broader_range}, candidate_indices: {candidate_indices}")
    subsample_indices = np.random.choice(candidate_indices, size=num_candidates, replace=False)

    # Return the selected candidates
    return [ga_instance.population[idx] for idx in subsample_indices]