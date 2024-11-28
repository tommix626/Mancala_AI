# mancala.py

import numpy as np
import random
from collections import deque, namedtuple

# For Reinforcement Learning
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Mancala Game Simulator
# -----------------------------

class MancalaGame:
    def __init__(self, stones_per_pit=4):
        self.stones_per_pit = stones_per_pit
        self.reset()

    def reset(self):
        # Initialize the board: 6 pits per player and their Mancalas
        self.board = [self.stones_per_pit]*6 + [0] + [self.stones_per_pit]*6 + [0]
        self.current_player = 1  # Player 1 starts
        self.game_over = False

    def get_legal_moves(self):
        # Return indices of non-empty pits for the current player
        if self.current_player == 1:
            return [i for i in range(6) if self.board[i] > 0]
        else:
            return [i for i in range(7, 13) if self.board[i] > 0]

    def make_move(self, pit_index):
        stones = self.board[pit_index]
        self.board[pit_index] = 0
        index = pit_index
        while stones > 0:
            index = (index + 1) % 14
            # Skip opponent's Mancala
            if (self.current_player == 1 and index == 13) or (self.current_player == 2 and index == 6):
                continue
            self.board[index] += 1
            stones -= 1
        self.handle_capture(index)
        self.check_game_over()
        if not self.game_over and not self.is_extra_turn(index):
            self.current_player = 2 if self.current_player == 1 else 1
        return self.get_state(), self.game_over

    def handle_capture(self, index):
        if self.current_player == 1 and 0 <= index <= 5 and self.board[index] == 1:
            opposite_index = 12 - index
            if self.board[opposite_index] > 0:
                self.board[6] += self.board[opposite_index] + 1
                self.board[index] = 0
                self.board[opposite_index] = 0
        elif self.current_player == 2 and 7 <= index <= 12 and self.board[index] == 1:
            opposite_index = 12 - index
            if self.board[opposite_index] > 0:
                self.board[13] += self.board[opposite_index] + 1
                self.board[index] = 0
                self.board[opposite_index] = 0

    def is_extra_turn(self, index):
        return (self.current_player == 1 and index == 6) or (self.current_player == 2 and index == 13)

    def check_game_over(self):
        if sum(self.board[0:6]) == 0 or sum(self.board[7:13]) == 0:
            self.game_over = True
            self.collect_remaining_stones()

    def collect_remaining_stones(self):
        self.board[6] += sum(self.board[0:6])
        self.board[13] += sum(self.board[7:13])
        for i in range(14):
            if i != 6 and i != 13:
                self.board[i] = 0

    def get_state(self):
        # Returns the current game state as a numpy array
        return np.array(self.board + [self.current_player])

    def serialize(self):
        return ','.join(map(str, self.board + [self.current_player]))

    @staticmethod
    def deserialize(state_str):
        values = list(map(int, state_str.split(',')))
        game = MancalaGame()
        game.board = values[:-1]
        game.current_player = values[-1]
        return game

# -----------------------------
# Player Interface and Opponents
# -----------------------------

class Player:
    def choose_move(self, game):
        raise NotImplementedError("This method should be overridden by subclasses.")

class RandomPlayer(Player):
    def choose_move(self, game):
        legal_moves = game.get_legal_moves()
        return random.choice(legal_moves)

# -----------------------------
# Reinforcement Learning Components
# -----------------------------

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class RLAgent(Player):
    def __init__(self, action_size=6):
        self.state_size = 14 + 1  # Board state + current player
        self.action_size = action_size
        self.policy_net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory()
        self.steps_done = 0
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor

    def choose_move(self, game):
        state = torch.FloatTensor(game.get_state())
        legal_moves = game.get_legal_moves()
        if random.random() < self.epsilon:
            action = random.choice(legal_moves)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                q_values_np = q_values.detach().numpy()
                # Mask illegal moves
                q_values_np = [q_values_np[i%6] if i in legal_moves else -float('inf') for i in range(6)]
                action = legal_moves[np.argmax([q_values_np[i%6] for i in legal_moves])]
        return action

    def optimize_model(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(batch.next_state)
        done_batch = torch.FloatTensor(batch.done)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = nn.functional.mse_loss(q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# -----------------------------
# Genetic Algorithm Components
# -----------------------------

class GAAgent(Player):
    def __init__(self, weights=None):
        if weights is None:
            self.weights = np.random.rand(6)
        else:
            self.weights = weights

    def choose_move(self, game):
        legal_moves = game.get_legal_moves()
        move_scores = {i: self.weights[i%6] for i in legal_moves}
        return max(move_scores, key=move_scores.get)

# -----------------------------
# Helper Functions
# -----------------------------

def calculate_reward(game, game_over, player):
    if game_over:
        if game.board[6] > game.board[13]:
            return 10 if player == 1 else -10
        elif game.board[6] < game.board[13]:
            return -10 if player == 1 else 10
        else:
            return 0
    else:
        return (game.board[6] - game.board[13]) / sum(game.board)

# -----------------------------
# Training Functions
# -----------------------------

def train_rl_agent(num_episodes=1000):
    agent = RLAgent(action_size=6)
    opponent = RandomPlayer()

    for episode in range(num_episodes):
        game = MancalaGame()
        state = game.get_state()
        while not game.game_over:
            if game.current_player == 1:
                action = agent.choose_move(game)
                player = 1
            else:
                action = opponent.choose_move(game)
                player = 2
            next_state, game_over = game.make_move(action)
            reward = calculate_reward(game, game_over, player)
            done = float(game_over)
            agent.memory.push(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
        agent.update_epsilon()
        if episode % 10 == 0:
            agent.update_target_net()
    return agent

def evolve_population(population_size=50, generations=100, mutation_rate=0.1):
    population = [GAAgent() for _ in range(population_size)]
    for generation in range(generations):
        fitnesses = [fitness(agent, population) for agent in population]
        total_fitness = sum(fitnesses)
        selection_probs = [f / total_fitness for f in fitnesses]
        selected_indices = np.random.choice(len(population), size=len(population), p=selection_probs)
        selected_agents = [population[i] for i in selected_indices]
        next_generation = []
        for i in range(0, population_size, 2):
            parent1 = selected_agents[i]
            parent2 = selected_agents[(i+1)%population_size]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            next_generation.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        population = next_generation
    return population

# -----------------------------
# Genetic Algorithm Functions
# -----------------------------

def fitness(agent, opponents, num_games=5):
    score = 0
    for _ in range(num_games):
        opponent = random.choice(opponents)
        game = MancalaGame()
        while not game.game_over:
            if game.current_player == 1:
                action = agent.choose_move(game)
            else:
                action = opponent.choose_move(game)
            game.make_move(action)
        if game.board[6] > game.board[13]:
            score += 1
    return score / num_games

def crossover(parent1, parent2):
    crossover_point = random.randint(1, 5)
    child_weights = np.concatenate((parent1.weights[:crossover_point], parent2.weights[crossover_point:]))
    return GAAgent(weights=child_weights)

def mutate(agent, mutation_rate=0.1):
    for i in range(len(agent.weights)):
        if random.random() < mutation_rate:
            agent.weights[i] = np.random.rand()
    return agent

# -----------------------------
# Main Execution Block
# -----------------------------

if __name__ == "__main__":
    # Train the Reinforcement Learning Agent
    print("Training Reinforcement Learning Agent...")
    rl_agent = train_rl_agent(num_episodes=1000)
    print("RL Agent Training Completed.")

    # # Evolve the Genetic Algorithm Agents
    # print("Evolving Genetic Algorithm Agents...")
    # ga_population = evolve_population(population_size=50, generations=100, mutation_rate=0.1)
    # print("GA Agents Evolution Completed.")