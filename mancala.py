# mancala.py
import os

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
                self.board[index] = 0 # TODO: Make this our own version where we don't capture the last stone
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
            return True
        return False

    def collect_remaining_stones(self):
        self.board[6] += sum(self.board[0:6])
        self.board[13] += sum(self.board[7:13])
        for i in range(14):
            if i != 6 and i != 13:
                self.board[i] = 0

    def check_winner(self):
        if self.game_over:
            if self.board[6] > self.board[13]:
                return 1
            elif self.board[6] < self.board[13]:
                return 2
            else:
                return 0
        else:
            raise ValueError("Game is not over yet.")

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
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayMemory()
        self.steps_done = 0
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9998
        self.gamma = 0.9  # Discount factor
        # print a message for all the constants
        print(f"state_size: {self.state_size}")
        print(f"action_size: {self.action_size}")
        print(f"epsilon: {self.epsilon}")
        print(f"epsilon_min: {self.epsilon_min}")
        print(f"epsilon_decay: {self.epsilon_decay}")
        print(f"gamma: {self.gamma}")


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

        new_epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if (self.epsilon>0.5 and new_epsilon<=0.5):
            print("Epsilon has reached 0.5")
        elif (self.epsilon>0.3 and new_epsilon<=0.3):
            print("Epsilon has reached 0.3")
        elif (self.epsilon>0.1 and new_epsilon<=0.1):
            print("Epsilon has reached 0.1")
        self.epsilon = new_epsilon

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath="rl_agent.pth"):
        """Saves the RL agent's policy network to a file."""
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Model saved to {filepath}.")

    def load_model(self, filepath="rl_agent.pth"):
        """Loads the RL agent's policy network from a file."""
        if os.path.exists(filepath):
            self.policy_net.load_state_dict(torch.load(filepath))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {filepath}.")
        else:
            print(f"No model found at {filepath}. Starting with a fresh model.")

class DDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # Increased the number of neurons for a larger network
        self.fc2 = nn.Linear(128, 128)        # Added more depth
        self.fc3 = nn.Linear(128, 128)        # New layer
        self.fc4 = nn.Linear(128, 64)         # Added one more layer before output
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.out(x)


class DRLAgent(RLAgent):
    # use DDQN instead of DQN
    def __init__(self, action_size=6):
        self.state_size = 14 + 1
        self.action_size = action_size
        self.policy_net = DDQN(self.state_size, self.action_size)
        self.target_net = DDQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayMemory()
        self.steps_done = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9998
        self.gamma = 0.8
        # print a message for all the constants
        print(f"state_size: {self.state_size}")
        print(f"action_size: {self.action_size}")
        print(f"epsilon: {self.epsilon}")
        print(f"epsilon_min: {self.epsilon_min}")
        print(f"epsilon_decay: {self.epsilon_decay}")
        print(f"gamma: {self.gamma}")


# -----------------------------
# Helper Functions
# -----------------------------

def calculate_reward(game, game_over, ai_player=1):
    ai_mancala = 6 if ai_player == 1 else 13
    opponent_mancala = 13 if ai_player == 1 else 6

    if game_over:
        if game.board[ai_mancala] > game.board[opponent_mancala]:  # AI wins
            return 10
        elif game.board[ai_mancala] < game.board[opponent_mancala]:  # Opponent wins
            return -10
        else:  # Draw
            return 0
    else:
        # Provide intermediate reward based on Mancala difference
        total_stones = sum(game.board)
        if total_stones == 0:  # Prevent division by zero
            return 0
        return (game.board[ai_mancala] - game.board[opponent_mancala]) / total_stones
def evaluate_win_rate(agent, num_games=10000):
    """Evaluates the win rate of the RL agent against a RandomPlayer."""
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
# Training Functions
# -----------------------------

def train_rl_agent(num_episodes=10000, eval_interval=500):
    agent = DRLAgent(action_size=6)
    opponent = RandomPlayer()
    win_rates = []

    for episode in range(num_episodes):
        game = MancalaGame()
        state = game.get_state()
        while not game.game_over:
            if game.current_player == 1:
                action = agent.choose_move(game)
                player = 1  # After the last line, game.current_player will change, so use player instead
            else:
                action = opponent.choose_move(game)
                player = 2
            next_state, game_over = game.make_move(action)
            reward = calculate_reward(game, game_over)
            done = float(game_over)
            if player == 1:  # Only store AI's transitions
                agent.memory.push(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
        agent.update_epsilon()
        if episode % 10 == 0:
            agent.update_target_net()

        # Evaluate win rate every eval_interval episodes
        if (episode + 1) % eval_interval == 0:
            print(f"Evaluating at Episode {episode + 1}...")
            win_rate = evaluate_win_rate(agent, num_games=10000)
            win_rates.append((episode + 1, win_rate))

    print("Evaluating Final RL Agent...")
    final_win_rate = evaluate_win_rate(agent, num_games=100000)
    # Save the model after training
    class_name = agent.__class__.__name__
    model_name = f"{class_name}_hidden{agent.policy_net.hidden_size}_epsilon{agent.epsilon:.2f}_gamma{agent.gamma}_winrate{final_win_rate}.pth"
    agent.save_model(f"saved_models/{model_name}")
    return agent, win_rates

# -----------------------------
# Main Execution Block
# -----------------------------

if __name__ == "__main__":
    # make torch deterministic
    torch.manual_seed(1)

    print("Mancala Game Simulator")

    # Example: Evaluate the win rate of a RandomPlayer
    print("Evaluating Random Player...")
    evaluate_win_rate(RandomPlayer(), num_games=1000)

    # Train the Reinforcement Learning Agent
    print("Training Reinforcement Learning Agent...")
    rl_agent, win_rates = train_rl_agent(num_episodes=20000)
    print("RL Agent Training Completed.")




    # plot win rate
    import matplotlib.pyplot as plt
    epochs, win_rates = zip(*win_rates)
    plt.plot(epochs, win_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Win Rate')
    plt.title('RL Agent Win Rate over Training')
    plt.show()


    # for epoch, win_rate in win_rates:
    #     print(f"Epoch {epoch}: Win Rate = {win_rate:.2%}")

    # Example: Load and re-evaluate the model
    # print("Loading and Evaluating Saved Model...")
    # saved_agent = RLAgent()
    # saved_agent.load_model()
    # evaluate_win_rate(saved_agent, num_games=100000)
