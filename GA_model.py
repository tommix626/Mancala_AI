import torch
import torch.nn as nn
import numpy as np
import random


# -----------------------------
# Neural Network Model
# -----------------------------


class MancalaNet(nn.Module):
    def __init__(self, position):
        super(MancalaNet, self).__init__()
        assert position in [1,2]
        self.position = position
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
        state = torch.FloatTensor(game.get_state(player_perspective=self.position))
        with torch.no_grad():
            q_values = self(state)
        legal_moves = game.get_legal_moves()
        q_values_np = q_values.detach().numpy()
        q_values_np = [q_values_np[i % 6] if i in legal_moves else -float('inf') for i in range(6)]
        return legal_moves[np.argmax([q_values_np[i % 6] for i in legal_moves])]

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

class RandomPlayer:
    def choose_move(self, game):
        legal_moves = game.get_legal_moves()
        return random.choice(legal_moves)