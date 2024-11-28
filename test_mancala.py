# test_mancala.py

import unittest
import numpy as np
import random
import torch

from mancala import MancalaGame, RandomPlayer, RLAgent, GAAgent, calculate_reward

class TestMancalaGameLogic(unittest.TestCase):

    def setUp(self):
        self.game = MancalaGame()

    def test_initial_board_setup(self):
        # Test that the initial board has the correct number of stones
        expected_board = [4]*6 + [0] + [4]*6 + [0]
        self.assertEqual(self.game.board, expected_board)
        self.assertEqual(self.game.current_player, 1)

    def test_move_execution(self):
        # Player 1 makes a move from pit 0
        self.game.make_move(0)
        # Expected board after move
        expected_board = [0, 5, 5, 5, 5, 4, 0, 4, 4, 4, 4, 4, 4, 0]
        self.assertEqual(self.game.board, expected_board)

    def test_capture_mechanic(self):
        # Set up a scenario where a capture should occur
        self.game.board = [0, 0, 0, 1, 0, 0, 0, 8, 8, 8, 8, 8, 7, 0]
        self.game.current_player = 1
        self.game.make_move(3)
        # Player 1 should capture stones from pit 9
        expected_board = [0, 0, 0, 0, 1, 0, 8, 8, 8, 8, 8, 0, 7, 0]
        self.assertEqual(self.game.board, expected_board)
        self.assertEqual(self.game.board[6], 8)  # Mancala store updated

    def test_extra_turn(self):
        # Player 1 makes a move that ends in their Mancala
        self.game.board = [4, 4, 4, 4, 4, 1, 3, 4, 4, 4, 4, 4, 4, 0]
        self.game.current_player = 1
        _, game_over = self.game.make_move(5)
        # Player 1 should get an extra turn
        self.assertEqual(self.game.current_player, 1)
        self.assertFalse(game_over)

    def test_game_over_detection(self):
        # Empty one side of the board to trigger game over
        self.game.board = [0, 0, 0, 0, 0, 0, 24, 4, 4, 4, 4, 4, 4, 0]
        self.game.current_player = 2
        self.game.check_game_over()
        self.assertTrue(self.game.game_over)
        # Remaining stones should be collected
        self.assertEqual(self.game.board[13], 24)
        for i in range(14):
            if i not in [6, 13]:
                self.assertEqual(self.game.board[i], 0)

    def test_invalid_move(self):
        # Attempt to make a move from an empty pit
        self.game.board[0] = 0
        legal_moves = self.game.get_legal_moves()
        self.assertNotIn(0, legal_moves)

class TestRLAgent(unittest.TestCase):

    def setUp(self):
        self.agent = RLAgent()
        self.opponent = RandomPlayer()
        self.game = MancalaGame()

    def test_state_representation(self):
        state = self.game.get_state()
        self.assertEqual(len(state), 15)  # 14 pits + current player
        self.assertIsInstance(state, np.ndarray)

    def test_reward_function_win(self):
        self.game.board[6] = 25  # Player 1's Mancala
        self.game.board[13] = 23  # Player 2's Mancala
        reward = calculate_reward(self.game, True, player=1)
        self.assertEqual(reward, 10)

    def test_reward_function_loss(self):
        self.game.board[6] = 20
        self.game.board[13] = 30
        reward = calculate_reward(self.game, True, player=1)
        self.assertEqual(reward, -10)

    def test_choose_move(self):
        move = self.agent.choose_move(self.game)
        self.assertIn(move, self.game.get_legal_moves())

    def test_policy_network_update(self):
        # Create a fake transition
        state = torch.FloatTensor(self.game.get_state())
        action = 2
        reward = 1.0
        next_state = torch.FloatTensor(self.game.get_state())
        done = 0.0
        self.agent.memory.push(state.numpy(), action, reward, next_state.numpy(), done)
        initial_weights = self.agent.policy_net.fc1.weight.clone()
        self.agent.optimize_model(batch_size=1)
        updated_weights = self.agent.policy_net.fc1.weight
        self.assertFalse(torch.equal(initial_weights, updated_weights))

class TestGAAgent(unittest.TestCase):

    def setUp(self):
        self.agent = GAAgent()
        self.opponent = GAAgent()
        self.population = [GAAgent() for _ in range(10)]

    def test_choose_move(self):
        game = MancalaGame()
        move = self.agent.choose_move(game)
        self.assertIn(move, game.get_legal_moves())

    def test_fitness_function(self):
        # Use a small population for testing
        fitness_scores = [fitness(agent, self.population, num_games=2) for agent in self.population]
        self.assertEqual(len(fitness_scores), len(self.population))
        for score in fitness_scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_crossover(self):
        parent1 = GAAgent(weights=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        parent2 = GAAgent(weights=np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
        child = crossover(parent1, parent2)
        self.assertEqual(len(child.weights), 6)
        # Check that child's weights are from parents
        for w in child.weights:
            self.assertIn(w, np.concatenate((parent1.weights, parent2.weights)))

    def test_mutation(self):
        original_weights = self.agent.weights.copy()
        mutated_agent = mutate(self.agent, mutation_rate=1.0)  # Force mutation
        self.assertFalse(np.array_equal(original_weights, mutated_agent.weights))

class TestIntegration(unittest.TestCase):

    def test_rl_agent_vs_random(self):
        agent = RLAgent()
        opponent = RandomPlayer()
        game = MancalaGame()
        while not game.game_over:
            if game.current_player == 1:
                action = agent.choose_move(game)
            else:
                action = opponent.choose_move(game)
            game.make_move(action)
        # Game should complete without errors
        self.assertTrue(game.game_over)

    def test_ga_agents_competition(self):
        agents = [GAAgent() for _ in range(5)]
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                game = MancalaGame()
                agent1 = agents[i]
                agent2 = agents[j]
                while not game.game_over:
                    if game.current_player == 1:
                        action = agent1.choose_move(game)
                    else:
                        action = agent2.choose_move(game)
                    game.make_move(action)
                # Game should complete without errors
                self.assertTrue(game.game_over)

class TestPerformance(unittest.TestCase):

    def test_rl_training_time(self):
        agent = RLAgent()
        opponent = RandomPlayer()
        game = MancalaGame()
        import time
        start_time = time.time()
        for _ in range(10):
            game.reset()
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
        end_time = time.time()
        self.assertLess(end_time - start_time, 60)  # Training should take less than 60 seconds

class TestEdgeCases(unittest.TestCase):

    def test_full_gameplay_random(self):
        game = MancalaGame()
        players = [RandomPlayer(), RandomPlayer()]
        while not game.game_over:
            current_player = game.current_player
            action = players[current_player - 1].choose_move(game)
            game.make_move(action)
        # Verify that all stones are in the Mancalas
        total_stones = sum(game.board)
        self.assertEqual(total_stones, game.board[6] + game.board[13])

    def test_invalid_state_deserialization(self):
        with self.assertRaises(ValueError):
            MancalaGame.deserialize('invalid_state_string')

    def test_negative_stones(self):
        # Force negative stones in a pit to test robustness
        self.game.board[0] = -1
        with self.assertRaises(ValueError):
            self.game.make_move(0)

if __name__ == '__main__':
    unittest.main()