# rl_agent.py
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import game

# Enhanced State Encoding Parameters
COLOR_CHANNELS = 5  # Number of colors in game.COLOR_LIST
TYPE_CHANNELS = 8   # Number of block types
BLOCK_TYPES = ["normal", "bomb", "firework_v", "firework_h",
               "balloon", "disco", "rocket", "box"]

class ToonBlastEnv:
    def __init__(self):
        self.game = game.ToonBlastGame()
        self.action_space = game.BOARD_ROWS * game.BOARD_COLS
        self.observation_space = (COLOR_CHANNELS + TYPE_CHANNELS, 
                                 game.BOARD_ROWS, game.BOARD_COLS)

    def reset(self):
        self.game.reset()
        return self.get_state()

    def get_state(self):
        """Multi-channel state representation:
        - First 5 channels: one-hot color encoding
        - Next 8 channels: one-hot block type encoding
        """
        state = np.zeros(self.observation_space, dtype=np.float32)
        
        for r in range(game.BOARD_ROWS):
            for c in range(game.BOARD_COLS):
                block = self.game.board[r][c]
                if block is None:
                    continue
                
                # Color encoding
                if block.color in game.COLOR_LIST:
                    color_idx = game.COLOR_LIST.index(block.color)
                    state[color_idx, r, c] = 1
                
                # Type encoding
                if block.type in BLOCK_TYPES:
                    type_idx = BLOCK_TYPES.index(block.type)
                    state[COLOR_CHANNELS + type_idx, r, c] = 1
        return state

    def step(self, action):
        row = action // game.BOARD_COLS
        col = action % game.BOARD_COLS
        prev_score = self.game.score
        valid_move = True

        # Validate move before processing
        block = self.game.board[row][col]
        if block is None:
            valid_move = False
        elif block.type == "normal":
            group = self.game.get_connected_group(row, col, block.color)
            if len(group) < 2:
                valid_move = False

        if valid_move:
            self.game.process_move(row, col)
            while self.game.anim_state != "idle":
                self.game.update_animations()
            reward = self.game.score - prev_score
        else:
            reward = -5  # Penalize invalid moves

        done = self.game.is_game_over()
        return self.get_state(), reward, done, {}

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), \
               np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_net = QNetwork(env.observation_space, env.action_space).to(self.device)
        self.target_net = QNetwork(env.observation_space, env.action_space).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer(100000)
        self.batch_size = 64
        self.gamma = 0.99
        
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 2000
        self.epsilon = self.epsilon_start
        
        self.target_update_freq = 50
        self.train_start = 2000

    def get_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.env.action_space-1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state).cpu().numpy().squeeze()
        
        # Action masking for valid moves
        valid_actions = []
        for action in range(self.env.action_space):
            row = action // game.BOARD_COLS
            col = action % game.BOARD_COLS
            block = self.env.game.board[row][col]
            if block is None:
                continue
            if block.type == "normal":
                group = self.env.game.get_connected_group(row, col, block.color)
                if len(group) >= 2:
                    valid_actions.append(action)
            else:
                valid_actions.append(action)
        
        if valid_actions:
            return valid_actions[np.argmax(q_values[valid_actions])]
        return random.randint(0, self.env.action_space-1)

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                      np.exp(-1. * episode / self.epsilon_decay)

    def train(self, num_episodes):
        stats = {'rewards': [], 'epsilons': []}
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(self.buffer) >= self.train_start:
                    self._optimize_model()
            
            # Update target network
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            
            # Update exploration rate
            self.update_epsilon(episode)
            
            stats['rewards'].append(total_reward)
            stats['epsilons'].append(self.epsilon)
            
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Reward: {total_reward}, "
                  f"Epsilon: {self.epsilon:.3f}, "
                  f"Buffer: {len(self.buffer)}")
        
        return stats

    def _optimize_model(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            next_q[dones] = 0.0
            target_q = rewards + self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def demonstrate(model_path="best_model.pth"):
    env = ToonBlastEnv()
    agent = DQNAgent(env)
    agent.q_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.q_net.eval()
    
    state = env.reset()
    done = False
    total_reward = 0
    
    print("\n--- Demonstration Run ---")
    while not done:
        action = agent.get_action(state, eval_mode=True)
        row = action // game.BOARD_COLS
        col = action % game.BOARD_COLS
        block = env.game.board[row][col]
        
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        print(f"Action: ({row}, {col}) | "
              f"Type: {block.type if block else 'Empty'} | "
              f"Reward: {reward} | "
              f"Total: {total_reward}")
    
    print(f"\nFinal Score: {env.game.score}")
    print(f"Target Score: {env.game.target_score}")
    print(f"Moves Remaining: {env.game.moves_remaining}")

if __name__ == "__main__":
    # Training
    env = ToonBlastEnv()
    agent = DQNAgent(env)
    stats = agent.train(num_episodes=1000)
    
    # Save trained model
    torch.save(agent.q_net.state_dict(), "best_model.pth")
    
    # Demonstration
    demonstrate()