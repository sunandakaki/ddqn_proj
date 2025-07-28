import gymnasium as gym
import ale_py
import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import random
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
from collections import deque
from sumtree import SumTreeRM
from rank import RankBasedHeapRM
from uniform import UniformReplayBuffer
from ddqn import DDQN

matplotlib.use('Agg')
gym.register_envs(ale_py)   # Register ALE environments with Gymnasium

# Choose device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.env_id = "ALE/AirRaid-v5"
        self.env = gym.make(self.env_id, obs_type='grayscale',) #render_mode="human"
        self.n_actions = self.env.action_space.n
        self.policy_net = DDQN(self.n_actions).to(device)
        self.target_net = DDQN(self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.replay_memory_size = 1_000_000
        self.batch_size = 32
        

        self.eps_init = 1
        self.eps_decay = 0.9995
        self.eps_min = 0.05
        self.epsilon = self.eps_init

        self.sync_rate = 1000
        self.learning_rate = 0.0001
        self.discount_factor = 0.99
        self.alpha = 0.6
        self.beta_start = 0.4
        self.beta = self.beta_start
        self.beta_frames = 1000000
        self.train_freq = 4
        self.training_step = 0

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr = self.learning_rate)
        self.loss_fn = torch.nn.HuberLoss()

        self.transform = T.Compose([T.ToPILImage(), T.Resize((84,84)), T.ToTensor()])
        self.rewards = []
        self.losses = []
        self.td_errors = []
        os.makedirs("test1", exist_ok=True)
        self.graph_path = "test1/graph1.png"

        self.eval_interval = 1000
        self.eval_episodes = 5
        self.eval_frames   = []
        self.eval_means    = []
        self.eval_bests    = []


    def preprocess(self, frame):
        frame = self.transform(frame)
        return frame.squeeze(0)

    def stack_frames(self, frame_stack, new_frame):
        frame_stack.append(self.preprocess(new_frame))
        return np.stack(frame_stack, axis=0)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.policy_net(state).argmax().item()
        
    def evaluate(self, n_episodes=5):
        old_eps = self.epsilon
        self.epsilon = 0.0
        total = 0.0
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            frames = deque([self.preprocess(state)]*4, maxlen=4)
            done = False; ep_r = 0
            while not done:
                a = self.select_action(np.stack(frames,0))
                s2, r, term, trunc, _ = self.env.step(a)
                frames.append(self.preprocess(s2))
                ep_r += r
                done = term or trunc
            total += ep_r
        self.epsilon = old_eps
        return total / n_episodes


    def train(self, method, max_frames = 200_000):
        self.method = method
        if method == "proportional":
            self.memory = SumTreeRM(self.replay_memory_size, self.batch_size)
        elif method == "rank":
            self.memory = RankBasedHeapRM(self.replay_memory_size, self.batch_size)
        elif method == "uniform":
            self.memory = UniformReplayBuffer(self.replay_memory_size, self.batch_size)
        else:
            raise ValueError(f"Unknown replay method {method}")
        
        print(f"Training with {method} replay...")
        state, info = self.env.reset()
        frame_stack = deque([self.preprocess(state)] * 4, maxlen=4)
        stacked_state = np.stack(frame_stack, axis=0)
        episode_reward = 0

        for frame_num in range(1, max_frames+1):
            action = self.select_action(stacked_state)
            new_state, reward, terminated, truncated, info = self.env.step(action)
            reward = torch.tensor(reward, dtype = torch.float, device=device)

            new_frame = self.preprocess(new_state)
            frame_stack.append(new_frame)
            next_stacked_state = np.stack(frame_stack, axis=0)

            stacked_state = torch.Tensor(stacked_state)
            next_stacked_state = torch.Tensor(next_stacked_state)
            self.memory.push((stacked_state, action, next_stacked_state, reward, terminated))
            stacked_state = next_stacked_state
            episode_reward += reward.item()

            if frame_num % self.train_freq == 0 and self.memory.num_entries >= 1000:
                sample = self.memory.sample()
                if self.method in ["proportional", "rank"]:
                    batch, indices = sample
                else:
                    batch, indices = sample, None
                self.optimize(batch, indices)
            
            if frame_num % self.sync_rate == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            episode_over = terminated or truncated

            if episode_over:
                print(f"Frame {frame_num}: Episode reward = {episode_reward}")
                self.rewards.append(episode_reward)
                episode_reward = 0
                state, _ = self.env.reset()
                frame_stack = deque([self.preprocess(state)] * 4, maxlen=4)
                stacked_state = np.stack(frame_stack, axis=0)
                # self.save_graph()
                # check_best_reward = episode_reward>best_reward
            
            if frame_num % self.eval_interval == 0:
                avg = self.evaluate(self.eval_episodes)
                self.eval_frames.append(frame_num)
                self.eval_means .append(avg)
                best = max(self.eval_bests[-1], avg) if self.eval_bests else avg
                self.eval_bests .append(best)
                print(f"Frame {frame_num}: eval avg={avg:.1f}, best so far={best:.1f}")


            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

        self.env.close()


    def optimize(self, mini_batch, indices=None):
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminations = zip(*mini_batch)
        batch_states = torch.stack(batch_states).to(device)
        batch_next_states = torch.stack(batch_next_states).to(device)

        scalar_actions = [action for action in batch_actions]  # Extract scalar values
        batch_actions = torch.tensor(scalar_actions, dtype=torch.long, device=device)
        
        batch_rewards = torch.stack(batch_rewards)
        batch_terminations = torch.tensor(batch_terminations).float().to(device)

        current_q = self.policy_net(batch_states).gather(dim=1, index=batch_actions.unsqueeze(dim=1)).squeeze()

        with torch.no_grad():
            # target_q = batch_rewards + (1-batch_terminations)*self.discount_factor*target(batch_next_states).max(dim=1)[0] #Ddqn
            best_actions = self.policy_net(batch_next_states).argmax(dim=1)
            target_q = batch_rewards + (1-batch_terminations)*self.discount_factor*self.target_net(batch_next_states).gather(dim=1, index=best_actions.unsqueeze(dim=1)).squeeze()
        
        if self.method == "uniform":
            loss = F.smooth_l1_loss(current_q, target_q)
        else:
            td_error = torch.abs(current_q - target_q).detach()
            alpha = self.alpha
            noise = 1e-6

            if self.method == "proportional":
                # p_i = |td| + noise
                p_i = td_error.cpu().numpy() + noise
            else:  # rank-based
                sorted_idx = np.argsort(-td_error.cpu().numpy())    # rank transitions by descending td
                ranks = np.empty_like(sorted_idx)
                ranks[sorted_idx] = np.arange(1, len(sorted_idx)+1)
                p_i = 1.0 / ranks

            p_i_alpha = p_i**alpha
            P = p_i_alpha / p_i_alpha.sum()

            self.beta = min(1.0, self.beta_start + self.training_step * (1.0 - self.beta_start) / self.beta_frames)

            N = self.memory.num_entries
            w = (1.0 / (N * P))**self.beta
            w = w / w.max()
            w = torch.tensor(w, dtype=torch.float32, device=device)
            loss = (w * self.loss_fn(current_q, target_q)).mean()

            for idx, pi in zip(indices, p_i):
                self.memory.update(idx, float(pi))

            self.training_step += 1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())


    def save_graph(self, ):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Reward per Episode")
        plt.plot(self.rewards[-200:])
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.subplot(1, 2, 2)
        plt.title("Loss per Training Step")
        plt.plot(self.losses[-1000:])
        plt.xlabel("Training Step")
        plt.ylabel("Loss")

        plt.tight_layout()
        plt.savefig(self.graph_path)
        plt.close()


def run_and_collect(method, max_frames):
    agent = Agent()
    agent.train(method, max_frames=max_frames)
    return agent.eval_frames, agent.eval_means

def plot_comparison(results, out_path="figure4_methods.png"):
    plt.figure(figsize=(8,6))
    for method, (frames, means) in results.items():
        plt.plot(frames, means, label=method)
    plt.xlabel("Frames")
    plt.ylabel(f"Avg reward over {Agent().eval_episodes} eval eps")
    plt.title("Evaluation curves for different replay methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved comparison plot to {out_path}")

if __name__ == '__main__':
    methods = ["uniform","proportional","rank"]
    max_frames = 50_000 
    all_results = {}
    for m in methods:
        print(f"\n=== running {m} replay ===")
        frames, means = run_and_collect(m, max_frames)
        all_results[m] = (frames, means)

    plot_comparison(all_results)
