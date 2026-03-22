import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.distributions import Normal
from collections import deque
import time


# ==========================================
# 1. 训练配置类 (Config)
# ==========================================
class PPOConfig:
    def __init__(self):
        # 基础训练参数
        self.total_steps = 1000000  # 总训练步数
        self.batch_size = 8192  # 每次收集多少步数据进行一次更新
        self.mini_batch_size = 1024  # 网络更新时的 Mini-batch
        self.n_epochs = 15  # 每次更新迭代的 Epoch 数

        # PPO 核心超参数
        self.lr = 2e-4  # 初始学习率
        self.gamma = 0.995  # 折扣因子
        self.gae_lambda = 0.95  # GAE 优势估计平滑参数
        self.clip_epsilon = 0.3  # PPO 裁剪范围
        self.vloss_coef = 0.5  # 价值损失系数
        self.ent_coef_start = 0.01  # 初始探索熵系数
        self.ent_coef_end = 0.00005  # 最终探索熵系数 (线性衰减)
        self.max_grad_norm = 0.5  # 梯度裁剪防爆炸

        # 硬件与环境维度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_stocks = 227
        self.window_size = 20
        self.n_features = 22
        self.hidden_dim = 256  # ResNet 隐藏层维度

        # 存储与日志
        self.save_dir = "./ppo_results"
        os.makedirs(self.save_dir, exist_ok=True)


# ==========================================
# 2. 神经网络类 (Network)
# ==========================================
class ResNormBlock(nn.Module):
    """基于层归一化的残差块，完美复现论文设计"""

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 残差连接：x + ReLU(LayerNorm(Linear(x)))
        return x + self.relu(self.ln(self.fc(x)))


class FinFeatureExtractor(nn.Module):
    """
    终极版特征提取器：1D-CNN (扩容版) + ResNet (全局资金调配)
    """

    def __init__(self, config):
        super().__init__()
        self.n_stocks = config.n_stocks
        self.window_size = config.window_size  # 20
        self.n_features = config.n_features  # 22

        # 1. 扩容后的 1D-CNN (提升特征抓取能力)
        self.stock_cnn = nn.Sequential(
            # Conv1: 通道数 22 -> 32
            nn.Conv1d(in_channels=self.n_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 长度 20 -> 10

            # Conv2: 通道数 32 -> 64
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 长度 10 -> 5

            nn.Flatten(),
            # 展平后维度 64 * 5 = 320
            # Linear: 压缩到 16 维
            nn.Linear(320, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )

        # 压缩后维度：227 * 16 = 3632 维，拼接 227 维当前持仓权重 = 3859 维
        combined_dim = self.n_stocks * 16 + self.n_stocks

        # 2. 全局特征提取保持不变
        self.global_extractor = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            ResNormBlock(config.hidden_dim),
            ResNormBlock(config.hidden_dim)
        )

    def forward(self, market_data, current_weights):
        batch_size = market_data.size(0)

        # market_data 原始 shape: (Batch, 227, 20, 22)
        # Conv1d 需要的 shape: (Batch*227, Channels, Length) -> (B*227, 22, 20)

        # 将 Batch 和 Stocks 维度合并
        x = market_data.view(-1, self.window_size, self.n_features)
        # 交换时间和特征的维度，适配 PyTorch 的 Conv1d
        x = x.transpose(1, 2)

        # 提取时序特征 -> 输出 shape: (Batch * 227, 16)
        compressed_stocks = self.stock_cnn(x)

        # 展平为全局市场特征 -> 输出 shape: (Batch, 227 * 16)
        global_market_feat = compressed_stocks.view(batch_size, -1)

        # 拼接持仓权重 (Batch, 3859)
        combined_feat = torch.cat([global_market_feat, current_weights], dim=1)

        # 提取最终特征 (Batch, 256)
        return self.global_extractor(combined_feat)


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.extractor = FinFeatureExtractor(config)

        # Actor: 输出连续动作的高斯分布均值 (mu)
        self.actor_mu = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.n_stocks),
            nn.Tanh()  # 映射到 [-1, 1]
        )
        # Actor: 动作的标准差 (log_std)，使用可学习的参数，不依赖 state
        self.actor_logstd = nn.Parameter(torch.zeros(1, config.n_stocks))

        # Critic: 输出当前状态的价值评估
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, market_data, current_weights):
        feat = self.extractor(market_data, current_weights)
        mu = self.actor_mu(feat)
        value = self.critic(feat)

        std = self.actor_logstd.expand_as(mu).exp()
        dist = Normal(mu, std)
        return dist, value


# ==========================================
# 3. 智能体类 (Agent & Buffer)
# ==========================================
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.market_datas = []
        self.weights = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, market, weight, action, logprob, reward, value, done):
        self.market_datas.append(market)
        self.weights.append(weight)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)


class PPOAgent:
    def __init__(self, config):
        self.cfg = config
        self.net = ActorCritic(config).to(config.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr, eps=1e-5)

        # 学习率调度器：线性衰减到 0
        total_updates = config.total_steps // config.batch_size
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01,
                                                     total_iters=total_updates)

        self.buffer = RolloutBuffer()
        self.current_step = 0

    def select_action(self, market_data_np, weight_np):
        market_t = torch.FloatTensor(market_data_np).unsqueeze(0).to(self.cfg.device)
        weight_t = torch.FloatTensor(weight_np).unsqueeze(0).to(self.cfg.device)

        with torch.no_grad():
            dist, value = self.net(market_t, weight_t)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)

        return action.cpu().numpy()[0], logprob.cpu().item(), value.cpu().item()

    def compute_gae(self, next_value, next_done):
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.cfg.device)
        values = torch.tensor(self.buffer.values, dtype=torch.float32).to(self.cfg.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32).to(self.cfg.device)

        advantages = torch.zeros_like(rewards).to(self.cfg.device)
        last_gae_lam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - next_done
                next_v = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_v = values[t + 1]

            delta = rewards[t] + self.cfg.gamma * next_v * next_non_terminal - values[t]
            advantages[
                t] = last_gae_lam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + values
        return advantages, returns

    def update(self, next_state_market, next_state_weight, next_done):
        # 获取当前进度，计算探索率(熵系数)的线性衰减
        progress = min(1.0, self.current_step / self.cfg.total_steps)
        current_ent_coef = self.cfg.ent_coef_start - progress * (self.cfg.ent_coef_start - self.cfg.ent_coef_end)

        # 获取下一个状态的价值用于 GAE 计算
        market_t = torch.FloatTensor(next_state_market).unsqueeze(0).to(self.cfg.device)
        weight_t = torch.FloatTensor(next_state_weight).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            _, next_value = self.net(market_t, weight_t)
            next_value = next_value.item()

        advantages, returns = self.compute_gae(next_value, next_done)

        # 转化 buffer 数据为 Tensor
        b_markets = torch.FloatTensor(np.array(self.buffer.market_datas)).to(self.cfg.device)
        b_weights = torch.FloatTensor(np.array(self.buffer.weights)).to(self.cfg.device)
        b_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.cfg.device)
        b_old_logprobs = torch.FloatTensor(self.buffer.logprobs).to(self.cfg.device)

        # 优势标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 记录用于日志的 Loss
        clip_fracs, approx_kls = [], []
        policy_losses, value_losses, entropy_losses = [], [], []

        dataset = torch.utils.data.TensorDataset(b_markets, b_weights, b_actions, b_old_logprobs, advantages, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg.mini_batch_size, shuffle=True)

        for _ in range(self.cfg.n_epochs):
            for batch in loader:
                mb_markets, mb_weights, mb_actions, mb_old_logprobs, mb_advs, mb_returns = batch

                dist, new_values = self.net(mb_markets, mb_weights)
                new_logprobs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    approx_kls.append(approx_kl)
                    clip_fracs.append(((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean().item())

                # PPO Clipped Loss
                pg_loss1 = mb_advs * ratio
                pg_loss2 = mb_advs * torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon)
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                # Value Loss (MSE)
                value_loss = nn.MSELoss()(new_values.squeeze(-1), mb_returns)

                # Total Loss
                loss = policy_loss + self.cfg.vloss_coef * value_loss - current_ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        self.scheduler.step()
        self.buffer.clear()

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "ent_coef": current_ent_coef,
            "kl": np.mean(approx_kls),
            "lr": self.scheduler.get_last_lr()[0]
        }


# ==========================================
# 4. 数据保存与绘图函数
# ==========================================
def save_and_plot_results(log_data, cfg):
    print(f"\n>>> 训练结束，正在保存数据与模型至: {cfg.save_dir}")

    # 保存 CSV
    df = pd.DataFrame(log_data)
    csv_path = os.path.join(cfg.save_dir, "training_log.csv")
    df.to_csv(csv_path, index=False)

    # 采用滑动窗口平滑数据
    window_size = max(5, len(df) // 50)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 图 1: Episode Reward
    axes[0, 0].plot(df['step'], df['ep_reward'], alpha=0.3, color='blue')
    axes[0, 0].plot(df['step'], df['ep_reward'].rolling(window_size).mean(), color='blue', linewidth=2)
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].set_xlabel('Total Steps')

    # 图 2: Policy Loss
    axes[0, 1].plot(df['step'], df['policy_loss'].rolling(window_size).mean(), color='red', linewidth=2)
    axes[0, 1].set_title('Policy Loss (Smoothed)')
    axes[0, 1].set_xlabel('Total Steps')

    # 图 3: Value Loss
    axes[1, 0].plot(df['step'], df['value_loss'].rolling(window_size).mean(), color='green', linewidth=2)
    axes[1, 0].set_title('Value Loss (Smoothed)')
    axes[1, 0].set_xlabel('Total Steps')

    # 图 4: Entropy
    axes[1, 1].plot(df['step'], df['entropy'].rolling(window_size).mean(), color='purple', linewidth=2)
    axes[1, 1].set_title('Policy Entropy (Smoothed)')
    axes[1, 1].set_xlabel('Total Steps')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "training_curves.png"), dpi=300)
    print("✓ 数据和图表已保存！")


# ==========================================
# 5. 主训练函数 (Main Training Loop)
# ==========================================
def train_ppo(data_path):
    # 导入并初始化环境 (确保 env.py 存在并配置正确)
    from env import FinDataProcessor, PortfolioEnv
    dp = FinDataProcessor(data_path=data_path, mode='train')

    # 注意：请确保 env.py 里的 reward_scaling 已经调小（如 2.0），防极值问题
    env = PortfolioEnv(data_processor=dp)

    cfg = PPOConfig()
    agent = PPOAgent(cfg)

    state, _ = env.reset()
    market_state = state['market_data']
    weight_state = state['current_weights']

    log_data = []
    ep_reward = 0
    ep_len = 0
    ep_count = 0

    start_time = time.time()
    print(f"\n🚀 开始 PPO 训练 - 目标步数: {cfg.total_steps} (使用设备: {cfg.device})")

    for step in range(1, cfg.total_steps + 1):
        agent.current_step = step

        # 1. 采集动作
        action, logprob, value = agent.select_action(market_state, weight_state)

        # 2. 与环境交互
        next_state, reward, done, _, info = env.step(action)
        next_market = next_state['market_data']
        next_weight = next_state['current_weights']

        # 3. 存入 Buffer
        agent.buffer.store(market_state, weight_state, action, logprob, reward, value, done)

        ep_reward += reward
        ep_len += 1

        market_state = next_market
        weight_state = next_weight

        if done:
            state, _ = env.reset()
            market_state = state['market_data']
            weight_state = state['current_weights']
            ep_count += 1
            # 记录用于日志
            last_ep_reward = ep_reward
            ep_reward = 0
            ep_len = 0

        # 4. 触发网络更新
        if step % cfg.batch_size == 0:
            update_info = agent.update(market_state, weight_state, done)

            # 记录日志数据
            log_record = {
                'step': step,
                'ep_reward': last_ep_reward if 'last_ep_reward' in locals() else 0,
                'policy_loss': update_info['policy_loss'],
                'value_loss': update_info['value_loss'],
                'entropy': update_info['entropy'],
                'ent_coef': update_info['ent_coef'],
                'kl': update_info['kl'],
                'lr': update_info['lr']
            }
            log_data.append(log_record)

            # 打印调试信息
            elapsed = time.time() - start_time
            fps = step / elapsed
            print(f"[{step}/{cfg.total_steps}] "
                  f"Ep: {ep_count} | "
                  f"Reward: {log_record['ep_reward']:.2f} | "
                  f"V-Loss: {update_info['value_loss']:.3f} | "
                  f"Ent: {update_info['entropy']:.2f} | "
                  f"LR: {update_info['lr']:.2e} | "
                  f"FPS: {fps:.0f}")

    # 5. 训练结束，保存结果和模型
    save_and_plot_results(log_data, cfg)

    model_path = os.path.join(cfg.save_dir, "ppo_actor_critic.pth")
    torch.save(agent.net.state_dict(), model_path)
    print(f"✓ 网络模型已保存至: {model_path}")


if __name__ == "__main__":
    # 请替换为你的张量数据集路径
    DATA_PATH = "tensor_data_raw.npz"
    train_ppo(DATA_PATH)