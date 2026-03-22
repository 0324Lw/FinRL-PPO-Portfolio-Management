import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class FinDataProcessor:
    """
    金融数据集处理引擎
    负责加载 .npz 张量数据，并按时间步提供观测状态和真实计算价格。
    """

    def __init__(self, data_path, mode='train'):
        self.data_path = data_path
        self.mode = mode
        self.tensor_data = None
        self.dates = None
        self.stock_codes = None
        self.feature_names = None

        # 维度信息
        self.n_samples = 0
        self.n_stocks = 0
        self.window_size = 0
        self.n_features = 0
        self.close_idx = 3  # 'close' 在特征列表中的索引

        self.load_data()

    def load_data(self):
        print(f"[{self.mode.upper()}] 正在加载张量数据: {self.data_path} ...")
        try:
            data = np.load(self.data_path, allow_pickle=True)

            if self.mode == 'train':
                self.tensor_data = data['train_tensor']
                self.dates = data['train_dates']
            elif self.mode == 'valid':
                self.tensor_data = data['valid_tensor']
                self.dates = data['valid_dates']
            elif self.mode == 'test':
                self.tensor_data = data['test_tensor']
                self.dates = data['test_dates']
            else:
                raise ValueError("mode 必须是 'train', 'valid' 或 'test'")

            self.stock_codes = data['stock_codes']
            self.feature_names = data['feature_names']

            # 获取维度: (samples, stocks, window, features)
            self.n_samples, self.n_stocks, self.window_size, self.n_features = self.tensor_data.shape

            # 确认 close 价格的索引位置 (通常是 3)
            feature_list = list(self.feature_names)
            if 'close' in feature_list:
                self.close_idx = feature_list.index('close')

            print(
                f"[{self.mode.upper()}] 加载成功! 样本数: {self.n_samples}, 股票数: {self.n_stocks}, 特征维度: {self.window_size}x{self.n_features}")

        except Exception as e:
            raise RuntimeError(f"数据加载失败，请检查路径。详细错误: {e}")

    def get_state(self, step_idx):
        """返回指定时间步的 3D 市场特征张量 (stocks, window, features)"""
        if step_idx >= self.n_samples:
            step_idx = self.n_samples - 1
        return self.tensor_data[step_idx].astype(np.float32)

    def get_close_prices(self, step_idx):
        """提取指定时间步最后一天的收盘价，用于计算真实收益率"""
        if step_idx >= self.n_samples:
            step_idx = self.n_samples - 1
        # 切片获取所有股票、时间窗口最后一天、close特征的值
        return self.tensor_data[step_idx, :, -1, self.close_idx].astype(np.float32)


class PortfolioEnv(gym.Env):
    """
    动态资产配置强化学习环境
    严格对齐论文中的复合奖励函数、交易成本及涨跌停逻辑约束。
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_processor: FinDataProcessor, config: dict = None):
        super(PortfolioEnv, self).__init__()

        self.dp = data_processor
        if config is None:
            config = {}

        # 环境超参数
        self.initial_capital = float(config.get('initial_capital', 100000.0))
        self.commission_rate = float(config.get('commission_rate', 0.0003))  # 佣金 0.03%
        self.tax_rate = float(config.get('tax_rate', 0.001))  # 印花税 (仅卖出) 0.1%
        self.risk_free_rate = float(config.get('risk_free_rate', 0.0001))  # 每日无风险利率近似
        self.reward_scaling = float(config.get('reward_scaling', 2.0))  # 奖励放大系数

        # 奖励惩罚系数 (对应论文中的 lambda 1, 2, 3)
        self.lambda_1 = float(config.get('lambda_1', 1.0))  # 佣金惩罚
        self.lambda_2 = float(config.get('lambda_2', 1.0))  # 滑差惩罚
        self.lambda_3 = float(config.get('lambda_3', 1.0))  # 冲击成本惩罚

        self.n_stocks = self.dp.n_stocks
        self.max_steps = self.dp.n_samples - 1

        # --- 动作空间 ---
        # 连续空间 [-1, 1], PPO 输出 logits，底层通过 softmax 转为实际权重
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_stocks,),
            dtype=np.float32
        )

        # --- 观测空间 (Dict) ---
        # 包含 3D 市场序列信息 和 1D 当前持仓权重
        self.observation_space = spaces.Dict({
            "market_data": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_stocks, self.dp.window_size, self.dp.n_features),
                dtype=np.float32
            ),
            "current_weights": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.n_stocks,),
                dtype=np.float32
            )
        })

        # 状态变量初始化
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.current_weights = np.zeros(self.n_stocks, dtype=np.float32)
        # 记录过去 T 天的收益率，用于计算下行风险 (T=20)
        self.history_returns = deque(maxlen=20)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio_value = self.initial_capital

        # 初始权重平均分配
        self.current_weights = np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        self.history_returns.clear()

        # 构建初始 State
        state = {
            "market_data": self.dp.get_state(self.current_step),
            "current_weights": self.current_weights.copy()
        }

        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "date": self.dp.dates[self.current_step]
        }

        return state, info

    def step(self, action):
        """执行动作，返回下一状态、奖励、完成状态及详细信息"""

        # ==========================================
        # 改进 1: 增大温度系数 (Temperature)
        # 将温度从 1.0 调大到 5.0。这会“软化”网络输出的极值，
        # 让初始状态下的资金分配更加均匀，防止 Agent 在探索期因为一点点噪音就全仓某只股票。
        # ==========================================
        temperature = 2.0
        exp_action = np.exp(action / temperature)
        target_weights = exp_action / np.sum(exp_action)

        # ==========================================
        # 改进 2: 引入“交易死区” (Trading Deadzone / Inertia)
        # 如果某只股票的目标权重与当前权重相差不到 1% (0.01)，
        # 则判定为网络的无意义震荡(噪音)，强行保持仓位不变。
        # 这将极大地砍掉无意义的微小换手率，拯救手续费。
        # ==========================================
        deadzone_threshold = 0.001
        weight_diffs = np.abs(target_weights - self.current_weights)
        in_deadzone = weight_diffs < deadzone_threshold

        # 触发死区的股票，目标权重回退为当前权重
        target_weights[in_deadzone] = self.current_weights[in_deadzone]

        # 重新归一化 (死区保护后，总和可能略微偏离 1.0)
        target_weights = target_weights / np.sum(target_weights)

        # 获取当天和下一天的价格
        current_prices = self.dp.get_close_prices(self.current_step)
        next_prices = self.dp.get_close_prices(self.current_step + 1)

        # 2. 涨跌停约束处理 (涨跌幅超 9.5% 限制交易)
        price_change_ratio = np.zeros_like(current_prices)
        valid_idx = current_prices > 0
        price_change_ratio[valid_idx] = (next_prices[valid_idx] - current_prices[valid_idx]) / current_prices[valid_idx]

        limit_up_idx = price_change_ratio >= 0.095
        limit_down_idx = price_change_ratio <= -0.095

        # 跌停：目标权重不能小于当前权重 (卖不出去)
        target_weights[limit_down_idx] = np.maximum(target_weights[limit_down_idx],
                                                    self.current_weights[limit_down_idx])
        # 涨停：目标权重不能大于当前权重 (买不进来)
        target_weights[limit_up_idx] = np.minimum(target_weights[limit_up_idx], self.current_weights[limit_up_idx])

        # 再次归一化
        target_weights = target_weights / np.sum(target_weights)

        # 3. 计算资产收益率向量
        asset_returns = np.zeros_like(current_prices)
        asset_returns[valid_idx] = (next_prices[valid_idx] - current_prices[valid_idx]) / current_prices[valid_idx]

        # 投资组合该步原始收益率
        portfolio_return = np.dot(target_weights, asset_returns)

        # 4. 计算交易成本组件
        delta_weights = target_weights - self.current_weights
        buy_weights = np.maximum(delta_weights, 0)
        sell_weights = np.maximum(-delta_weights, 0)

        turnover_buy = np.sum(buy_weights)
        turnover_sell = np.sum(sell_weights)

        # 佣金与印花税 (C_commission)
        c_commission = (turnover_buy * self.commission_rate +
                        turnover_sell * (self.commission_rate + self.tax_rate))

        # 滑点/买卖价差成本 (C_spread)
        c_spread = 0.0005 * np.sum(np.abs(delta_weights))

        # 市场冲击成本 (C_impact)
        c_impact = 0.001 * np.sum(delta_weights ** 2)

        total_cost_rate = c_commission + c_spread + c_impact

        # 5. 更新账户总价值并计算对数收益
        next_portfolio_value = self.portfolio_value * (1.0 + portfolio_return) * (1.0 - total_cost_rate)
        if next_portfolio_value <= 0:
            log_return = -1.0
            next_portfolio_value = 1.0
        else:
            log_return = np.log(next_portfolio_value / self.portfolio_value)

        self.history_returns.append(log_return)

        # 6. 计算下行风险 (Downside Deviation)
        downside_risk = 0.0
        if len(self.history_returns) > 0:
            returns_array = np.array(self.history_returns)
            downside_diffs = np.maximum(self.risk_free_rate - returns_array, 0)
            downside_risk = np.sqrt(np.mean(downside_diffs ** 2))

        # 7. 组装终极奖励函数
        raw_reward = (log_return / (downside_risk + 0.01)) \
                     - (self.lambda_1 * c_commission) \
                     - (self.lambda_2 * c_spread) \
                     - (self.lambda_3 * c_impact)

        # 放大与硬截断
        scaled_reward = raw_reward * self.reward_scaling

        if log_return < -0.05:
            scaled_reward = -2.0

        step_reward = float(np.clip(scaled_reward, -2.0, 2.0))

        # 8. 状态更新
        self.portfolio_value = next_portfolio_value
        self.current_weights = target_weights
        self.current_step += 1

        done = self.current_step >= self.max_steps
        truncated = False

        next_state = {
            "market_data": self.dp.get_state(self.current_step),
            "current_weights": self.current_weights.copy()
        }

        # 9. Info 字典
        info = {
            "step": self.current_step,
            "date": self.dp.dates[self.current_step] if self.current_step < len(self.dp.dates) else "END",
            "portfolio_value": self.portfolio_value,
            "log_return": log_return,
            "reward_components": {
                "raw_reward": raw_reward,
                "scaled_reward": scaled_reward,
                "final_clipped_reward": step_reward,
                "downside_risk": downside_risk,
                "c_commission": c_commission,
                "c_spread": c_spread,
                "c_impact": c_impact,
                "turnover": turnover_buy + turnover_sell
            }
        }

        return next_state, step_reward, done, truncated, info

    def render(self, mode='human'):
        print(
            f"Step: {self.current_step} | Value: {self.portfolio_value:.2f} | Date: {self.dp.dates[self.current_step]}")
