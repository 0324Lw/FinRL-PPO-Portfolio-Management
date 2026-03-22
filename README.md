# 🚀 FinRL-PPO-Portfolio-Management

> 基于 PPO (Proximal Policy Optimization) 深度强化学习与 1D-CNN 架构的动态资产配置与量化投资组合管理算法。
> 本项目针对金融市场极低信噪比的特性，构建了一个高度自定义的多资产 `gymnasium` 强化学习环境。支持 227 只股票的高维连续权重解算，内置了带层归一化 (Layer Normalization) 的残差卷积特征提取器，以及完备的交易成本惩罚与防抖动机制。

## ✨ 核心特性 (Features)

* 🌍 **定制化高维金融环境**：构建了支持 227 只股票并行回测的交互环境，内置严谨的涨跌停板掩码拦截 (Masking) 与破产兜底保护机制，完美模拟 A 股/美股的真实交易摩擦。
* 🛸 **时序动量与全局配置**：摒弃传统 MLP，采用 **1D-CNN + ResNet 混合特征提取器**。1D-CNN 的滑动卷积核专门负责提取单只股票过去 20 天的局部量价形态与宏观特征，ResNet 模块负责统揽 227 只股票的全局资金分配。
* 🧠 **PPO 强化学习底座**：针对金融 POMDP 问题专门调优的 PPO 架构。采用极低探索率、大批次 (Batch Size = 8192) 与低学习率，有效过滤市场噪音；内置学习率线性衰减机制，保障策略在千万步级长线训练中的收敛稳定性。
* 📊 **防高频换手机制**：设计**交易死区 (Trading Deadzone)** 与 **温度系数缩放 (Temperature Scaling)**。通过底层拦截微小的权重噪音信号，解决连续动作空间导致的“高频无效换手与手续费磨损”死局。
* 📈 **复合风险调整奖励**：严格对齐金融量化逻辑，奖励函数融合了对数收益率、下行风险惩罚 (类似 Sortino Ratio) 以及三种交易摩擦成本（佣金、滑点、市场冲击），引导智能体学会“长线潜伏”而非“短线追涨”。

## 🧠 强化学习环境设计 (Environment Design)

本项目针对高维金融数据的维度灾难与噪音耦合难题进行了底层重构。为了让智能体学会在复杂震荡市中进行“风险规避”与“利润收割”，核心的状态空间、动作空间与奖励函数设计如下：

### 1. 状态空间 (Observation Space)

智能体的观测状态被设计为 **Dict 混合空间**，兼顾市场宏观时序与账户微观持仓：

| 键值 (Key) | 维度形状 (Shape) | 物理含义 (Description) / 数据说明 |
| :---: | :---: | :--- |
| `market_data` | **(227, 20, 22)** | **全局市场张量**：包含 227 只股票过去 20 天的 22 维高质量特征（开高低收、成交量、MACD、RSI 及宏观因子等）。数据已进行严格的横截面 Z-score 归一化。 |
| `current_weights` | **(227,)** | **当前持仓状态**：账户当前资金在 227 只股票上的分布权重。引导网络计算调仓成本与换手率。 |

*注：PPO 的 Actor-Critic 网络前端配备了自定义的 `FinFeatureExtractor`，将近 10 万维的原始字典空间高效压缩为 256 维的密集语义向量。*

### 2. 动作空间 (Action Space)

采用 **227 维连续动作空间 (Continuous 227D)**，底层解算基于 Softmax 归一化模型：

* **动作输出 ($a_t$)**：网络输出 **[-1.0, 1.0]** 的原始 Logits。
* **权重映射**：环境内部引入温度系数 $T$ 进行 Softmax 处理，自动保证所有股票的目标权重和为 1，且非负（满足不可做空约束）。配合 **0.1% 的交易死区阈值**，自动过滤无效噪音。

### 3. 奖励函数设计 (Reward Function)

本环境的奖励函数专为“稳健复利”而生，结合了收益、风险与真实交易摩擦：

$$r_{t} = \frac{R_{t}^{port}}{\sigma_{t}^{downside} + 0.01} - \lambda_{1}C^{commission} - \lambda_{2}C^{spread} - \lambda_{3}C^{impact}$$

* 🟢 **投资组合收益 ($R_{t}^{port}$)**：基于扣除交易成本后的净资产变化计算对数收益率。
* 🔴 **下行风险惩罚 ($\sigma_{t}^{downside}$)**：统计过去 20 天低于无风险利率的收益率方差，动态惩罚资产回撤，引导网络追求夏普/索提诺比率最大化。
* 🔴 **交易成本惩罚 ($C$)**：计算新旧权重差异带来的换手率，精确扣除固定佣金 ($C^{commission}$)、买卖价差滑点 ($C^{spread}$) 与市场冲击成本 ($C^{impact}$)。

**🏆 极端行情截断与防崩溃设计：**
为了防止金融市场偶发的极端暴涨暴跌破坏 Critic 网络的价值评估：
1. **硬截断限幅**：将放大系数缩放后的单步总奖励严格限幅 (Clip) 在 **[-2.0, 2.0]** 之间。
2. **破产与回撤兜底**：若单日账户净值回撤超过 5% 或触发破产保护，直接给予 **-2.0** 的极值惩罚，为价值网络提供最清晰的风险红线。

## 🛠️ 环境依赖 (Requirements)

本项目基于 Python 3.10 开发，推荐使用国内清华源极速配置环境：

核心依赖库：
* `torch` (PyTorch >= 2.0)
* `gymnasium`
* `numpy`
* `pandas`
* `matplotlib`

**快速安装指南 (Conda + Pip)：**
```bash
# 1. 创建并激活虚拟环境
conda create -n finrl_ppo python=3.10 -y
conda activate finrl_ppo

# 2. 安装科学计算与强化学习基石
conda install numpy pandas matplotlib gymnasium -c conda-forge -y

# 3. 安装 PyTorch (以 CUDA 12.4 为例，推荐使用 pip + 清华源保证速度)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
