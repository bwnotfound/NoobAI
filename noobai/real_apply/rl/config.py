from dataclasses import dataclass

@dataclass
class BaseConfig:
    env_name: str = "CartPole-v1"  # 环境名字
    seed : int= 1437  # 随机种子
    max_steps: int = None  # 每个回合的最大步数
    device: str = "cuda"  # device to use

    train_eps: int = 10000000  # 训练的回合数，按照batch_size算
    eval_freq: int = 5000
    eval_eps: int = 10  # 评估的回合数

    batch_size: int = 256
    mini_batch_size: int = 256
    env_num_workers: int = 128
    

    # random_ratio = 0.1  # 随机动作的概率
    # gamma = 0.99  # 折扣因子
    # lamda = 0.98  # GAE参数
    # k_epochs = 5  # 更新策略网络的次数
    
    # eps_clip = 0.15  # epsilon-clip
    # entropy_coef = 0.01  # entropy的系数
    
    lr: float = 3e-4
    grad_clip_max: float = None

    num_actions: int = None
    num_states: int = None
    max_action: float = None

@dataclass
class ReplayConfig:
    capacity: int = int(1e6)

@dataclass
class ACConfig(BaseConfig):
    actor_hidden_dim: int = 128  # actor网络的隐藏层维度
    critic_hidden_dim: int = 128  # critic网络的隐藏层维度
    actor_lr: float = 3e-4  # actor网络的学习率
    critic_lr: float = 3e-4  # critic网络的学习率
    
    random_ratio: float = 0.1  # 随机动作的概率
    gamma: float = 0.99  # 折扣因子
    lamda: float = 0.98  # GAE参数
    
@dataclass
class TD3Config(BaseConfig, ReplayConfig):
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    discount: float = 0.99
    tau: float = 0.005
    expl_noise: float = 0.1
    
    warmup_steps: int = 25000