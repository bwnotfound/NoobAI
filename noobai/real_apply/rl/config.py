from dataclasses import dataclass

@dataclass
class BaseConfig:
    env_name = "CartPole-v1"  # 环境名字
    seed = 1437  # 随机种子
    device = "cuda"  # device to use

    train_eps = 10000000  # 训练的回合数，按照batch_size算
    max_steps = None  # 每个回合的最大步数
    eval_freq = 100  # 评估的回合数
    batch_size = 2**8
    mini_batch_size = 2**8
    env_num_workers = 128

    # random_ratio = 0.1  # 随机动作的概率
    # gamma = 0.99  # 折扣因子
    # lamda = 0.98  # GAE参数
    # k_epochs = 5  # 更新策略网络的次数
    
    # eps_clip = 0.15  # epsilon-clip
    # entropy_coef = 0.01  # entropy的系数
    
    grad_clip_max = None

    num_actions = None
    num_states = None
    max_action = None
    
@dataclass
class ACConfig(BaseConfig):
    actor_hidden_dim = 128  # actor网络的隐藏层维度
    critic_hidden_dim = 128  # critic网络的隐藏层维度
    actor_lr = 3e-4  # actor网络的学习率
    critic_lr = 3e-4  # critic网络的学习率
    
    random_ratio = 0.1  # 随机动作的概率
    gamma = 0.99  # 折扣因子
    lamda = 0.98  # GAE参数
    
@dataclass
class TD3Config(ACConfig):
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    discount = 0.99
    tau = 0.005
    