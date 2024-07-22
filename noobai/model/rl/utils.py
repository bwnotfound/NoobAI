import numpy as np

class GAE:
    '''
    Recommand gamma in [0.96~0.995] , lambda in [0.94~0.99]. 
    For gamma/lambda, you can try 0.99/0.98 or 0.99/0.96 or 0.98/0.96.
    Paper: https://arxiv.org/abs/1506.02438
    '''
    def __init__(self, gamma, gae_lambda, batched=False):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batched = batched
    
    def __call__(self, values, rewards):
        values, rewards = np.array(values), np.array(rewards)
        if self.batched:
            assert len(values.shape) == 2
            values, rewards = values.transpose(0, 1), rewards.transpose(0, 1)
        length = values.shape[0]
        advantages = []
        delta = rewards[-1]
        last_advantage = 0
        for i in reversed(range(length)):
            if i < length - 1:
                delta = rewards[i] + self.gamma * values[i + 1] - values[i]
            advantages.append(delta + self.gamma * self.gae_lambda * last_advantage)
        
        advantages = np.array(list(reversed(advantages)))
        returns = []
        for i in range(length):
            returns.append(advantages[i] + values[i])
        return advantages, np.array(returns)