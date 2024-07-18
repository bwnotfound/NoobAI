class BaseReplyBuffer:
    def __init__(self):
        pass
    
    def sample(self, batch_size):
        '''
        can be sampled if real_size < batch_size
        '''
        raise NotImplementedError

    def can_sample(self, batch_size):
        raise NotImplementedError

    def add(self, example):
        '''
        should be a tuple or list of one step result.
        '''
        raise NotImplementedError