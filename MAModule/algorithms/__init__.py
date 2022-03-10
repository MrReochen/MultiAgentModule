class Algorithm:
    def prep_rollout(self):
        raise NotImplementedError
    
    def prep_training(self):
        raise NotImplementedError

    def train(self, buffer, progress):
        raise NotImplementedError