class Policy:
    def save(self, dir_path, episode):
        raise NotImplementedError

    def get_actions(self):
        raise NotImplementedError
    
    def act(self):
        raise NotImplementedError