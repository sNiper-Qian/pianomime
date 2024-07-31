class LR_Scheduler():
    def __init__(self,
                 initial_lr: float,
                 decay_rate: float,):
        self.lr = initial_lr
        self.decay_rate = decay_rate 
    
    def lr_schedule(self, remaining_progress: float) -> float:
        """Linearly decay the learning rate to zero."""
        if remaining_progress == 0.0:
            # It means that the progress has been reset.
            self.remaining_progress = remaining_progress
            self.lr *= self.decay_rate
            print("Learning rate decayed to {}.".format(self.lr))
        return self.lr
        