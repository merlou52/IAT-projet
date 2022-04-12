class EpsilonProfile:
    def __init__(self, initial=1., final=0., dec_episode=1., dec_step=0.):
        self.initial = initial
        self.final = final
        self.dec_episode = dec_episode
        self.dec_step = dec_step