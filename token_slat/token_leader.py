# Fast-TRELLIS SLaT token-carving step scheduler.
#
# The carving path reads two pieces of state from the leader:
# `full_sampling_steps` (the warm-up step count below which no token is carved)
# and `current_step` (advanced once per Euler step). The skip budget per carved
# step is computed by the sampler as int(carving_ratio * N).
class TokenLeader:
    def __init__(self):
        self.num_steps = 0
        self.resolution = 16
        self.full_sampling_steps = 6
        self.current_step = 0
        self.total_tokens = 0

    def set_parameters(self, args):
        self.num_steps = args.effective_steps
        self.resolution = args.resolution

        if args.use_token:
            self.full_sampling_steps = 2

        self.current_step = 0

    def increase_step(self):
        self.current_step += 1
