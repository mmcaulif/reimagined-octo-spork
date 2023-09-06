import gymnasium as gym
import reverb
import logging
import numpy as np
import tensorflow as tf


"""
Reverb tutorial: https://github.com/deepmind/reverb/blob/master/examples/demo.ipynb
"""


class SarlRunner():
    def __init__(self) -> None:
        self.env = gym.make('CartPole-v1')
        self.server = reverb.Server(tables=[
            reverb.Table(
                name='uniform_erb',
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=100_000,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                # signature = None, # need to create signature/adder for easier processing
                )
            ],
        )

        self.client = reverb.Client(f'localhost:{self.server.port}')
        logging.info(self.client.server_info())
        pass

    def run(self):
        with self.client.trajectory_writer(num_keep_alive_refs=1) as writer:
            o_t = self.env.reset()
            for _ in range(5):
                a_t = self.env.action_space.sample()
                o_tp1, r_t, d_t, _, i = self.env.step(a_t)
                # print(o_t, a_t, o_tp1, r_t, d_t)

                self.client.insert([o_t, a_t, o_tp1, r_t, d_t], priorities={'uniform_erb': 1.0})
                
                o_t = o_tp1
                if d_t:
                    o_t, _ = self.env.reset()
                    
        batch = self.client.sample('uniform_erb', num_samples=1)
        print(list(batch)[0][0].data)

runner = SarlRunner()
runner.run()