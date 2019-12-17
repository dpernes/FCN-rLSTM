r"""
Adapted from the visdom-tutorial project from Noa Garcia:
https://github.com/noagarcia/visdom-tutorial

All credit goes to the original author.
"""

from visdom import Visdom
import numpy as np


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8097):
        self.vis = Visdom(port=port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel='Epochs',
                    ylabel=var_name)
                )
        else:
            self.vis.line(X=np.array([x]),
                          Y=np.array([y]),
                          env=self.env,
                          win=self.plots[var_name],
                          name=split_name,
                          update='append')


class VisdomDictPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8097):
        self.vis = Visdom(port=port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, keys_name, title_name, dct):
        keys = list(dct.keys())
        values = list(dct.values())
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.bar(
                X=np.array(values),
                env=self.env,
                opts=dict(
                    rownames=keys,
                    title=title_name,
                    xlabel=keys_name,
                    ylabel=var_name)
                )
        else:
            self.vis.bar(X=np.array(values),
                         env=self.env,
                         win=self.plots[var_name],
                         opts=dict(
                             rownames=keys,
                             title=title_name,
                             xlabel=keys_name,
                             ylabel=var_name)
                         )


class VisdomImgsPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8097):
        self.vis = Visdom(port=port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, images, labels):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.images(
                images,
                env=self.env,
                opts=dict(
                    title=var_name,
                    caption=labels)
            )
        else:
            self.vis.images(
                images,
                env=self.env,
                win=self.plots[var_name],
                opts=dict(
                    title=var_name,
                    caption=labels)
            )
