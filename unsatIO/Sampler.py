from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Sampler(ABC):

    def __init__(self, array):
        self.array = array

    @property
    def shape(self):
        return self.array.shape

    @abstractmethod
    def sample(self):
        """Return the sample"""
        pass

    @abstractmethod
    def plot(self):
        """Plot the sampler's shape on top of the array
        
        This is useful for visually checking that the sample is correct.
        """
        pass

    @abstractmethod
    def is_out(self):
        """Return True if and only if at least one sampling
        point is out of boundaries
        """
        pass

class RectangularSampler(Sampler):

    def __init__(self, array2D, loc, size):
        super().__init__(array2D)
        self.loc = loc
        self.size = size

    @property
    def horizontal_bounds(self):
        return [self.loc[1], self.loc[1] + self.size[1]]

    @property
    def vertical_bounds(self):
        return [self.loc[0], self.loc[0] + self.size[0]]

    def sample(self):
        return self.array[self.loc[0]:self.loc[0]+self.size[0], 
                          self.loc[1]:self.loc[1]+self.size[1]]
    
    def plot(self, linewidth=1, edgecolor='r', *args, **kwargs):
        _, ax = plt.subplots()
        ax.imshow(self.array, *args, **kwargs)
        # Notice the inversion of indices below.
        # This is due to to the notation:
        # [0] = row = vertical,
        # [1] = col = horizontal
        rect = patches.Rectangle((self.loc[1], self.loc[0]), width=self.size[1], height=self.size[0], linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)

    def is_out(self):
        return max(self.horizontal_bounds) > self.shape[1] or \
               max(self.vertical_bounds) > self.shape[0] or \
               min(self.horizontal_bounds) < 0 or \
               min(self.vertical_bounds) < 0

class ParallelepipedalSampler(Sampler):

    def sample(self):
        pass

    def plot(self):
        pass

    def is_out(self):
        pass
