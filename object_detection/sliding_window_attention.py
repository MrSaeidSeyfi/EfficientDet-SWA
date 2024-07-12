import numpy as np

class SlidingWindowAttention:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def generate_windows(self, image):
        windows = []
        for y in range(0, image.shape[0] - self.window_size[1] + 1, self.step_size[1]):
            for x in range(0, image.shape[1] - self.window_size[0] + 1, self.step_size[0]):
                window = image[y:y+self.window_size[1], x:x+self.window_size[0]]
                windows.append((window, x, y))
        return windows
