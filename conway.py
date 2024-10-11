import numpy as np
import time
from lib import fft_convolve2d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
plt.ion()

def conway(state, k=None):
    """
    Conway's game of life state transition
    """

    # set up kernel if not given
    if k is None:
        m, n = state.shape
        k = np.zeros((m, n))
        k[m//2-1 : m//2+2, n//2-1 : n//2+2] = np.array([[1,1,1],[1,0,1],[1,1,1]])

    # computes sums around each pixel
    b = fft_convolve2d(state, k).round()
    c = np.zeros(b.shape)

    c[(b == 2) & (state == 1)] = 1
    c[(b == 3) & (state == 1)] = 1
    c[(b == 3) & (state == 0)] = 1

    # return new state
    return c

if __name__ == "__main__":
    # set up board
    m, n = 100, 100
    A = np.random.random((m, n)).round()

    # set up the figure and animation
    fig, ax = plt.subplots()
    img_plot = ax.imshow(A, interpolation="nearest", cmap=plt.cm.gray)
    
    def update(frame):
        global A
        A = conway(A)
        img_plot.set_array(A)
        return [img_plot]
    
    # create the animation
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    
    # save the animation as a gif
    writer = PillowWriter(fps=25)
    anim.save("conway_game_of_life.gif", writer=writer)
    
    print("GIF saved as 'conway_game_of_life.gif'")
    
    plt.show()
