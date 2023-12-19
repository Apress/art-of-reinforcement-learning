import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)


if __name__ == '__main__':
    x = np.linspace(-10, 10)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))

    for ax in [ax1, ax2, ax3, ax4]:
        # Move the left and bottom spines to x = 0 and y = 0, respectively.
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        # Hide the top and right spines.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
        # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
        # respectively) and the other one (1) is an axes coordinate (i.e., at the very
        # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
        # actually spills out of the axes.
        ax.plot(1, 0, '>k', transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, '^k', transform=ax.get_xaxis_transform(), clip_on=False)

    ax1.plot(x, sigmoid(x))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax2.plot(x, tanh(x))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax3.plot(x, relu(x))
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(5))

    ax4.plot(x, leaky_relu(x))
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(5))

    # labeling subplots
    trans = transforms.ScaledTranslation(0 / 72, 0 / 72, fig.dpi_scale_trans)
    for label, ax in [
        ('Sigmoid', ax1),
        ('Tanh', ax2),
        ('ReLU', ax3),
        ('LeakyReLU', ax4),
    ]:
        ax.text(
            0.0,
            1.0,
            label,
            transform=ax.transAxes + trans,
            fontsize='16',
            va='bottom',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0),
        )

    plt.tight_layout()

    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    plt.show()
