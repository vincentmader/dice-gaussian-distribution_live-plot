import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import optimize

from config import NR_OF_DICE, NR_OF_SIDES, BATCH_SIZE
from config import PLOT_FIT, X_PAD, BAR_WIDTH
from config import MPL_THEME
import functions
import plotting


def plot_fit(x, y):
    if NR_OF_DICE == 1:
        f, fname = functions.constant, "constant"
        p0 = [1 / NR_OF_DICE]
    elif NR_OF_DICE == 2:
        f, fname = functions.triangular, "triangular"
        p0 = [
            np.mean(MULTI_THROW_OUTCOMES),
            np.mean(MULTI_THROW_OUTCOMES) / (len(MULTI_THROW_OUTCOMES)/2)
        ]
    else:
        f, fname = functions.gaussian, "gaussian"
        p0 = [np.mean(MULTI_THROW_OUTCOMES), np.sqrt(NR_OF_DICE)]

    popt, _ = optimize.curve_fit(f, x, y, p0=p0)
    x = np.linspace(x[0]-X_PAD, x[-1]+X_PAD)
    y = f(x, *popt)
    plt.plot(x, y, color="red", linewidth=2, label=f"{fname} fit")
    plt.legend(loc="upper right", frameon=False)


def throw_dice():
    value = 0
    for _ in range(NR_OF_DICE):
        value += random.choice(SINGLE_THROW_OUTCOMES)
    return value


if __name__ == "__main__":
    SINGLE_THROW_OUTCOMES = range(1, NR_OF_SIDES+1)
    MULTI_THROW_OUTCOMES = range(NR_OF_DICE, NR_OF_DICE*NR_OF_SIDES+1)

    histogram = {outcome: 0 for outcome in MULTI_THROW_OUTCOMES}

    plt.style.use(MPL_THEME)
    plt.figure(figsize=(7, 5))

    frame_idx = 0
    while True:
        nr_of_throws = (frame_idx+1) * BATCH_SIZE
        for _ in range(BATCH_SIZE):
            value = throw_dice()
            histogram[value] += 1

        x = MULTI_THROW_OUTCOMES
        y = np.array(list(histogram.values())) / nr_of_throws
        plt.bar(x, y, width=BAR_WIDTH, label="dice throws")

        if PLOT_FIT:
            plot_fit(x, y)

        ax = plt.gca()
        plotting.hide_axis_frame(ax)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

        title = "sum distribution of {} dice with {} sides after {} rolls".format(
            NR_OF_DICE, NR_OF_SIDES, nr_of_throws
        )
        plt.title(title)
        plt.xlabel("sum value")

        plt.xlim(x[0]-X_PAD, x[-1]+X_PAD)
        plt.ylim(0, 1.2*max(y))

        plt.pause(0.00000001)
        plt.cla()
        frame_idx += 1

plt.show()
