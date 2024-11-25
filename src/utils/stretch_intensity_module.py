# Imports
import numpy as np
import matplotlib.pyplot as plt

CurveString = "Multi-Root"


## Main Function --------------------------------------------------------------


def make_one_curve(upsilon=0.35, xprime=None, upsilon_high=None, eq_num=4):
    """Given an exponent upsilon and an in_array array, return a single curve
    :param upsilon: scalar
    :param xprime: array of length N
    :return: norm curve of length N
    """
    if xprime is None:  # Defaults
        xprime = demo_make_xprime()
    upsilon_high = upsilon_high or upsilon

    # print("Using upsilon = {:0.5f}".format(upsilon))
    # print("Equation Number", eq_num)
    # And this makes the curve!
    if eq_num == 1:
        # f(x,a) = 1/2 + 2^(a-1) * s(x) * |x|^a
        # upsilon /= 10
        out_curve = 0.5 + (2.0 ** (upsilon - 1.0)) * np.sign(xprime - 0.5) * (
            np.abs(xprime - 0.5) ** upsilon
        )
    elif eq_num == 2:
        out_curve = 0.5 + (np.abs(2.0 * xprime - 1.0) ** (upsilon - 1.0)) * (xprime - 0.5)
    elif eq_num == 3:
        upsilon = -0.75
        out_curve = 0.5 + xprime**3 - upsilon * xprime
    else:
        lows = xprime < 0.5
        highs = xprime >= 0.5

        x_low = xprime[lows]
        x_high = xprime[highs]

        curve_low = ((2 * x_low) ** upsilon) / 2
        curve_high = -(((2 - 2 * x_high) ** upsilon_high) / 2 - 1)

        out_curve = np.zeros_like(xprime)
        out_curve[lows] = curve_low
        out_curve[highs] = curve_high

        # outcurve = np.fmax(curve,curve)

    return out_curve


## DEMO STUFF --------------------------------------------------------------
def demo_make_xprime(nx=10001, eq_num=1):
    """Makes the xprime array
    :param nx:
    """
    if eq_num == 1:
        xprime = np.linspace(0, 1, num=nx)
    elif eq_num == 2:
        xprime = np.linspace(-5, 5, num=nx)
    # elif eq_num == 4:
    #     xprime = np.linspace(-4,4, num=nx)
    else:
        xprime = np.linspace(0, 1, num=nx)
    return xprime


def demo_make_upsilon_array(nupsilon=6, range=2.0, eq_num=1):
    """Prepare the upsilon array"""
    print(f"Eqn #: {eq_num}")
    if eq_num == 1:
        # upsilon_array = np.linspace(1., range, num=nupsilon)
        upsilon_array = np.logspace(0, np.log10(range), num=nupsilon)
    elif eq_num == 2:
        upsilon_array = np.linspace(1.0, range, num=10)
    elif eq_num == 3:
        upsilon_array = np.linspace(-0.73, -0.76, num=10)
    elif eq_num == 4:
        upsilon_array = np.logspace(0, -1, num=nupsilon)
        upsilon_list = list(upsilon_array)
        if 1 not in upsilon_list:
            upsilon_list.append(1)
            upsilon_list.sort()
            upsilon_array = np.asarray(upsilon_list)
    else:
        upsilon_array = [1]
    # upsilon_array = np.linspace(1, 2, 5)
    print(upsilon_array)
    return upsilon_array


def demo_make_all_curves(upsilons_list=None, xprime=None, eq_num=None):
    """Make a set of curves at a number of upsilons"""
    # if upsilons_list is None:  # Defaults
    #     upsilons_list = demo_make_upsilon_array(eq_num=eq_num)

    # Make the Curves
    curve_list = []
    for alph in upsilons_list:
        curve = make_one_curve(upsilon=alph, xprime=xprime, eq_num=eq_num)
        curve_list.append(curve)
    return curve_list


def demo_plot_many_upsilons(
    curve_list=None,
    upsilons_list=None,
    xprime=None,
    axis=None,
    first0=True,
    eq_num=None,
    **kwargs,
):
    """Demonstrate the Effect of the upsilon Parameter"""

    if upsilons_list is None:  # Defaults
        upsilons_list = demo_make_upsilon_array(eq_num=eq_num) * 10
    if xprime is None:  # Defaults
        xprime = demo_make_xprime(eq_num=eq_num)
    if curve_list is None:  # Defaults
        curve_list = demo_make_all_curves(upsilons_list, xprime, eq_num=eq_num)

    wid = 3
    off = 0 if first0 else wid
    torun = axis if axis else plt
    lbl = r"$\gamma$ Curves"
    for curve, upsilon in zip(reversed(curve_list), reversed(upsilons_list)):
        # kwargs['ls'] = (off, (wid,wid)) if upsilon == 1 else kwargs['ls']
        kwargs["ls"] = "-" if upsilon == 1 else kwargs["ls"]
        torun.plot(xprime, curve, **kwargs)
        kwargs["label"] = None

    use_color = "darkred" if first0 else "navy"
    upsilon = 3.5 if first0 else 0.35
    # upsilon = 0.1 if first0 else 0.01
    use_curve1 = make_one_curve(upsilon=upsilon * 10, xprime=xprime, eq_num=eq_num)
    # plt.figure()
    torun.plot(xprime, use_curve1, ls="-", c=use_color, lw=4, zorder=10000)

    for curve, upsilon in zip(reversed(curve_list), reversed(upsilons_list)):
        # lls = ":" if upsilon==1 else ":"
        # print(upsilon)
        plt.plot(
            xprime,
            (xprime) ** upsilon,
            c="coral",
            ls=":",
            zorder=-10,
            label=lbl,
        )
        plt.plot(
            xprime, (xprime) ** (upsilon / 10), c="coral", ls=":", zorder=-10
        )
        lbl = None

    # plt.show(block=True)
    return xprime


def demo_plot_white_noise(upsilon=0.35):
    """Demonstrate the Algorithm on Random Input"""

    in_array = np.random.random_sample(size=50) - 0.5
    out_array = norm_stretch(in_array, upsilon=upsilon)

    plt.scatter(in_array, out_array)
    plt.title("Demonstration of the Algorithm on Random Input")

    plt.show()


def demo_plot_2D_method(in_array=None, upsilon=0.35, do_plot=True):
    if in_array is None:
        in_array = np.random.random_sample(size=(400, 400)) - 0.5
    out_array = norm_stretch(in_array, upsilon=upsilon)
    if do_plot:
        plot_2d(in_array, out_array)
    return out_array


def plot_2d(in_array=None, out_array=None, upsilon=None, do_plot=True):
    if in_array is None:
        in_array = np.random.random_sample(size=(100, 100)) - 0.5
    if out_array is None:
        out_array = norm_stretch(in_array, upsilon=upsilon)

    if do_plot:
        fig, (ax0, ax1) = plt.subplots(1, 2, sharex="all", sharey="all")

        ax0.set_title("Input")
        im0 = ax0.imshow(in_array + 0.5, origin="lower", vmin=0, vmax=1)
        plt.colorbar(im0, ax=ax0)

        ax1.set_title("Output")
        im1 = ax1.imshow(out_array, origin="lower", vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax1)

        fig.set_size_inches(8, 4)
        plt.suptitle("upsilon = {}".format(upsilon))
        plt.tight_layout()
        plt.show()
    return out_array


def norm_stretch(in_array, upsilon=0.35, upsilon_high=None, eq_num=4):
    """The only function anyone outside will ever see"""
    return make_one_curve(
        xprime=in_array, upsilon=upsilon, upsilon_high=upsilon_high, eq_num=eq_num
    )


def many_upsilons():
    fig, ax = plt.subplots(1, 1)
    first0 = False
    for eq_num, CurveString, ls, c in zip(
        [4, 2],
        ["Redistribution Curves", "Flat Middle"],
        ["-", "--"],
        ["dodgerblue", "tomato"],
    ):
        xprime = demo_plot_many_upsilons(
            axis=ax, ls=ls, c=c, first0=first0, label=CurveString, eq_num=eq_num
        )

        first0 = True
        break

    plt.ylim((0, 1))
    trim = 0.01
    lims = (0 - trim, 1 + trim)
    if eq_num == 1:
        plt.xlim(lims)
        plt.ylim(lims)

    elif eq_num == 2:
        plt.xlim((0.5, 1.5))

    elif eq_num == 4:
        plt.xlim(lims)
        plt.ylim(lims)
    else:
        plt.xlim((0, 1))

    lsed = "-"
    lsmid = "--"
    ced = "k"
    cmid = "k"

    plt.axhline(0, c=ced, ls=lsed)
    plt.axhline(0.5, c=cmid, ls=lsmid)
    plt.axhline(1, c=ced, ls=lsed)

    plt.axvline(0, c=ced, ls=lsed)
    plt.axvline(0.5, c=cmid, ls=lsmid)
    # plt.axvline(-0.5, c=ced,  ls=":")
    plt.axvline(1, c=ced, ls=lsed)

    plt.scatter(1, 1)
    plt.scatter(0.5, 0.5)
    plt.scatter(0, 0)

    # plt.title("Demonstration of Curve Shapes".format(CurveString))
    plt.xlabel("Input Intensity Value")
    plt.ylabel("Normalized Output Value")

    # plt.legend(ncol=2)
    # plt.show(block=True)

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    fig.set_size_inches((8, 8))
    plt.tight_layout()
    # plt.savefig(r"C:\Users\chgi7364\Dropbox\All School\CU\My Research\My Papers\Sunback\fig\redistribution_both.pdf")
    # plt.savefig(r"C:\Users\chgi7364\Dropbox\All School\CU\My Research\My Papers\Sunback\fig\redistribution_both.png")

    plt.show(block=True)


if __name__ == "__main__":
    # many_upsilons()
    demo_plot_many_upsilons()
    # many_upsilons()

    # pass
    # fig, ax = plt.subplots(1,1)
    # first0 = True
    # for eq_num, CurveString, ls, c in zip([1, 4], ["lev1p0", "New Roots Idea"], ["-", "-"], ["r", "b"]):
    #     demo_plot_many_upsilons(axis=ax, ls=ls, c=c, first0=first0, label=CurveString)
    #     first0 = False
    #
    #
    # ax.legend()
    # fig.set_size_inches((8,8))
    # plt.tight_layout()
    # # plt.savefig()
    # plt.show(block=True)
    # # demo_plot_white_noise()
    # # demo_plot_2D_method()

    # for upsilon in np.linspace(1,2,10):
    #     plot_2d(upsilon=upsilon)
#
#
# The stretching parameter is "upsilon".
#
# upsilon=1 is linear... upsilon>1 is stretched.   The largest values in the plot (like 7 or 8) are probably way too extreme.
#
#

# #original implimentation
# for i in range(nx):
#     for j in range(nupsilon):
#         xprime = x_input_array[i] - 0.5
#         y_output_array[i, j] = 0.5 + (2. ** (upsilon[j] - 1.)) * np.sign(xprime) * (np.abs(xprime) ** upsilon[j])
