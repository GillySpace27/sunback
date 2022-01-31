# Imports
import numpy as np
import matplotlib.pyplot as plt
use_equation_num = 4
CurveString = "Multi-Root"
first0 = True

## Main Function --------------------------------------------------------------

def make_one_curve(alpha=0.35, xprime=None):
    """Given an exponent alpha and an input array, return a single curve
    :param alpha: scalar
    :param xprime: array of length N
    :return: norm curve of length N
    """
    if xprime is None:  # Defaults
        xprime = demo_make_xprime()
        
    
    # print("Using alpha = {:0.5f}".format(alpha))

    # And this makes the curve!
    if use_equation_num == 1:
        # f(x,a) = 1/2 + 2^(a-1) * s(x) * |x|^a
        out_curve = 0.5 + (2. ** (alpha - 1.)) * np.sign(xprime-0.5) * (np.abs(xprime-0.5) ** alpha)
    elif use_equation_num == 2:
        out_curve = 0.5 + (np.abs(2.*xprime-1.)**(alpha-1.))*(xprime-0.5)
    elif use_equation_num == 3:
        alpha = -0.75
        out_curve = 0.5 + xprime**3 - alpha*xprime
    elif use_equation_num == 4:
        
        lows = xprime < 0.5
        highs = xprime > 0.5
        
        x_low = xprime[lows]
        x_high = xprime[highs]
        
        curve_low = ((2*x_low) ** alpha)/2
        curve_high = -(((2*(-x_high)+2) ** alpha)/2 - 1)
        
        out_curve = np.zeros_like(xprime)
        out_curve[lows] = curve_low
        out_curve[highs]= curve_high
       
        # outcurve = np.fmax(curve,curve)
        
    return out_curve

## DEMO STUFF --------------------------------------------------------------
def demo_make_xprime(nx=100000):
    """Makes the xprime array
    :param nx:
    """
    if use_equation_num == 1:
        xprime = np.linspace(0, 1, num=nx)
    elif use_equation_num == 2:
        xprime = np.linspace(-5,5, num=nx)
    # elif use_equation_num == 4:
    #     xprime = np.linspace(-4,4, num=nx)
    else:
        xprime = np.linspace(0, 1, num=nx)
    return xprime
        
def demo_make_alpha_array(nalpha=8, range=10.):
    """ Prepare the alpha array"""
    if use_equation_num == 1:
        alpha_array = np.linspace(1., range, num=nalpha)
    elif use_equation_num == 2:
        alpha_array = np.linspace(1., range, num=10)
    elif use_equation_num == 3:
        alpha_array = np.linspace(-0.73, -0.76, num=10)
    elif use_equation_num == 4:
        alpha_array = np.logspace(0,-1, 8)
        alpha_list = list(alpha_array)
        if 1 not in alpha_list:
            alpha_list.append(1)
            alpha_list.sort()
            alpha_array = np.asarray(alpha_list)
    else:
        alpha_array = [1]
    return alpha_array


def demo_make_all_curves(alphas_list=None, xprime=None):
    """ Make a set of curves at a number of alphas"""
    if alphas_list is None:  # Defaults
        alphas_list = demo_make_alpha_array()
    
    # Make the Curves
    curve_list = []
    for alph in alphas_list:
        curve = make_one_curve(alpha=alph, xprime=xprime)
        curve_list.append(curve)
    return curve_list


def demo_plot_many_alphas(curve_list=None, alphas_list=None, xprime=None, axis=None, first0 = True, **kwargs):
    """ Demonstrate the Effect of the Alpha Parameter """
    
    if alphas_list is None:  # Defaults
        alphas_list = demo_make_alpha_array()
    if curve_list is None:   # Defaults
        curve_list = demo_make_all_curves(alphas_list)
    if xprime is None:  # Defaults
        xprime = demo_make_xprime()


    for curve, alpha in zip(curve_list, alphas_list):
        if axis:
            torun = axis
        else:
            torun = plt
            
        torun.plot(xprime, curve, **kwargs)
        kwargs['label'] = None
        
    final = norm_stretch(xprime, alpha=0.35)
    
    if not first0:
        torun.plot(xprime, final, c='g', lw=4, label="Final Curve")
    
    plt.ylim((0,1))
    
    if use_equation_num==1:
        plt.xlim((-0.1,1.1))
        plt.ylim((-0.1,1.1))
        
    elif use_equation_num==2:
        plt.xlim((0.5,1.5))
        
    elif use_equation_num==4:
        plt.xlim((-0.1,1.1))
        plt.ylim((-0.1,1.1))
    else:
        plt.xlim((0,1))
    
    
    plt.axhline(0, ls=":")
    plt.axhline(0.5, ls=":")
    plt.axhline(1, ls=":")
    
    plt.axvline(0, ls=":")
    plt.axvline(0.5, ls=":")
    plt.axvline(-0.5, ls=":")
    plt.axvline(1, ls=":")
    
    plt.scatter(1,1)
    plt.scatter(0.5,0.5)
    plt.scatter(0,0)
    
    plt.title("Demonstration of Curve Shapes".format(CurveString))
    # plt.legend(ncol=2)
    # plt.show()

def demo_plot_white_noise(alpha=0.35):
    """ Demonstrate the Algorithm on Random Input """
    
    in_array = np.random.random_sample(size=50) - 0.5
    out_array = norm_stretch(in_array, alpha=alpha)
    
    plt.scatter(in_array,out_array)
    plt.title("Demonstration of the Algorithm on Random Input")
    
    plt.show()
    
    
def demo_plot_2D_method(in_array=None, alpha=0.35, do_plot=True):
    if in_array is None:
        in_array = np.random.random_sample(size=(400, 400)) - 0.5
    out_array = norm_stretch(in_array, alpha=alpha)
    if do_plot:
        plot_2d(in_array, out_array)
    return out_array


def plot_2d(in_array=None, out_array=None, alpha=None, do_plot=True):
    if in_array is None:
        in_array = np.random.random_sample(size=(100, 100)) - 0.5
    if out_array is None:
        out_array = norm_stretch(in_array, alpha=alpha)
        
    if do_plot:
        fig, (ax0, ax1) = plt.subplots(1,2, sharex='all', sharey='all')
        
        ax0.set_title("Input")
        im0 = ax0.imshow(in_array+0.5, origin="lower", vmin=0, vmax=1)
        plt.colorbar(im0, ax=ax0)
    
        ax1.set_title("Output")
        im1 = ax1.imshow(out_array, origin="lower", vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax1)
        
        fig.set_size_inches(8,4)
        plt.suptitle("Alpha = {}".format(alpha))
        plt.tight_layout()
        plt.show()
    return out_array

def norm_stretch(in_array, alpha=0.35):
    """The only function anyone outside will ever see"""
    return make_one_curve(xprime=in_array, alpha=alpha)

def many_alphas():
    fig, ax = plt.subplots(1,1)
    
    for use_equation_num, CurveString, ls, c in zip([1,4], ["Original", "New Roots Idea"], ["-","-"], ["r","b"]):
        demo_plot_many_alphas(axis=ax, ls=ls, c=c, label=CurveString)
    
    ax.legend()
    fig.set_size_inches((8,8))
    plt.show()
    
if __name__ == "__main__":
    pass
    fig, ax = plt.subplots(1,1)
    
    for use_equation_num, CurveString, ls, c in zip([1,4], ["Original", "New Roots Idea"], ["-","-"], ["r","b"]):
        demo_plot_many_alphas(axis=ax, ls=ls, c=c, first0=first0, label=CurveString)
        first0 = False
        
    
    ax.legend()
    fig.set_size_inches((8,8))
    plt.tight_layout()
    plt.show()
    # demo_plot_white_noise()
    # demo_plot_2D_method()

    # for alpha in np.linspace(1,2,10):
    #     plot_2d(alpha=alpha)

#
# The stretching parameter is "alpha".
#
# alpha=1 is linear... alpha>1 is stretched.   The largest values in the plot (like 7 or 8) are probably way too extreme.
#
#

# #original implimentation
# for i in range(nx):
#     for j in range(nalpha):
#         xprime = x_input_array[i] - 0.5
#         y_output_array[i, j] = 0.5 + (2. ** (alpha[j] - 1.)) * np.sign(xprime) * (np.abs(xprime) ** alpha[j])
