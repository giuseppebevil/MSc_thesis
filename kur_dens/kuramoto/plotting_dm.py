import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from kuramoto.plotting import *
    
def plot_S(tau1, tau2, tau3, tau4, S1_1, S2_1, S12_1, S1_2, S2_2, S12_2, S1_3, S2_3, S12_3, S1_4, S2_4, S12_4, value, title, var):
    # Function to plot both the entropy in function of tau for different values of the coupling
    fig, ax = plt.subplots(2, 2, figsize = (24, 12))
    fig.subplots_adjust(hspace=0.3)
    fs = 16
    if title is not None:
        plt.suptitle(title, fontsize = 25)
           
    # ax[0,0].plot(tau1, S1_1, linewidth = 0.9, color = 'navy', label = 'S1')
    # ax[0,0].plot(tau1, S2_1, linewidth = 0.9, color = 'green', label = 'S2')
    ax[0,0].plot(tau1, (np.array(S1_1) + np.array(S2_1))/2, linewidth = 0.9, color = 'red', label = 'S1+S2')
    ax[0,0].plot(tau1, S12_1, linewidth = 0.9, color = 'black', label = 'S12')
    ax[0,0].set_xlabel('$\\tau$', fontsize = 22)
    ax[0,0].set_ylabel('S', fontsize = 22)
    ax[0,0].set_title(f'{var} = {value[0]}', fontsize = 22)
    ax[0,0].set_xscale('symlog')
    ax[0,0].legend(fontsize = fs)
    ax[0,0].tick_params(axis = 'both', which = 'major', labelsize = 20)
    
    # ax[0,1].plot(tau2, S1_2, linewidth = 0.9, color = 'navy', label = 'S1')
    # ax[0,1].plot(tau2, S2_2, linewidth = 0.9, color = 'green', label = 'S2')
    ax[0,1].plot(tau2, (np.array(S1_2) + np.array(S2_2))/2, linewidth = 0.9, color = 'red', label = 'S1+S2')
    ax[0,1].plot(tau2, S12_2, linewidth = 0.9, color = 'black', label = 'S12')
    ax[0,1].set_xlabel('$\\tau$', fontsize = 22)
    ax[0,1].set_ylabel('S', fontsize = 22)
    ax[0,1].set_title(f'{var} = {value[1]}', fontsize = 22)
    ax[0,1].set_xscale('symlog')
    ax[0,1].legend(fontsize = fs)
    ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = 20)
    
    # ax[1,0].plot(tau3, S1_3, linewidth = 0.9, color = 'navy', label = 'S1')
    # ax[1,0].plot(tau3, S2_3, linewidth = 0.9, color = 'green', label = 'S2')
    ax[1,0].plot(tau3, (np.array(S1_3) + np.array(S2_3))/2, linewidth = 0.9, color = 'red', label = 'S1+S2')
    ax[1,0].plot(tau3, S12_3, linewidth = 0.9, color = 'black', label = 'S12')
    ax[1,0].set_xlabel('$\\tau$', fontsize = 22)
    ax[1,0].set_ylabel('S', fontsize = 22)
    ax[1,0].set_title(f'{var} = {value[2]}', fontsize = 22)
    ax[1,0].set_xscale('symlog')
    ax[1,0].legend(fontsize = fs)
    ax[1,0].tick_params(axis = 'both', which = 'major', labelsize = 20)
    
    # ax[1,1].plot(tau4, S1_4, linewidth = 0.9, color = 'navy', label = 'S1')
    # ax[1,1].plot(tau4, S2_4, linewidth = 0.9, color = 'green', label = 'S2')
    ax[1,1].plot(tau4, (np.array(S1_4) + np.array(S2_4))/2, linewidth = 0.9, color = 'red', label = 'S1+S2')
    ax[1,1].plot(tau4, S12_4, linewidth = 0.9, color = 'black', label = 'S12')
    ax[1,1].set_xlabel('$\\tau$', fontsize = 22)
    ax[1,1].set_ylabel('S', fontsize = 22)
    ax[1,1].set_title(f'{var} = {value[3]}', fontsize = 22)
    ax[1,1].set_xscale('symlog')
    ax[1,1].legend(fontsize = fs)
    ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = 20)

    #ax[1,1].set_xlim(-0.25,23)
    plt.savefig('c_ext_S.png')
    return ax


def plot_S_n(tau1, tau2, tau3, S1_1, S2_1, S12_1, S1_2, S2_2, S12_2, S1_3, S2_3, S12_3, value, title, var, scale):
    # Function to plot both the entropy in function of tau for different values of the coupling
    fs = 16
    lw = 4
    fig, ax = plt.subplots(figsize = (18, 11))
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax.plot(tau1, (np.array(S1_1) + np.array(S2_1))/2, linewidth = lw, color = 'red', linestyle = ':')
    #ax.plot(tau1, S12_1, linewidth = lw, color = 'cyan', linestyle = '--')
    ax.plot(tau1, S12_1, linewidth = lw, color = 'red', linestyle = 'solid', label = 'S12')
    ax.plot(tau2, (np.array(S1_2) + np.array(S2_2))/2, linewidth = lw, color = 'blue', linestyle = ':')
    ax.plot(tau2, S12_2, linewidth = lw, color = 'blue', linestyle = 'solid', label = 'S12')
    ax.plot(tau3, (np.array(S1_3) + np.array(S2_3))/2, linewidth = lw, color = 'darkgreen', linestyle = ':')
    ax.plot(tau3, S12_3, linewidth = lw, color = 'darkgreen', linestyle = 'solid', label = 'S12')
    # ax.plot(tau4, (np.array(S1_4) + np.array(S2_4))/2, linewidth = lw, color = 'purple', linestyle = ':')
    # ax.plot(tau4, S12_4, linewidth = lw, color = 'purple', linestyle = 'solid', label = 'S12')


    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    # left, bottom, width, height = [0.2, 0.2, 0.3, 0.3]
    # ax2 = fig.add_axes([left, bottom, width, height])
    # ax2
    # if scale == 'symlog':
    #     ax2.set_xscale('symlog')
    # ax2.plot(tau1, np.array(S2_1)/2, linewidth = lw, color = 'red', linestyle = '--', label = 'S2')
    # ax2.legend()
    # ax2.set_xlabel('$\\tau$')
    # ax2.set_ylabel('S')


    ax.set_xlabel('$\\tau$', fontsize = 40)
    ax.set_ylabel('S', fontsize = 40)
    if scale == 'symlog':
        ax.set_xscale('symlog')
    red_patch = mpatches.Patch(color = 'red', label = f'{var} = {value[0]}')
    blue_patch = mpatches.Patch(color = 'blue', label=f'{var} = {value[1]}')
    green_patch = mpatches.Patch(color = 'darkgreen', label = f'{var} = {value[2]}')
    dot_patch = Line2D([0,1],[0,1], color='black', linestyle=':', label = 'S1 + S2')
    solid_patch = Line2D([],[], color = 'black', linestyle = 'solid', label = 'S12')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 38)

    # Create another legend for the second line.
    ax.legend(handles=[red_patch, blue_patch, green_patch, dot_patch, solid_patch], fontsize = 40)
    #ax.set_xlim(-0.25,23)
    plt.savefig('S.png')
    return ax

def plot_entropy(tau1, tau2, tau3, S1_1, S2_1, S12_1, S1_2, S2_2, S12_2, S1_3, S2_3, S12_3, value, title, var, scale):
    
    fig, ax = plt.subplots(figsize = (18, 9))
    lw = 1.8
    
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax.plot(tau1, (np.array(S1_1) + np.array(S2_1))/2, linewidth = lw, color = 'red', label = f'{var} = {value[0]}')
    ax.plot(tau2, (np.array(S1_2) + np.array(S2_2))/2, linewidth = lw, color = 'blue', label = f'{var} = {value[1]}')
    ax.plot(tau3, (np.array(S1_3) + np.array(S2_3))/2, linewidth = lw, color = 'black', label = f'{var} = {value[2]}')
    
    ax.set_xlabel('$\\tau$', fontsize = 22)
    ax.set_ylabel('S', fontsize = 22)
    if scale == 'symlog':
        ax.set_xscale('symlog')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    #ax.legend(fontsize = 20)
    plt.savefig('S.png')
    
    return ax

def plot_S_n_inset(tau1, tau2, tau3, tau4, S1_1, S2_1, S12_1, S1_2, S2_2, S12_2, S1_3, S2_3, S12_3, S1_4, S2_4, S12_4, value, title, var, scale,
                    ord2_1, ord2_2, ord2_3, ord2_4, dt, t_max, title1, var1, value1):
    # Function to plot both the entropy in function of tau for different values of the coupling
    fs = 16
    lw = 1.8
    fig, ax = plt.subplots(figsize = (18, 9))
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax.plot(tau1, (np.array(S1_1) + np.array(S2_1))/2, linewidth = lw, color = 'red', linestyle = ':')
    #ax.plot(tau1, S2_1/2, linewidth = lw, color = 'cyan', linestyle = '--')
    ax.plot(tau1, S12_1, linewidth = lw, color = 'red', linestyle = 'solid', label = 'S12')
    ax.plot(tau2, (np.array(S1_2) + np.array(S2_2))/2, linewidth = lw, color = 'blue', linestyle = ':')
    ax.plot(tau2, S12_2, linewidth = lw, color = 'blue', linestyle = 'solid', label = 'S12')
    ax.plot(tau3, (np.array(S1_3) + np.array(S2_3))/2, linewidth = lw, color = 'darkgreen', linestyle = ':')
    ax.plot(tau3, S12_3, linewidth = lw, color = 'darkgreen', linestyle = 'solid', label = 'S12')
    # ax.plot(tau4, (np.array(S1_4) + np.array(S2_4))/2, linewidth = lw, color = 'purple', linestyle = ':')
    # ax.plot(tau4, S12_4, linewidth = lw, color = 'purple', linestyle = 'solid', label = 'S12')

    ax.set_xlabel('$\\tau$', fontsize = 22)
    ax.set_ylabel('S', fontsize = 22)
    if scale == 'symlog':
        ax.set_xscale('symlog')
    red_patch = mpatches.Patch(color = 'red', label = f'{var} = {value[0]}')
    blue_patch = mpatches.Patch(color = 'blue', label=f'{var} = {value[1]}')
    green_patch = mpatches.Patch(color = 'darkgreen', label = f'{var} = {value[3]}')
    purple_patch = mpatches.Patch(color = 'purple', label=f'{var} = {value[3]}')
    dot_patch = Line2D([0,1],[0,1], color='black', linestyle=':', label = 'S1 + S2')
    solid_patch = Line2D([],[], color = 'black', linestyle = 'solid', label = 'S12')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

    # Create another legend for the second line.
    ax.legend(handles=[red_patch, blue_patch, green_patch, dot_patch, solid_patch], fontsize = 18, loc = 'lower left')
    #ax.set_xlim(-0.25,23)
    
    left, bottom, width, height = [0.63, 0.4, 0.26, 0.36]
    ax1 = fig.add_axes([left, bottom, width, height])
    plot_many_r_inset(ax1, ord2_1, ord2_2, ord2_3, ord2_4, dt, t_max, title1, var1, value1)

    plt.savefig('S_inset.png')
    return ax

def plot_S_single(tau1, S, title, scale):
    # Function to plot both the entropy in function of tau for different values of the coupling
    fs = 16
    lw = 1.8
    fig, ax = plt.subplots(figsize = (18, 9))
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax.plot(tau1, np.array(S)/2, linewidth = lw, color = 'red', linestyle = '--', label = 'S2')
    
    ax.set_xlabel('$\\tau$', fontsize = 22)
    ax.set_ylabel('S', fontsize = 22)
    if scale == 'symlog':
        ax.set_xscale('symlog')
    ax.legend(fontsize = fs)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

    plt.savefig('c_S1.png')
    return ax

def plot_C_n(tau1, tau2, tau3, tau4, tau5, tau6, S1_1, S2_1, S12_1, S1_2, S2_2, S12_2, S1_3, S2_3, S12_3,
              S12_4, S1_4, S2_4, S1_5, S2_5, S12_5, S1_6, S2_6, S12_6, value, title, var):
    # Function to plot both the entropy in function of tau for different values of the coupling
    fs = 20
    lw = 1.8
    fig, ax = plt.subplots(figsize = (18, 9))
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    #ax.plot(tau1, (np.array(S1_1) + np.array(S2_1))/2, linewidth = lw, color = 'red', linestyle = ':')
    ax.plot(tau1, S12_1, linewidth = lw, color = 'red', linestyle = 'solid')
    #ax.plot(tau2, (np.array(S1_2) + np.array(S2_2))/2, linewidth = lw, color = 'blue', linestyle = ':')
    ax.plot(tau2, S12_2, linewidth = lw, color = 'blue', linestyle = 'solid')
    #ax.plot(tau3, (np.array(S1_3) + np.array(S2_3))/2, linewidth = lw, color = 'darkgreen', linestyle = ':')
    ax.plot(tau3, S12_3, linewidth = lw, color = 'darkgreen', linestyle = 'solid')
    #ax.plot(tau4, (np.array(S1_4) + np.array(S2_4))/2, linewidth = lw, color = 'purple', linestyle = ':')
    ax.plot(tau4, S12_4, linewidth = lw, color = 'black', linestyle = 'solid')
    ax.plot(tau5, S12_5, linewidth = lw, color = 'goldenrod', linestyle = 'solid')
    ax.plot(tau6, S12_6, linewidth = lw, color = 'magenta', linestyle = 'solid')
    
    ax.set_xlabel('$\\tau$', fontsize = 22)
    ax.set_ylabel('C', fontsize = 22)
    #ax.set_xscale('symlog')
    red_patch = mpatches.Patch(color = 'red', label = f'{var} = {value[0]}')
    blue_patch = mpatches.Patch(color = 'blue', label=f'{var} = {value[1]}')
    green_patch = mpatches.Patch(color = 'darkgreen', label = f'{var} = {value[2]}')
    black_patch = mpatches.Patch(color = 'black', label=f'{var} = {value[3]}')
    gold_patch = mpatches.Patch(color = 'goldenrod', label=f'{var} = {value[4]}')
    magenta_patch = mpatches.Patch(color = 'magenta', label=f'{var} = {value[5]}')
    dot_patch = Line2D([0,1],[0,1], color='black', linestyle=':', label = 'S1 + S2')
    dash_patch = Line2D([],[], color = 'black', linestyle = 'solid', label = 'S12')
    
    # Create another legend for the second line.
    ax.legend(handles=[red_patch, blue_patch, green_patch, black_patch, gold_patch, magenta_patch], fontsize = fs)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    #ax.set_xlim(-0.25,23)
    plt.savefig('C.png')
    return ax

def plot_tauc_c(x, tau1, tau2, tau3, err1, err2, err3, var, title, text):
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.errorbar(x, tau1, yerr = err1, linestyle = '--', color = 'blue', marker = '.', markersize = 30, label = 'BA', capsize = 7)
    ax.errorbar(x, tau2, yerr = err2, linestyle = '--', color = 'red', marker = '.', markersize = 30, label = 'ER', capsize = 7)
    ax.errorbar(x, tau3, yerr = err3, linestyle = '--', color = 'black', marker = '.', markersize = 30, label = 'ER-BA', capsize = 7)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 35)
    ax.set_ylabel('$\\tau_c$', fontsize = 40)
    ax.set_xlabel(f'{var}', fontsize = 40)
    ax.legend(fontsize = 28)
    ax.set_title(title, fontsize = 40)
    plt.text(0.0, 0.3, text, fontsize = 40)
    plt.savefig(f'{title}.png')
    return ax

def plot_tauc_c_com(x, tau1, tau2, tau3, err1, err2, err3, var, title, text):
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.errorbar(x, tau1, yerr = err1, linestyle = '--', color = 'blue', marker = '.', markersize = 30, label = 'BA', capsize = 7)
    ax.errorbar(x, tau2, yerr = err2, linestyle = '--', color = 'red', marker = '.', markersize = 30, label = 'ER', capsize = 7)
    ax.errorbar(x, tau3, yerr = err3, linestyle = '--', color = 'black', marker = '.', markersize = 30, label = 'ER-BA', capsize = 7)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 35)
    ax.set_ylabel('$\\tau_c$', fontsize = 40)
    ax.set_xlabel(f'{var}', fontsize = 40)
    #ax.legend(fontsize = 28)
    ax.set_title(title, fontsize = 40)
    plt.text(0.01, 0.12, text, fontsize = 40)
    plt.savefig(f'{title}.png')
    return ax

def plot_S_n1(tau1, tau2, S1_1, S2_1, S12_1, S1_2, S2_2, S12_2, value, title, var):
    # Function to plot both the entropy in function of tau for different values of the coupling
    fs = 16
    lw = 1.8
    fig, ax = plt.subplots(figsize = (18, 9))
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax.plot(tau1, (np.array(S1_1) + np.array(S2_1))/2, linewidth = lw, color = 'red', linestyle = ':')
    ax.plot(tau1, S12_1, linewidth = lw, color = 'red', linestyle = '--', label = 'S12')
    ax.plot(tau2, (np.array(S1_2) + np.array(S2_2))/2, linewidth = lw, color = 'blue', linestyle = ':')
    ax.plot(tau2, S12_2, linewidth = lw, color = 'blue', linestyle = '--', label = 'S12')
    
    ax.set_xlabel('$\\tau$', fontsize = 22)
    ax.set_ylabel('S', fontsize = 22)
    ax.set_xscale('symlog')
    ax.legend(fontsize = fs)
    red_patch = mpatches.Patch(color = 'red', label = f'{var} = {value[0]}')
    blue_patch = mpatches.Patch(color = 'blue', label=f'{var} = {value[1]}')
    dot_patch = Line2D([0,1],[0,1], color='black', linestyle=':', label = 'S1 + S2')
    dash_patch = Line2D([],[], color = 'black', linestyle = '--', label = 'S12')
    
    # Create another legend for the second line.
    ax.legend(handles=[red_patch, blue_patch, dot_patch, dash_patch])
    ax.grid(False)
    ax.set_xlim(-0.1,3)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    #plt.savefig('c_ext_S_n.png')
    return ax

def plot_F(tau1, tau2, tau3, tau4, F1_1, F2_1, F12_1, F1_2, F2_2, F12_2, F1_3, F2_3, F12_3, F1_4, F2_4, F12_4, value, title):
    var = 'c_ext'
    # Function to plot both the entropy in function of tau for different values of the coupling
    fig, ax = plt.subplots(2, 2, figsize = (24, 12))
    fig.subplots_adjust(hspace=0.3)
    fs = 16
    if title is not None:
        plt.suptitle(title, fontsize = 25)
           
    ax[0,0].plot(tau1, F1_1, linewidth = 0.9, color = 'navy', label = 'F1')
    ax[0,0].plot(tau1, F2_1, linewidth = 0.9, color = 'green', label = 'F2')
    ax[0,0].plot(tau1, np.array(F1_1) + np.array(F2_1), linewidth = 0.9, color = 'red', label = 'F1+F2')
    ax[0,0].plot(tau1, F12_1, linewidth = 0.9, color = 'cyan', label = 'F12')
    ax[0,0].set_xlabel('$\\tau$', fontsize = 22)
    ax[0,0].set_ylabel('F', fontsize = 22)
    ax[0,0].set_title(f'{var} = {value[0]}')
    ax[0,0].set_xscale('symlog')
    ax[0,0].legend(fontsize = fs)

    
    ax[0,1].plot(tau2, F1_2, linewidth = 0.9, color = 'navy', label = 'F1')
    ax[0,1].plot(tau2, F2_2, linewidth = 0.9, color = 'green', label = 'F2')
    ax[0,1].plot(tau2, np.array(F1_2) + np.array(F2_2), linewidth = 0.9, color = 'red', label = 'F1+F2')
    ax[0,1].plot(tau2, F12_2, linewidth = 0.9, color = 'cyan', label = 'F12')
    ax[0,1].set_xlabel('$\\tau$', fontsize = 22)
    ax[0,1].set_ylabel('F', fontsize = 22)
    ax[0,1].set_title(f'{var} = {value[1]}')
    ax[0,1].set_xscale('symlog')
    ax[0,1].legend(fontsize = fs)
    
    ax[1,0].plot(tau3, F1_3, linewidth = 0.9, color = 'navy', label = 'F1')
    ax[1,0].plot(tau3, F2_3, linewidth = 0.9, color = 'green', label = 'F2')
    ax[1,0].plot(tau3, np.array(F1_3) + np.array(F2_3), linewidth = 0.9, color = 'red', label = 'F1+F2')
    ax[1,0].plot(tau3, F12_3, linewidth = 0.9, color = 'cyan', label = 'F12')
    ax[1,0].set_xlabel('$\\tau$', fontsize = 22)
    ax[1,0].set_ylabel('F', fontsize = 22)
    ax[1,0].set_title(f'{var} = {value[3]}')
    ax[1,0].set_xscale('symlog')
    ax[1,0].legend(fontsize = fs)
    
    ax[1,1].plot(tau4, F1_4, linewidth = 0.9, color = 'navy', label = 'F1')
    ax[1,1].plot(tau4, F2_4, linewidth = 0.9, color = 'green', label = 'F2')
    ax[1,1].plot(tau4, np.array(F1_4) + np.array(F2_4), linewidth = 0.9, color = 'red', label = 'F1+F2')
    ax[1,1].plot(tau4, F12_4, linewidth = 0.9, color = 'cyan', label = 'F12')
    ax[1,1].set_xlabel('$\\tau$', fontsize = 22)
    ax[1,1].set_ylabel('F', fontsize = 22)
    ax[1,1].set_title(f'{var} = {value[3]}')
    ax[1,1].set_xscale('symlog')
    ax[1,1].legend(fontsize = fs)
    plt.savefig('c_ext_F.png')
    
    return ax