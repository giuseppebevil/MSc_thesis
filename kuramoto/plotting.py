import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .kuramoto import Kuramoto


def plot_activity(activity, dt, t_max):
    """
    Plot sin(angle) vs time for each oscillator time series.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    _, ax = plt.subplots(figsize = (12, 4))
    colors = sns.color_palette("crest", n_colors = len(activity))
    for i in range(len(activity)):
        ax.plot(np.linspace(0, t_max, int(t_max/dt)), np.sin(activity[i]), linewidth = 0.9, color = colors[i])
    ax.set_xlabel('Time', fontsize = 22)
    ax.set_ylabel(r'$\sin(\theta)$', fontsize = 22)
    
    return ax

def plot_phase_coherence(order, dt, t_max, title):
    """
    Plot order parameter phase_coherence vs time.

    activity: 2D-np.ndarray
        Activity time series, node vs. time; ie output of Kuramoto.run()
    return:
        matplotlib axis for further customization
    """
    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), order, 'b', markersize = 0.5)
    ax.set_ylabel('r', fontsize = 22)
    ax.set_xlabel('t', fontsize = 22)
    #ax.set_ylim((-0.01, 1))
    if title is not None:
        plt.suptitle(title, fontsize = 25)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    plt.savefig('r_chaotic.png')
    return ax

def plot_many_r(ord1, ord2, ord3, ord4, ord5, ord6, dt, t_max, title, var, value):
    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord1, 'b', markersize = 0.5, label = f'{var} = {value[0]}')
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord2, 'r', markersize = 0.5, label = f'{var} = {value[1]}')
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord3, color = 'darkgreen', markersize = 0.5, label = f'{var} = {value[2]}')
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord4, color = 'black', markersize = 0.5, label = f'{var} = {value[3]}')
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord5, color = 'goldenrod', markersize = 0.5, label = f'{var} = {value[4]}')
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord6, color = 'cyan', markersize = 0.5, label = f'{var} = {value[5]}')

    if title is not None:
        plt.suptitle(title, fontsize = 25)
    ax.set_xlabel('t', fontsize=22)
    ax.set_ylabel('r', fontsize = 22)
    ax.legend(fontsize = 14)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    plt.savefig('varying_c_sim.png')

def plot_many_r_inset(ax, ord1, ord2, ord3, ord4, dt, t_max, title, var, value):
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord1, 'r', markersize = 0.5, label = f'{var} = {value[0]}')
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord2, 'b', markersize = 0.5, label = f'{var} = {value[1]}')
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord3, color = 'black', markersize = 0.5, label = f'{var} = {value[3]}')
    ax.plot(np.linspace(0, t_max, int(t_max/dt)), ord4, 'g', markersize = 0.5, label = f'{var} = {value[5]}')
    if title is not None:
        plt.suptitle(title)
    ax.set_xlabel('t')
    ax.set_ylabel('r$_2$')
    ax.legend()
    ax.tick_params(axis = 'both', which = 'major')

def plot_order_parameters(ord1, r1, ord2, r2, dt, t_max, title, leg):
    fig, ax = plt.subplots(2, 1, figsize = (12, 6))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), ord1, 'b', markersize = 0.5, label = 'calculated r')
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), r1, 'r', markersize = 0.5, label = 'r ode')
    ax[0].set_xlabel('t', fontsize=22)
    ax[0].set_ylabel('r$_1$', fontsize = 22)
    if leg:
        ax[0].legend(fontsize = 17)
    ax[0].tick_params(axis = 'both', which = 'major', labelsize = 20)
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord2, 'b', markersize = 0.5, label = 'calculated r')
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), r2,  'r', markersize = 0.5, label = 'r ode')
    ax[1].set_xlabel('t', fontsize=22)
    ax[1].set_ylabel('r$_2$', fontsize = 22)
    if leg:
        ax[1].legend(fontsize = 17)
    ax[1].tick_params(axis = 'both', which = 'major', labelsize = 20)
    plt.savefig(f'{title}.png')
    
def plot_three_r(ord1, r1, ri1, ord2, r2, ri2, dt, t_max, title):
    fig, ax = plt.subplots(2, 1, figsize = (24, 12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), ord1, 'b', markersize = 0.5, label = 'calculated r')
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), r1, 'r', markersize = 0.5, label = 'r diff')
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), ri1, 'g', markersize = 0.5, label = 'r_i diff')
    ax[0].set_xlabel('t', fontsize=22)
    ax[0].set_ylabel('r', fontsize = 22)
    ax[0].legend()
    
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord2, 'b', markersize = 0.5, label = 'calculated r')
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), r2,  'r', markersize = 0.5, label = 'r diff')
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ri2, 'g', markersize = 0.5, label = 'r_i diff')
    ax[1].set_xlabel('t', fontsize=22)
    ax[1].set_ylabel('r', fontsize = 22)
    ax[1].legend()

def plot_diff_r(r1, ord1, r2, ord2, dt, t_max, title):
    fig, ax = plt.subplots(2, 1, figsize = (24, 12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), ord1-r1, 'b', markersize = 0.5)
    ax[0].set_xlabel('t', fontsize=22)
    ax[0].set_ylabel('r1', fontsize = 22)
    
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord2-r2, 'b', markersize = 0.5)
    ax[1].set_xlabel('t', fontsize=22)
    ax[1].set_ylabel('r2', fontsize = 22)

def plot_order_params(r1, r_link1, r2, r_link2, c_max, c_points, title, xlab):
    fig, ax = plt.subplots(2, 1, figsize = (24, 12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    ax[0].plot(np.linspace(0, c_max, c_points), r1, linewidth = 0.9, color = 'navy', label = 'ER', marker = '.', markersize = 20, linestyle = '--')
    ax[0].plot(np.linspace(0, c_max, c_points), r2, linewidth = 0.9, color = 'goldenrod', label = 'BA', marker = '.', markersize = 20, linestyle = '--')
    ax[0].set_xlabel(xlab, fontsize=30)
    ax[0].set_ylabel('r', fontsize = 30)
    ax[0].legend(fontsize = 22)
    ax[0].tick_params(axis = 'both', which = 'major', labelsize = 30)
    if title is not None:
        plt.suptitle(title, fontsize = 32)
        
    ax[1].plot(np.linspace(0, c_max, c_points), r_link1, linewidth = 0.9, color = 'navy', label = 'ER', marker = '.', markersize = 20, linestyle = '--')
    ax[1].plot(np.linspace(0, c_max, c_points), r_link2, linewidth = 0.9, color = 'goldenrod', label = 'BA', marker = '.', markersize = 20, linestyle = '--')
    ax[1].set_xlabel(xlab, fontsize=30)
    ax[1].set_ylabel(r'$r_{link}$', fontsize = 30)
    ax[1].tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.savefig('varying_sigma.png')
    
def plot_heat_n(r1, r_link1, r2, r_link2, c_max, c_points, ce_max, ce_points):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (24,12))
    fig.subplots_adjust(wspace=0.01)

    xlabels = ['{:3.2f}'.format(x) for x in np.linspace(0, ce_max, ce_points)]
    ylabels = ['{:3.2f}'.format(y) for y in np.linspace(0, c_max, c_points)]
    
    s1 = sns.heatmap(r1, cmap = 'crest', ax = ax1)
    ax1.set_xlabel('b', fontsize = 30)
    ax1.set_ylabel('c', fontsize = 30)
    ax1.set_title('$r_{ER}$', fontsize = 30)
    ax1.set_xticks(range(0, ce_points, 3))
    ax1.set_xticklabels(f'{c:.2f}' for c in np.arange(0.0, ce_max, 0.25))
    ax1.set_yticks(range(0, c_points, 3))
    ax1.set_yticklabels(f'{c:.2f}' for c in np.arange(0.0, c_max, 0.03))
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 25)
    cbar = ax1.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    
    s2 = sns.heatmap(r_link1, cmap = 'crest', xticklabels = xlabels, yticklabels = False, ax = ax2)
    ax2.set_xlabel('b', fontsize = 30)
    ax2.set_title('$r_{link,ER}$', fontsize = 30)
    ax2.set_xticks(range(0, 20, 3))
    ax2.set_xticklabels(f'{c:.2f}' for c in np.arange(0.0, ce_max, 0.25))
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 25)
    ax2.yaxis.tick_right()

    fig.subplots_adjust(wspace=0.001)
    plt.savefig('heat_r.png')
    
    fig1, (ax3, ax4) = plt.subplots(ncols=2, figsize = (24,12))
    fig1.subplots_adjust(wspace=0.01)
    s3 = sns.heatmap(r2, cmap = 'crest', xticklabels = xlabels, yticklabels = ylabels, ax = ax3)
    ax3.set_xlabel('b', fontsize = 30)
    ax3.set_ylabel('c', fontsize = 30)
    ax3.set_title('$r_{BA}$', fontsize = 30)
    ax3.set_xticks(range(0, ce_points, 3))
    ax3.set_xticklabels(f'{c:.2f}' for c in np.arange(0.0, ce_max, 0.25))
    ax3.set_yticks(range(0, c_points, 3))
    ax3.set_yticklabels(f'{c:.2f}' for c in np.arange(0.0, c_max, 0.03))
    ax3.tick_params(axis = 'both', which = 'major', labelsize = 25)
    cbar = ax3.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    
    s4 = sns.heatmap(r_link2, cmap = 'crest', xticklabels = xlabels, yticklabels = False, ax = ax4)
    ax4.set_xlabel('b', fontsize = 30)
    ax4.set_title('$r_{link,BA}$', fontsize = 30)
    ax4.set_xticks(range(0, 20, 3))
    ax4.set_xticklabels(f'{c:.2f}' for c in np.arange(0.0, ce_max, 0.25))
    ax4.tick_params(axis = 'both', which = 'major', labelsize = 25)
    ax4.yaxis.tick_right()
    
    plt.show()
    plt.savefig('heat_rlink.png')    

def plot_heat(r1, r_link1, r2, r_link2, c_max, c_points, ce_max, ce_points):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (24,12))
    fig.subplots_adjust(wspace=0.01)

    xlabels = ['{:3.2f}'.format(x) for x in np.linspace(0, ce_max, ce_points)]
    ylabels = ['{:3.2f}'.format(y) for y in np.linspace(0, c_max, c_points)]
    s1 = sns.heatmap(r1, cmap = 'crest', xticklabels = xlabels, yticklabels = ylabels, ax = ax1)
    s1.set(xlabel='b', ylabel = 'c', title = '$r_{ER}$')
    s2 = sns.heatmap(r2, cmap = 'crest', xticklabels = xlabels, yticklabels = False, ax = ax2)
    s2.set(xlabel='b', title = '$r_{BA}$')
    ax1.set_xticks(np.array([0,3,6,9,12,15,18,21]))
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 30)
    ax2.yaxis.tick_right()

    fig.subplots_adjust(wspace=0.001)
    
    fig1, (ax3, ax4) = plt.subplots(ncols=2, figsize = (24,12))
    fig1.subplots_adjust(wspace=0.01)
    s3 = sns.heatmap(r_link1, cmap = 'crest', xticklabels = xlabels, yticklabels = ylabels, ax = ax3)
    s3.set(xlabel='b', ylabel = 'c', title = '$r_{link,ER}$')
    s4 = sns.heatmap(r_link2, cmap = 'crest', xticklabels = xlabels, yticklabels = False, ax = ax4)
    s4.set(xlabel='b', title = '$r_{link,BA}$')
    ax4.yaxis.tick_right()

    plt.show()

def plot_both(activity, r, dt, t_max, title):
    
    # Function to plot both the trajescories and the order parameter in function of time
    
    fig, ax = plt.subplots(2, 1, figsize = (24, 12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    colors = sns.color_palette("crest", n_colors = len(activity))
    for i in range(int(len(activity)/2)):
        ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), np.sin(activity[i]), linewidth = 0.9, color = colors[i])
    for i in range(int(len(activity)/2), len(activity)):
        ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), np.sin(activity[i]), linewidth = 0.9, color = colors[i])
    ax[0].set_xlabel('t', fontsize=22)
    ax[0].set_ylabel(r'$\sin(\theta)$', fontsize = 22)
    ax[0].set_title("Trajectories", fontsize = 23)
    if title is not None:
        plt.suptitle(title, fontsize = 25)
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), r, 'bo', markersize = 1)
    #ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), [Kuramoto.phase_coherence(vec)[1] for vec in activity.T], 'ro', markersize = 1, label = 'order2')
    ax[1].set_ylabel('Order parameter', fontsize = 22)
    ax[1].set_xlabel('t', fontsize = 22)
    ax[1].set_ylim((-0.01, 1))
    
    return ax

def plot_field_order(b, ord1, ord2, dt, t_max, title):
    
    # Function to plot both the trajescories and the order parameter in function of time
    
    fig, ax = plt.subplots(3, 1, figsize = (24, 12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), b, linewidth = 0.9, color = 'navy')
    ax[0].set_xlabel('t', fontsize = 22)
    ax[0].set_ylabel('b', fontsize = 22)
    #ax[0].set_title('External field', fontsize = 25)
    ax[0].set_ylim((np.min(b)-0.3, np.max(b) + 0.1))
    ax[0].tick_params(axis = 'both', which = 'major', labelsize = 20)
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord1, 'b', markersize = 0.5)
    ax[1].set_ylabel('r', fontsize = 22)
    ax[1].set_xlabel('t', fontsize = 22)
    ax[1].set_ylim((np.min(ord1)-0.1, np.max(ord1) + 0.1))
    ax[1].set_title('First network', fontsize = 23)
    ax[1].tick_params(axis = 'both', which = 'major', labelsize = 20)
    
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), ord2, 'b', markersize = 0.5)
    ax[2].set_ylabel('r', fontsize = 22)
    ax[2].set_xlabel('t', fontsize = 22)
    ax[2].set_ylim((np.min(ord2)-0.1, np.max(ord2) + 0.1))
    ax[2].set_title('Second network', fontsize = 23)
    ax[2].tick_params(axis = 'both', which = 'major', labelsize = 20)

    return ax

def plot_field_order_four(b1, b2, b3, b4, ord1_1, ord1_2, ord1_3, ord1_4, ord2_1, ord2_2, ord2_3, ord2_4, dt, t_max, title, var, value):
    
    # Function to plot both the trajescories and the order parameter in function of time
    
    fig, ax = plt.subplots(3, 1, figsize = (24, 12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), b1, color = 'blue', label = f'{var} = {value[0]}')
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), b2, color = 'red', label = f'{var} = {value[1]}')
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), b3, color = 'black', label = f'{var} = {value[2]}')
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), b4, color = 'green', label = f'{var} = {value[3]}')
    ax[0].set_xlabel('t', fontsize = 35)
    ax[0].set_ylabel('b', fontsize = 35)
    #ax[0].set_title('External field', fontsize = 25)
    #ax[0].set_ylim((np.min(b)-0.3, np.max(b) + 0.1))
    ax[0].tick_params(axis = 'both', which = 'major', labelsize = 28)
    if title is not None:
        plt.suptitle(title, fontsize = 30)
        
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord1_1, color = 'blue', label = f'{var} = {value[0]}', markersize = 0.5)
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord1_2, color = 'red', label = f'{var} = {value[1]}', markersize = 0.5)
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord1_3, color = 'black', label = f'{var} = {value[2]}', markersize = 0.5)
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord1_4, color = 'green', label = f'{var} = {value[3]}', markersize = 0.5)
    ax[1].set_ylabel('r', fontsize = 35)
    ax[1].set_xlabel('t', fontsize = 35)
    #ax[1].set_ylim((np.min(ord1_1)-0.1, np.max(ord1_1) + 0.1))
    ax[1].set_title('First network', fontsize = 30)
    ax[1].tick_params(axis = 'both', which = 'major', labelsize = 28)
    
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), ord2_1, color = 'blue', label = f'{var} = {value[0]}', markersize = 0.5)
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), ord2_2, color = 'red', label = f'{var} = {value[1]}', markersize = 0.5)
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), ord2_3, color = 'black', label = f'{var} = {value[2]}', markersize = 0.5)
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), ord2_4, color = 'green', label = f'{var} = {value[3]}', markersize = 0.5)
    ax[2].set_ylabel('r', fontsize = 35)
    ax[2].set_xlabel('t', fontsize = 35)
    #ax[2].set_ylim((np.min(ord2)-0.1, np.max(ord2) + 0.1))
    ax[2].set_title('Second network', fontsize = 30)
    ax[2].tick_params(axis = 'both', which = 'major', labelsize = 28)
    ax[1].legend(fontsize = 22)
    
    plt.savefig('varying_c_com.png')
    
    return ax

def plot_field_order_two(b, ord1, ord2, r1, r2, dt, t_max, title):
    
    # Function to plot both the trajescories and the order parameter in function of time
    
    fig, ax = plt.subplots(3, 1, figsize = (24, 12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), b, linewidth = 0.9, color = 'navy')
    ax[0].set_xlabel('t', fontsize = 22)
    ax[0].set_ylabel('b(t)', fontsize = 22)
    #ax[0].set_title('External field', fontsize = 25)
    ax[0].set_ylim((np.min(b)-0.3, np.max(b) + 0.1))
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord1, 'b', markersize = 0.5, label = 'calculated r')
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), r1,  'g', markersize = 0.5, label = 'ri')
    ax[1].set_ylabel('r', fontsize = 22)
    ax[1].set_xlabel('t', fontsize = 22)
    ax[1].set_ylim((np.min([r1, ord1])-0.1, np.max([r1, ord1]) + 0.1))
    ax[1].set_title('First network', fontsize = 23)
    ax[1].legend()
    
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), ord2, 'b', markersize = 0.5, label = 'calculated r')
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), r2,  'g', markersize = 0.5, label = 'ri')
    ax[2].set_ylabel('r', fontsize = 22)
    ax[2].set_xlabel('t', fontsize = 22)
    ax[2].set_ylim((np.min([r2, ord2])-0.1, np.max([r2, ord2]) + 0.1))
    ax[2].set_title('Second network', fontsize = 23)
    ax[2].legend()
    
    return ax

def plot_betas(b_1, b_2, ord1_1, ord1_2, r1_1, r2_1, ord2_1, ord2_2, r1_2, r2_2, betas_1, betas_2, dt, t_max, title):
    
    # Function to plot both the trajescories and the order parameter in function of time
    fig, ax = plt.subplots(3, 2, figsize = (24, 12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    fs = 10
    
    ax[0,0].plot(np.linspace(0, t_max, int(t_max/dt)), b_1, linewidth = 0.9, color = 'navy')
    ax[0,0].set_xlabel('t', fontsize = 22)
    ax[0,0].set_ylabel('b(t)', fontsize = 22)
    #ax[0].set_title('External field', fontsize = 25)
    ax[0,0].set_ylim((np.min(b_1)-0.3, np.max(b_1) + 0.1))
    ax[0,0].set_title('Barabasi-Albert')
    if title is not None:
        plt.suptitle(title, fontsize = 25)
    
    for i in range(len(betas_1)):
            ax[1,0].plot(np.linspace(0, t_max, int(t_max/dt)), r1_1[i], markersize = 0.5, label = f'$\\beta = ${betas_1[i]}', linestyle = ':')
            
    ax[1,0].plot(np.linspace(0, t_max, int(t_max/dt)), ord1_1, color = 'black', markersize = 0.5, label = 'calculated r', linestyle = '--')
    ax[1,0].set_ylabel('r', fontsize = 22)
    ax[1,0].set_xlabel('t', fontsize = 22)
    ax[1,0].set_ylim((np.min(ord1_1)-0.1, np.max(ord1_1) + 0.1))
    ax[1,0].set_title('First network', fontsize = 23)
    ax[1,0].legend(fontsize = fs)
    
    for i in range(len(betas_1)):
        ax[2,0].plot(np.linspace(0, t_max, int(t_max/dt)), r2_1[i], markersize = 0.5, label = f'$\\beta = ${betas_1[i]}', linestyle = ':')
    ax[2,0].plot(np.linspace(0, t_max, int(t_max/dt)), ord2_1, color = 'black', markersize = 0.5, label = 'calculated r', linestyle = '--')
    ax[2,0].set_ylabel('r', fontsize = 22)
    ax[2,0].set_xlabel('t', fontsize = 22)
    ax[2,0].set_ylim((np.min(ord2_1)-0.1, np.max(ord2_1) + 0.1))
    ax[2,0].set_title('Second network', fontsize = 23)
    ax[2,0].legend(fontsize = fs)


    ax[0,1].plot(np.linspace(0, t_max, int(t_max/dt)), b_2, linewidth = 0.9, color = 'navy')
    ax[0,1].set_xlabel('t', fontsize = 22)
    ax[0,1].set_ylabel('b(t)', fontsize = 22)
    #ax[0].set_title('External field', fontsize = 25)
    ax[0,1].set_ylim((np.min(b_2)-0.3, np.max(b_2) + 0.1))
    ax[0,1].set_title('Erdos-Renyi')
    
    for i in range(len(betas_2)):
            ax[1,1].plot(np.linspace(0, t_max, int(t_max/dt)), r1_2[i], markersize = 0.5, label = f'$\\beta = ${betas_2[i]}', linestyle = ':')
            
    ax[1,1].plot(np.linspace(0, t_max, int(t_max/dt)), ord1_2, color = 'black', markersize = 0.5, label = 'calculated r', linestyle = '--')
    ax[1,1].set_ylabel('r', fontsize = 22)
    ax[1,1].set_xlabel('t', fontsize = 22)
    ax[1,1].set_ylim((np.min(ord1_2)-0.1, np.max(ord1_2) + 0.1))
    ax[1,1].set_title('First network', fontsize = 23)
    ax[1,1].legend(fontsize = fs)
    
    for i in range(len(betas_2)):
        ax[2,1].plot(np.linspace(0, t_max, int(t_max/dt)), r2_2[i], markersize = 0.5, label = f'$\\beta = ${betas_2[i]}', linestyle = ':')
    ax[2,1].plot(np.linspace(0, t_max, int(t_max/dt)), ord2_2, color = 'black', markersize = 0.5, label = 'calculated r', linestyle = '--')
    ax[2,1].set_ylabel('r', fontsize = 22)
    ax[2,1].set_xlabel('t', fontsize = 22)
    ax[2,1].set_ylim((np.min(ord2_2)-0.1, np.max(ord2_2) + 0.1))
    ax[2,1].set_title('Second network', fontsize = 23)
    ax[2,1].legend(fontsize = fs)
      
    return ax

def plot_field_order_three(b, ord1, r1, ri1, ord2, r2, ri2, dt, t_max, title):
    
    # Function to plot both the trajescories and the order parameter in function of time
    
    fig, ax = plt.subplots(3, 1, figsize = (24, 12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), b, linewidth = 0.9, color = 'navy')
    ax[0].set_xlabel('t', fontsize = 22)
    ax[0].set_ylabel('b(t)', fontsize = 22)
    #ax[0].set_title('External field', fontsize = 25)
    ax[0].set_ylim((np.min(b)-0.3, np.max(b) + 0.1))
    if title is not None:
        plt.suptitle(title, fontsize = 25)
        
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ord1, 'b', markersize = 0.5, label = 'calculated r')
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), r1,  'r', markersize = 0.5, label = 'r diff')
    ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), ri1, 'g', markersize = 0.5, label = 'r_i diff')
    ax[1].set_ylabel('r', fontsize = 22)
    ax[1].set_xlabel('t', fontsize = 22)
    ax[1].set_ylim((np.min([r1, ord1])-0.1, np.max([r1, ord1]) + 0.1))
    ax[1].set_title('First network', fontsize = 23)
    ax[1].legend()
    
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), ord2, 'b', markersize = 0.5, label = 'calculated r')
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), r2,  'r', markersize = 0.5, label = 'r diff')
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), ri2, 'g', markersize = 0.5, label = 'r_i diff')
    ax[2].set_ylabel('r', fontsize = 22)
    ax[2].set_xlabel('t', fontsize = 22)
    ax[2].set_ylim((np.min([r2, ord2])-0.1, np.max([r2, ord2]) + 0.1))
    ax[2].set_title('Second network', fontsize = 23)
    ax[2].legend()
    
    return ax

def plot_field(activity, dt, t_max):
    _, ax = plt.subplots(figsize=(12, 4))

    ax.plot(np.linspace(0, t_max, int(t_max/dt)), activity, linewidth = 0.9, color = 'navy')
    ax.set_xlabel('t', fontsize = 22)
    ax.set_ylabel('b(t)', fontsize = 22)
    ax.set_title('External coupling', fontsize = 25)
    
def plot_all(activity1, activity2, ord1, ord2, field, dt, t_max):
    
    # Function to plot both the trajescories and the order parameter in function of time
    fig, ax = plt.subplots(5, 1, figsize = (30, 18))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    
    ax[0].plot(np.linspace(0, t_max, int(t_max/dt)), field, linewidth = 0.9, color = 'navy')
    ax[0].set_xlabel('t', fontsize = 22)
    ax[0].set_ylabel('b(t)', fontsize = 22)
    ax[0].set_title('External coupling', fontsize = 25)
    
    colors1 = sns.color_palette("crest", n_colors = len(activity1))
    for i in range(len(activity1)):
        ax[1].plot(np.linspace(0, t_max, int(t_max/dt)), np.sin(activity1[i]), linewidth = 0.9, color = colors1[i])
    ax[1].set_xlabel('t', fontsize=22)
    ax[1].set_ylabel(r'$\sin(\theta)$', fontsize = 22)
    ax[1].set_title('First network', fontsize = 23)
    #plt.suptitle('First network', fontsize = 25)
    ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), ord1, 'bo', markersize = 1, )
    #ax[2].plot(np.linspace(0, t_max, int(t_max/dt)), [Kuramoto.phase_coherence(vec) for vec in activity1.T], 'bo', markersize = 1, )
    ax[2].set_ylabel('Order parameter', fontsize = 22)
    ax[2].set_xlabel('t', fontsize = 22)
    ax[2].set_ylim((-0.01, 1))
    
    colors2 = sns.color_palette("crest", n_colors = len(activity2))
    for i in range(len(activity2)):
        ax[3].plot(np.linspace(0, t_max, int(t_max/dt)), np.sin(activity2[i]), linewidth = 0.9, color = colors2[i])
    ax[3].set_xlabel('t', fontsize=22)
    ax[3].set_ylabel(r'$\sin(\theta)$', fontsize = 22)
    ax[3].set_title('Second network', fontsize = 23)
    #plt.suptitle('First network', fontsize = 25)
    ax[4].plot(np.linspace(0, t_max, int(t_max/dt)), ord2, 'bo', markersize = 1, )
    #ax[4].plot(np.linspace(0, t_max, int(t_max/dt)), [Kuramoto.phase_coherence(vec) for vec in activity2.T], 'bo', markersize = 1, )
    ax[4].set_ylabel('Order parameter', fontsize = 22)
    ax[4].set_xlabel('t', fontsize = 22)
    ax[4].set_ylim((-0.01, 1))
    
    return ax

def plot_all1(activity1, activity2, field, dt, t_max):
    
    # Function to plot both the trajescories and the order parameter in function of time
    fig, ax = plt.subplots(2, 1, figsize = (24,12))
    fig.subplots_adjust(hspace=0) # needed in order to share the same x axis
    #ax[0] = plot_both(activity1, dt, t_max, 'First network')
    #ax[1] = plot_both(activity2, dt, t_max, 'Second network')
    ax[0].plot(np.linspace(0,2*np.pi), np.sin(np.linspace(0,2*np.pi)))
    ax[1].plot(np.linspace(0,2*np.pi), np.cos(np.linspace(0,2*np.pi)))
    return ax