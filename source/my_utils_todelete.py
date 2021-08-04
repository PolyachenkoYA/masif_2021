import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import re
import matplotlib.mlab as mlab
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.stats as stats
import pathlib
import subprocess
import sklearn.mixture as sk_mixture
from matplotlib.colors import LogNorm

user_home_path = os.path.expanduser('~')
error_str = '\n!!ERROR OCCURED!!\n'
my_eps = np.finfo(float).eps * 10
colors = [(0.8500, 0.3250, 0.0980),
          (0.9290, 0.6940, 0.1250),
          (0.4940, 0.1840, 0.5560),
          (0.4660, 0.6740, 0.1880),
          (0.3010, 0.7450, 0.9330),
          (0.6350, 0.0780, 0.1840),
          (1, 0, 0),
          (0, 0, 1),
          (0, 0.5, 0),              
          (0.25, 0.25, 0.25),
          (0.75, 0, 0.75),
          (0, 0.75, 0.75),
          (0.75, 0.75, 0),
          (0, 1, 0),
          (0, 0.4470, 0.7410),
          (0, 0, 0)]

float_regexp_full = r'[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
float_digital = r'[+-]?[0-9]+\.[0-9]+'

chain_ids_table = \
    {'2I25': {'uR':1, 'uL':0},  # 0
     '1Z0K': {'uR':0, 'uL':3},  # 1 
     '2HQS': {'uR':0, 'uL':1},  # 2
     '1R6Q': {'uR':0, 'uL':1},  # 3
     '2UUY': {'uR':0, 'uL':1},  # 4 +
     '1RKE': {'uR':1, 'uL':0},  # 5
     '1D6R': {'uR':0, 'uL':1},  # 6
     '1ZHH': {'uR':0, 'uL':1},  # 7
     '3SGQ': {'uR':0, 'uL':1},  # 8
     '1JTG': {'uR':1, 'uL':0},  # 9 +
     '2O3B': {'uR':0, 'uL':2},  # 10 +
     '1CGI': {'uR':0, 'uL':1},  # 11
     '1CLV': {'uR':0, 'uL':2},  # 12
     '3F1P': {'uR':0, 'uL':1},  # 13
     '1AK4': {'uR':0, 'uL':1},  # 14 +
     '1R0R': {'uR':0, 'uL':1},  # 15
     '1GPW': {'uR':0, 'uL':1},  # 16
     '1E96': {'uR':0, 'uL':3},  # 17
     '1FLE': {'uR':0, 'uL':1},  # 18
     '2OOB': {'uR':0, 'uL':1}}  # 19

def get_my_color(i):
    return colors[np.mod(i, len(colors))];

def totalMin(arr):
    while(type(arr) is list):
        arr = min(arr)
    return arr

def totalMax(arr):
    while(type(arr) is list):
        arr = max(arr)
    return arr

def sum(v, n1=0, n2=-1, s0=0):
    if(n2 == -1):
        n2 = len(v)
        
    s = s0
    for i in range(n1,n2):
        s += v[i]
    return s

def dot(v1, v2):
    #if(len(v1) != len(v2)):
    #    return NAN
    #return sum([ v1[i]*v2[i] for i in range(0,len(v1))])
    #return sum([ v1[i]*v2[i] for i in range(3)])
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    
def length(v):
    return math.sqrt(dot(v,v))
    
def arrDot(Varr1, Varr2):
    return [dot(Varr1[i], Varr2[i]) for i in range(min(len(Varr1), len(Varr2)))]
    
def arrFnc(v, fnc):
    return [fnc(_x) for _x in v]
        
def find_key(keys, key0):
    for _key in keys:
        if(_key == key0):
            return 1
    
    return 0
    
def run_it(command, shell=False):
    if(isinstance(command, str)):
        if(' ' in command):
            shell = True
            
    print(command)
    subprocess.run(command, shell=shell)
    
    #print(command)
    #ok = (os.system(command) == 0)
    #if(not ok):
    #    print(error_str)
    #return ok
        
def f2str(x, n=3):
    return '%s' % float(('%.' + str(n) + 'g') % x)
    
def eps_err(a, b): # exp(|ln(a/b)|) - 1
    x_min = min(abs(a),abs(b))
    if(x_min < my_eps):
        if(max(abs(a),abs(b)) < my_eps):
            return 0
        else:
            return 1/my_eps
    else:
        return abs(a - b)/x_min

def get_fig(xlbl, ylbl, title=None, xscl='linear', yscl='linear', projection=None, zscl='linear', zlbl='z'):
    if(title is None):
        title = ylbl + '(' + xlbl + ')'
    fig = plt.figure()
    ax = fig.gca(projection=projection)
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.xscale(xscl)
    plt.yscale(yscl)
    if(projection == '3d'):
        ax.set_zscale(zscl)
        ax.set_zlabel(zlbl)
    return fig, ax

def lorentzian(x, x0, y0, a, gam):
    #return y0 + a * gam**2 / ( gam**2 + ( x - x0 )**2)
    return y0 + a / ( 1 + ( (x - x0)/gam )**2)
def gaussian(x, x0, y0, a, gam):
    return y0 + a * np.exp(-((x - x0) / gam)**2)
def lin_fnc(x, k, b):
    return x * k + b

def linear_visualize(lin, d_lin, x, y, ax, d_x=None, d_y=None, \
                     color='black', data_lbl='data', lin_lbl='linear fit', \
                     conf_int_lbl='confid. int.', dy_lbl=None, dy_title=None):
    n = len(x)
    if(dy_lbl is None):
        dy_lbl = ax.get_ylabel()
    if(dy_title is None):
        dy_title = ax.get_title()
    if(d_y is None):
        d_y = np.ones(n)
    if(d_x is None):
        d_x = np.ones(n)
        
    ax.errorbar(x, y, yerr=d_y, xerr=d_x, \
                 color=color, fmt='o', mfc='white', capsize=3, ms=4, label=data_lbl, elinewidth=1)
    x_draw = np.linspace(min(x), max(x), 100)
    mean_x = np.mean(x)
    y_th = lin(x_draw)
    
    confid_dlt = np.sqrt((d_lin[0] * x_draw)**2 + d_lin[1]**2)
    
    #s_err = np.sum(((lin(x) - y) / d_y)**2)   # sum of the squares of the residuals     
    #confid_dlt = stats.t.ppf(0.95, n - 2) * np.sqrt((s_err / (n-2)) * \
    #             (1.0/n + ((x_draw - mean_x)**2 / (np.sum(x**2) - n * mean_x**2))))
    ax.plot(x_draw, y_th, \
             label=lin_lbl, color=color)
    ax.plot(x_draw, y_th + confid_dlt, \
             '--', color=color, lw=1)
    ax.plot(x_draw, y_th - confid_dlt, \
             '--', color=color, lw=1)
    
    fig2, ax2 = get_fig(xlbl=ax.get_xlabel(), ylbl=dy_lbl, title=dy_title)
    ax2.errorbar(x, y - lin(x), yerr=d_y, xerr=d_x, \
                 color=color, fmt='o', mfc='white', capsize=3, ms=4, label=data_lbl, elinewidth=1)
    ax2.plot(x_draw, [0] * len(x_draw), \
             color=color, label=lin_lbl)
    ax2.plot(x_draw, confid_dlt, '--', color=color, linewidth=1, label=conf_int_lbl)
    ax2.plot(x_draw, -confid_dlt, '--', color=color, linewidth=1)
    ax2.legend()
    
    return fig2, ax2

def linear_proc(x, y, d_x=None, d_y=None):
    if(d_y is None):
        d_y = np.array([1] * len(x))
        
    #lin_coefs, lin_cov = np.polyfit(x, y, 1, cov=True, w = 1/d_y)
#    lin_coefs, lin_cov = np.polyfit(x, y, 1, cov='unscaled', w = 1/d_y**2)
    k0 = np.mean((y[-1] - y[0]) / (x[-1] - x[0]))
    b0 = np.mean(y - k0 * x)
    lin_coefs, lin_cov = \
        scipy.optimize.curve_fit(lambda x, k, b: x*k + b, x, y, p0=[k0, b0], sigma=d_y)
    d_lin = np.sqrt(np.diag(lin_cov))
    lin = np.poly1d(lin_coefs)
    
    chi2 = np.sum(((lin(x) - y) / d_y)**2)
    chi2_u = chi2 / (len(x) - 2)
    p = scipy.stats.chi2.sf(chi2, len(x))
    
    return [lin, d_lin, chi2, chi2_u, p]
  
def fft_my(f, dt):
    A = abs(np.fft.rfft(f))
    return (A / len(A), np.linspace(0, 1, len(A)) * (np.pi / dt))

def butter_bandpass_filter(data, dt, highcut, lowcut, order=5):
    b, a = scipy.signal.butter(order, np.array([lowcut, highcut]) * (2 * dt), btype='band')
    y = scipy.signal.lfilter(b, a, data)
    return y
    
def uvec(v):
    return v / np.linalg.norm(v)
    
def y_range(y, margin=0.05):
    mn = min(y)
    mx = max(y)
    r = mx - mn
    return (mn - r * margin, mx + r * margin)

def pick_args(list, separators, ind):
    return list[ind+1 : min(separators[separators > ind])]

def index_safe(list, element, default=-1):
    return list.index(element) if element in list else default

def print_usage_default(flags, possible_values, possible_arg_numbers, default_values, exit_codes, N_pos_args):
    print('=============================')
    print('USAGE:')
    print('number of positional arguments:', N_pos_args)
    for i, f in enumerate(flags):
        print('flag:', f)
        print('\tpossible values:', possible_values[i])
        print('\tpossible arg numbers:', possible_arg_numbers[i])
        print('\tdafault value:', default_values[i])
        print('\texit code:', exit_codes[i])

def parse_args(args, flags, possible_values=None, possible_arg_numbers=None, default_values=None, exit_codes=None, N_pos_args=0, verbose=True):
    """
    possible_values:
    list of lists of possible values for each args group
    If(self[i] is None) then any value is permitted
    
    possible_arg_numbers:
    list, len(self)==len(flags)
    A list of acceptable len(flags_args[i]): return len(flags_args[i]) in self[i]
    If('+' in self[i]) then len(flags_args[i]) must be > 0
    If(self[i] is None) then any len(flags_args[i])
    
    exit_codes:
    list of exit codes to return in case arguments for the i-th flags are invalid
    if(self in None) then all codes are set to be 1
    If(self[i] is None) then don't exit of invalid arguments
    If(self is int) then self = [self] * N
    
    default_values:
    list of what to assign to flag_args[i] in case it has len==0 after all the parsing done
    """
    pos_args = args[0:N_pos_args]
    args = args[N_pos_args:]
    
    N_flags = len(flags)
    if(possible_values is None):
        possible_values = [None] * N_flags
    if(possible_arg_numbers is None):
        possible_arg_numbers = [None] * N_flags
    if(default_values is None):
        default_values = [None] * N_flags
    if(exit_codes is None):
        exit_codes = 1
    if(isinstance(exit_codes, int)):
        exit_codes = [exit_codes] * N_flags
    correct_input = [True] * N_flags
    
    all_wrong = False
    for pos_arg in pos_args:
        all_wrong = pos_arg in flags
        if(all_wrong):
            if(verbose):
                print(error_str, 'Not enough positional arguments (' + str(N_pos_args) + ' needed):\n', '"' + str(args) + '"', '\nThey must be set before the key-arguments.')
            all_wrong = True
            break    
    if(not args[0] in flags):
        if(verbose):
            print(error_str, 'Too many positional arguments (' + str(N_pos_args) + ' needed):\n', '"' + str(args) + '"', '\nThey must be set before the key-arguments.')
        all_wrong = True
    if(len(possible_values) != len(flags)):
        if(verbose):
            print(error_str, error_str + '\npossible_values: ', possible_values, '\nflags:', flags, '\nThey must have the same size', file=sys.stderr)
        all_wrong = True
    if(len(possible_arg_numbers) != len(flags)):
        if(verbose):
            print(error_str, 'possible_arg_numbers:', possible_arg_numbers, '\nflags:', flags, '\nThey must have the same size', file=sys.stderr)
        all_wrong = True    
    if(len(exit_codes) != len(flags)):
        if(verbose):
            print(error_str, 'exit_codes: ', exit_codes, '\nflags:', flags, '\nThey must have the same size', file=sys.stderr)
        all_wrong = True
    if(all_wrong):
        correct_input = [False] * N_flags

    for i in range(N_flags):
        if((not default_values[i] is None) and (not 0 in possible_arg_numbers[i])):
            if(verbose):
                print('Warning:\n The ' + str(i) + '-th option has default_value = ' + str(default_values[i]) + ' but must be set by a user (does not have 0 in the possible_arg_numbers[' + str(i) + '] = ' + str(possible_arg_numbers[i]) + ')')

    flags_positions = np.array([index_safe(args, flag) for flag in flags] + [len(args)])
    flags_args = [pick_args(args, flags_positions, f_pos) for f_pos in flags_positions[:-1]]
    for i in range(N_flags):
        if((not possible_arg_numbers[i] is None) and correct_input[i]):
            correct_input[i] = (len(flags_args[i]) > 0)  if('+' in possible_arg_numbers[i]) else (len(flags_args[i]) in possible_arg_numbers[i])
            if(not correct_input[i]):
                if(verbose):
                    print(error_str + '\nflags_args[', i, ']: ', flags_args[i], '\npossible numbers of arguments:', possible_arg_numbers[i], file=sys.stderr)
        if((not possible_values[i] is None) and correct_input[i]):
            for v in flags_args[i]:
                correct_input[i] = v in possible_values[i]
                if(not correct_input[i]):
                    flags_args[i] = [v]
                    break
    
    for i in range(N_flags):
        if(not possible_arg_numbers[i] is None):
            if((len(possible_arg_numbers[i]) == 1) and (possible_arg_numbers[i][0] == 1) and correct_input[i]):
                flags_args[i] = flags_args[i][0]
    
    for i,f in enumerate(flags_args):
        if((not exit_codes[i] is None) and (not correct_input[i])):
            if(verbose):
                print(error_str + '\nparameter ', flags_args[i], ' for the "', flags[i], '" is invalid')
            print_usage_default(flags, possible_values, possible_arg_numbers, default_values, exit_codes, N_pos_args)
            sys.exit(exit_codes[i])
                
    for i in range(N_flags):
        if(len(flags_args[i]) == 0):
            flags_args[i] = default_values[i]
                
    return pos_args + flags_args, [True] * N_pos_args + correct_input

def safe_copy(src, dst):
    if(src != dst):
        if(os.path.isfile(dst)):
            os.remove(dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

def git_root_path():
    return subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout.decode('utf-8')[:-1]

def gauss_classif(fig, ax, x, y, n_comps=2, draw_levels=True):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ax.scatter(x, y, s=3)
    ax.scatter(x_mean, y_mean, s=25, c='red')
    
    if(draw_levels):
        clf = sk_mixture.GaussianMixture(n_components=n_comps, covariance_type='full')
        X_train = np.array([x[:], y[:]]).T
        clf.fit(X_train)
        
        X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = -clf.score_samples(XX)
        Z = Z.reshape(X.shape)
        
        min_scale = 0
        max_scale = 3
        CS = ax.contour(X, Y, Z, norm=LogNorm(vmin=10**min_scale, vmax=10**max_scale), levels=np.logspace(min_scale, max_scale, num=10))
        CB = fig.colorbar(CS, shrink=0.8)
    
    return x_mean, y_mean
