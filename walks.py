import random as r
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
import math
def in_1d_boundary(position, boundary):
    return position > boundary[0] and position < boundary[1]

def run_1d_trial(steps, boundary):
    """
    steps: number of steps to take
    boundary: set of points that consititute a boundary (doesn't make sense with more than 2)

    Runs trial for one dimension starting from 0 WLOG since we have control over the boundary.
    Returns a map from square -> number of times we reached that square
    """
    starting_point = 0
    seen_squares = {starting_point: 1} #includes starting square 
    take_step = lambda : (-2) *(r.random() < 0.5) + 1 # take a step left (-1) with p = 1/2 and right (+ 1) with p = 1/2
    current_sq = starting_point
    for step in range(steps):
        step_vector = take_step()
        current_sq += step_vector
        if not in_1d_boundary(current_sq, boundary):
            break
        seen_squares[current_sq] = seen_squares.setdefault(current_sq, 0) + 1 #increase square count by 1
    
    return current_sq, seen_squares 

def combine_freq_dists(f1, f2):
    new_f = {}
    for coord in f1:
        new_f[coord] = new_f.setdefault(coord, 0) + f1[coord]
    
    for coord in f2:
        new_f[coord] = new_f.setdefault(coord, 0) + f2[coord]
    

    return new_f
    
def visualize_data(results):
    exit_points = {}

    total_frequencies = {}
    for i, r in enumerate(results): 
        exit_point, freq_dist = r
        exit_points[exit_point] = exit_points.setdefault(exit_point, 0) + 1
        total_frequencies = combine_freq_dists(total_frequencies, freq_dist)

 
    x_coords = []
    freqs = []
    for f in total_frequencies:
        x_coords.append(f)
        freqs.append(total_frequencies[f])
    
    plt.plot(x_coords, freqs)
def run_sim(trials, steps, boundary):
    """
    trials: Number of trials to run in the simulation
    steps: Number of steps to run per trial
    """
    trial_results = []
    for trial in range(trials):
        trial_results.append(run_1d_trial(steps, boundary))
        print(f"Trial {trial}: ", run_1d_trial(steps, boundary))
    # post processing of results
    visualize_data(trial_results)


def markov_processes():
    n = 9 
    m = np.zeros((n, n))
    m[0, 1] = .5
    m[n-1, n-2] = .5
    for i in range(1, n-1):
        m[i, i-1] = .5
        m[i, i+1] = .5
    print(m)
    p_m = linalg.matrix_power(m, 10000)
    np.set_printoptions(precision=4)
    print(p_m)
    v = np.zeros(n)
    v[n//2] = 1
    res = p_m@v
    print(res)
    print("end probability: ", res[n//2] )

def what_step_exit(s, b):
    pos = 0
    for i in range(0, len(s), 2):
        if s[i:i+2] == '00':
            pos += 1
        elif s[i:i+2] == '01':
            pos -= 1
        if pos >= b:
            return i + 1
    return len(s)

def out_before(s, b):
    # 00 -> u 01 -> d,10 -> l, 01 -> r
    pos = 0
    for i in range(0, len(s)-2, 2):
        if s[i:i+2] == '00':
            pos += 1
        elif s[i:i+2] == '01': 
            pos -= 1

        if pos >= b:
            return True

    return False

def out_at_end(s, b):
    pos = 0
    for i in range(0, len(s), 2):
        if s[i:i+2] == '00':
            pos += 1
        elif s[i:i+2] == '01':
            pos -= 1
    return pos >= b

def counting(s, b):
    # a word is a binary number. 0 represents down, 1 represents up
    words = []
    for i in range(4**s):
        words.append(format(i, f"0{2*s}b"))

    #print(words)
    #eliminate all words that would exit before the final step
    valid_words = []
    double_check = {}
    for w in words:
        if not out_before(w, b):
            valid_words.append(w)
    
    #find all words that escape on the last step
    escaped_words = []
    for w in valid_words:
        if out_at_end(w, b):
            escaped_words.append(w)
    
    print(f"Valid Words {len(valid_words)} ")#, valid_words)
    print(f"Escaped Words {len(escaped_words)}")#, escaped_words) 
    print(f"Probability Escaping on step {s}: ", len(escaped_words)/len(valid_words)) 

memo = {}
def compute_number_paths_out(s, d):
    """
    Computes number of paths out in exactly s steps, when the boundary is d steps away.

    Recurrence: C(s, d) = C(s-1, d-1) + 2C(s-1, d) + C(s-1, d+1)
    
    The recurrence is for a random walk in 2d, where the boundary
    is the half plane that is d steps away.

    Base Cases: 
        C(s, d) = 0 if s < d // less steps than to boundary then obviously can't reach it
        C(s, d) = 0 if d = 0 // we have already hit the boundary and and there are steps left
        C(s, d) = 1 if s= d // there is only one path that exactly hits the boundary (going straight there)
 
    Using memoization should lead to a runtime of O(s^2)
    O(1) work on each state, O(s)O(d) states is an overcount since s < d requires no recursion
    """
    memo[(0, 0)] = 1
    for s_i in range(s+1):
        for d_i in range(d+s_i+1):
            if (s_i, d_i) not in memo:
                if (s_i < d_i):
                    memo[(s_i, d_i)] = 0
                elif d_i == 0:
                    memo[(s_i, d_i)] = 0
                elif s_i == d_i:
                    memo[(s_i, d_i)] = 1
                else:
                    memo[(s_i, d_i)] = memo[(s_i - 1, d_i -1)] + 2*memo[(s_i - 1, d_i)] + memo[(s_i - 1, d_i + 1)]
    return memo[(s, d)] 
    # if (s, d) in memo:
    #     return memo[(s, d)]
    
    # if (d > s): 
    #     return 0
    # if (d == 0): 
    #     return 0
    # if (s == d): 
    #     return 1

    # c = lambda x, y : compute_number_paths_out(x, y)
    # ans = c(s-1, d-1) + 2*c(s-1, d) + c(s-1, d+1)
    # memo[(s, d)] = ans
    # return ans

def nCr(n, r):
    f = math.factorial
    return f(n)//(f(n-r)*f(r))
def closed_form_paths_out(s, d):
    #rn only works for d = 2
    #compute row (2(s-1)) column 2(s-1)/2 + 1 in pascals triangle
    # print("computing s, d", s, d)
    current_row = 2*(s-1)
    offset = d-1
    midpoint = current_row//2
    total_num_out_with_ob = nCr(current_row, midpoint + offset) 
    # extra paths are the entry immeadiatly above and to the right (but for some reason pascals goes by 2)
    extra_paths = nCr(current_row, current_row//2 + offset +2) if (current_row >= midpoint + offset + 2 ) else 0
    # print(extra_paths)
    return total_num_out_with_ob - extra_paths

def correction(n, val):
    print(f"(n, val) {(n, val)}")
    if (n == 0):
        return val
    if (n==1): 
        return 2*val
    return 2*correction(n-1, val) + ((n%2) + 1)*val

memo_tot_paths = {}
def compute_number_paths(s, d):
    """
    Computes total number of paths of length s, that do not exit early

    Recurrence: C(s, d) = C(s-1, d-1) + 2C(s-1, d) + C(s-1, d+1)
    
    The recurrence is for a random walk in 2d, where the boundary
    is the half plane that is d steps away.

    Base Cases: 
        C(s, d) = (4)^s if s <= d // all paths are viable none will leave early 
        C(s, d) = 0 if d = 0 // we have already hit the boundary and and there are steps left
 
    Using memoization should lead to a runtime of O(s^2)
    O(1) work on each state, O(s)O(d) states is an overcount since s < d requires no recursion
    """
    memo_tot_paths[(0, 0)] = 1
    for s_i in range(s+1):
        for d_i in range(d+s_i+1):
            if (s_i, d_i) not in memo_tot_paths:
                if (s_i <= d_i):
                    memo_tot_paths[(s_i, d_i)] = (4)**s_i
                elif d_i == 0:
                    memo_tot_paths[(s_i, d_i)] = 0
                else:
                    memo_tot_paths[(s_i, d_i)] = memo_tot_paths[(s_i - 1, d_i -1)] + 2*memo_tot_paths[(s_i - 1, d_i)] + memo_tot_paths[(s_i - 1, d_i + 1)]
    return memo_tot_paths[(s, d)] 

def probability_leaving(s, d):
    total_probability = 0
    for i in range(1,s+1):
        # extra_paths = sum([compute_number_paths_out(j, d) for j in range(1, i-1)])
        total_probability += compute_number_paths_out(i, d)/(4**i)#(compute_number_paths(i, d) + extra_paths)
    return total_probability

if __name__ == "__main__":
    trials = 10 
    bound = 9 
    for steps in range(bound, bound+trials):
        cf_paths_out = closed_form_paths_out(steps, bound)
        num_paths_out = compute_number_paths_out(steps, bound)
        num_paths = compute_number_paths(steps, bound)
        prob_leave = probability_leaving(steps, bound)
        print(f"CF paths out: {cf_paths_out} Paths Out: {num_paths_out} diff {cf_paths_out - num_paths_out}")
        # print(f"Probability of leaving in under {steps} steps: {prob_leave}")
    # counting(steps, bound)
    #markov_processes()
    # b_1d = (float("-inf"), 7)
    # trials = 1
    # steps = 10
    # run_sim(trials, steps, b_1d)