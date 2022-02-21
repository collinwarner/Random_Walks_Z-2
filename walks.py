import random as r
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg

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
    for i, c in enumerate(s):
        if c == '0':
            pos -= 1
        else:
            pos += 1
        if abs(pos) >= b:
            return i + 1
    return len(s)

def out_before(s, b):
    pos = 0
    for c in s[:-1]:
        if c == '0':
            pos -= 1
        else: 
            pos += 1

        if abs(pos) >= b:
            return True

    return False

def out_at_end(s, b):
    pos = 0
    for c in s:
        if c == '0':
            pos -= 1
        else:
            pos += 1
    return abs(pos) >= b

def counting(s, b):
    # a word is a binary number. 0 represents down, 1 represents up
    words = []
    for i in range(2**s):
        words.append(format(i, f"0{s}b"))
    
    #print(words)
    #eliminate all words that would exit before the final step
    valid_words = []
    double_check = {}
    for w in words:
        if not out_before(w, b):
            valid_words.append(w)
        else:
            double_check.setdefault(what_step_exit(w, b), set())
            double_check[what_step_exit(w, b)].add(w[:what_step_exit(w, b)])
    
    # print(double_check)
    extra_num = 0
    for k in double_check: 
        extra_num += len(double_check[k])
    #find all words that escape on the last step
    escaped_words = []
    for w in valid_words:
        if out_at_end(w, b):
            escaped_words.append(w)
    print(f"Valid Words {len(valid_words)} {extra_num}\n")#, valid_words)
    print(f"Escaped Words {len(escaped_words)}\n")#, escaped_words) 
    print(f"Probability Escaping on step {s}: ", len(escaped_words)/len(valid_words)) 
    print(f"Probability Escaping on step {s}: ", len(escaped_words)/(len(valid_words) + extra_num)) 

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
    if (s, d) in memo:
        return memo[(s, d)]
    
    if (d > s): 
        return 0
    if (d == 0): 
        return 0
    if (s == d): 
        return 1

    c = lambda x, y : compute_number_paths_out(x, y)
    ans = c(s-1, d-1) + 2*c(s-1, d) + c(s-1, d+1)
    memo[(s, d)] = ans
    return ans

if __name__ == "__main__":
    steps = 3
    bound = 2 
    num_paths_out = compute_number_paths_out(steps, bound)
    print(num_paths_out)
    #counting(steps, bound)
    #markov_processes()
    # b_1d = (float("-inf"), 7)
    # trials = 1
    # steps = 10
    # run_sim(trials, steps, b_1d)