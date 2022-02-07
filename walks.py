import random as r


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



def run_sim(trials, steps, boundary):
    """
    trials: Number of trials to run in the simulation
    steps: Number of steps to run per trial
    """
    for trial in range(trials):
        print(run_1d_trial(steps, boundary))


if __name__ == "__main__":
    b_1d = (float("-inf"), 7)
    trials = 1
    steps = 10
    run_sim(trials, steps, b_1d)