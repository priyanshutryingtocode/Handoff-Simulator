import numpy as np
import matplotlib.pyplot as plt

# Part 1

def get_coordinate(prompt_name):

    while True:
        try:
            x = float(input(f"Enter {prompt_name} x-coordinate (meters): "))
            y = float(input(f"Enter {prompt_name} y-coordinate (meters): "))
            return np.array([x, y])
        except ValueError:
            print("Invalid input. Please enter numeric values for coordinates.")

def get_float_value(prompt, example):

    while True:
        try:
            value = float(input(f"{prompt} (e.g., {example}): "))
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def get_user_parameters():

    print("Provide simulation parameters")
    
    print("\nBase Station (BS) Setup")
    bs_a_pos = get_coordinate("BS-A Position")
    bs_b_pos = get_coordinate("BS-B Position")
    
    print("\nUser Mobility Setup")
    start_pos = get_coordinate("User Start Position")
    end_pos = get_coordinate("User End Position")
    user_velocity = get_float_value("Enter user velocity in meters/second", 15)
    
    print("\nSignal Propagation Setup")
    p_tx_dbm = get_float_value("Enter transmit power in dBm", 40.0)
    path_loss_n = get_float_value("Enter path loss exponent", 2.8)
    
    print("\nHandoff Logic Setup")
    hysteresis_margin_db = get_float_value("Enter hysteresis margin in dB", 3.0)
    
    print("\nSimulation Control")
    time_step = get_float_value("Enter simulation time step in seconds", 0.5)

    params = {
        'bs_a_pos': bs_a_pos,
        'bs_b_pos': bs_b_pos,
        'start_pos': start_pos,
        'end_pos': end_pos,
        'user_velocity': user_velocity,
        'p_tx_dbm': p_tx_dbm,
        'path_loss_n': path_loss_n,
        'hysteresis_margin_db': hysteresis_margin_db,
        'time_step': time_step,
        'ref_distance': 1.0 
    }
    print("\nSimulating...")
    return params

# Part 2

def calculate_rss(distance, p_tx, n, d0):

    if distance < d0:
        distance = d0
    path_loss = 10 * n * np.log10(distance / d0)
    rss = p_tx - path_loss
    return rss

def run_simulation(params):

    bs_a_pos = params['bs_a_pos']
    bs_b_pos = params['bs_b_pos']
    start_pos = params['start_pos']
    end_pos = params['end_pos']
    user_velocity = params['user_velocity']
    time_step = params['time_step']

    path_vector = end_pos - start_pos
    path_length = np.linalg.norm(path_vector)
    if path_length == 0 or user_velocity == 0:
        print("Error: User path length or velocity is zero. Cannot simulate.")
        return None
        
    sim_time = path_length / user_velocity
    time_points = np.arange(0, sim_time, time_step)
    user_positions = [start_pos + t * user_velocity * path_vector / path_length for t in time_points]

    logs = {'rss_a': [], 'rss_b': [], 'serving_cell': [], 'handoffs': []}
    
    initial_dist_a = np.linalg.norm(user_positions[0] - bs_a_pos)
    initial_dist_b = np.linalg.norm(user_positions[0] - bs_b_pos)
    serving_cell = 'A' if initial_dist_a < initial_dist_b else 'B'

    for pos in user_positions:
        dist_a = np.linalg.norm(pos - bs_a_pos)
        dist_b = np.linalg.norm(pos - bs_b_pos)

        rss_a = calculate_rss(dist_a, params['p_tx_dbm'], params['path_loss_n'], params['ref_distance'])
        rss_b = calculate_rss(dist_b, params['p_tx_dbm'], params['path_loss_n'], params['ref_distance'])

        if serving_cell == 'A':
            if rss_b > rss_a + params['hysteresis_margin_db']:
                serving_cell = 'B'
                logs['handoffs'].append({'pos': pos[0], 'rss': rss_b})
                print(f"Handoff A -> B at position x={pos[0]:.2f} m")
        elif serving_cell == 'B':
            if rss_a > rss_b + params['hysteresis_margin_db']:
                serving_cell = 'A'
                logs['handoffs'].append({'pos': pos[0], 'rss': rss_a})
                print(f"Handoff B -> A at position x={pos[0]:.2f} m")

        logs['rss_a'].append(rss_a)
        logs['rss_b'].append(rss_b)
        logs['serving_cell'].append(1 if serving_cell == 'A' else 2)

    return user_positions, logs

# Part 3

def plot_results(user_positions, logs):

    if user_positions is None or logs is None:
        return
        
    user_x_positions = [p[0] for p in user_positions]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(user_x_positions, logs['rss_a'], label='RSS from Cell A', color='blue')
    ax1.plot(user_x_positions, logs['rss_b'], label='RSS from Cell B', color='red')
    ax1.set_ylabel('Received Signal Strength (dBm) 📶')
    ax1.set_title('Handoff Simulation based on RSS with Hysteresis')
    
    for ho in logs['handoffs']:
        ax1.axvline(x=ho['pos'], color='green', linestyle='--')
        ax1.plot(ho['pos'], ho['rss'], 'go', markersize=10, label=f'Handoff Event at x={ho["pos"]:.0f}m')
    ax1.legend()
    
    ax2.plot(user_x_positions, logs['serving_cell'], 'k.-', drawstyle='steps-post')
    ax2.set_xlabel('User Position along X-axis (meters)')
    ax2.set_ylabel('Serving Cell')
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['Cell A', 'Cell B'])
    ax2.set_ylim(0.5, 2.5)

    plt.tight_layout()
    plt.show()

# Main 

if __name__ == "__main__":
    
    # 1. Get parameters from the user
    simulation_params = get_user_parameters()
    
    # 2. Run the simulation
    positions, results_logs = run_simulation(simulation_params)
    
    # 3. Plot the results
    plot_results(positions, results_logs)