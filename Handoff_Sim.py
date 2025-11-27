import numpy as np
import matplotlib.pyplot as plt

# Part 1

def get_simulation_config():
   
    return {
        # Base Station Locations (meters)
        'bs_a_pos': np.array([0.0, 0.0]),
        'bs_b_pos': np.array([2000.0, 0.0]),

        # User Movement (meters)
        'start_pos': np.array([0.0, 200.0]),
        'end_pos':   np.array([2000.0, 200.0]),
        'user_velocity': 20.0, # m/s

        # Signal Parameters
        'p_tx_dbm': 40.0,      # Transmit power
        'path_loss_n': 2.8,    # Path loss exponent
        'ref_distance': 1.0,
        
        # New Parameters for Image Formula
        'frequency_hz': 2.4e9,     # Carrier Frequency (2.4 GHz)
        'shadowing_std_db': 0.1,   # Shadow Factor (SF) - Standard Deviation in dB

        # Handoff Logic
        'hysteresis_margin_db': 5.0, # dB margin required to switch

        # Simulation Fidelity
        'time_step': 0.5       
    }

# Part 2

def calculate_rss(distance, p_tx, n, d0, frequency, shadow_std):
    
    if distance < d0:
        distance = d0
        
    # 1. Calculate Wavelength (lambda = c / f)
    c = 3e8 
    wavelength = c / frequency
    
    # 2. Calculate Reference Path Loss (PL_ref) at d0
    # PL_ref = 20 * log10(4 * pi * d0 / lambda)
    pl_ref = 20 * np.log10((4 * np.pi * d0) / wavelength)
    
    # 3. Calculate Distance Component
    dist_loss = 10 * n * np.log10(distance / d0)
    
    # 4. Calculate Shadowing Component (SF * epsilon)
    shadowing = np.random.normal(0, shadow_std)
    
    # Total Path Loss
    path_loss = pl_ref + dist_loss + shadowing
    
    # Received Power
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
        return None, None
        
    sim_time = path_length / user_velocity
    time_points = np.arange(0, sim_time, time_step)
    # Generate all positions along the line
    user_positions = [start_pos + t * user_velocity * path_vector / path_length for t in time_points]

    logs = {
        'rss_a': [], 
        'rss_b': [], 
        'rss_diff': [], # NEW: Difference between B and A
        'serving_cell': [], 
        'handoffs': []
    }
    
    # Determine initial connection
    initial_dist_a = np.linalg.norm(user_positions[0] - bs_a_pos)
    initial_dist_b = np.linalg.norm(user_positions[0] - bs_b_pos)
    
    rss_a_init = calculate_rss(initial_dist_a, params['p_tx_dbm'], params['path_loss_n'], 
                               params['ref_distance'], params['frequency_hz'], 0)
    rss_b_init = calculate_rss(initial_dist_b, params['p_tx_dbm'], params['path_loss_n'], 
                               params['ref_distance'], params['frequency_hz'], 0)
                               
    serving_cell = 'A' if rss_a_init > rss_b_init else 'B'

    for pos in user_positions:
        dist_a = np.linalg.norm(pos - bs_a_pos)
        dist_b = np.linalg.norm(pos - bs_b_pos)

        # Calculate RSS with Shadowing
        rss_a = calculate_rss(dist_a, params['p_tx_dbm'], params['path_loss_n'], 
                              params['ref_distance'], params['frequency_hz'], params['shadowing_std_db'])
        rss_b = calculate_rss(dist_b, params['p_tx_dbm'], params['path_loss_n'], 
                              params['ref_distance'], params['frequency_hz'], params['shadowing_std_db'])

        # Handoff Logic
        if serving_cell == 'A':
            if rss_b > rss_a + params['hysteresis_margin_db']:
                serving_cell = 'B'
                # Log full position (x, y) for the spatial map
                logs['handoffs'].append({'pos': pos, 'rss': rss_b}) 
                print(f"Handoff A -> B at position x={pos[0]:.2f} m")
        elif serving_cell == 'B':
            if rss_a > rss_b + params['hysteresis_margin_db']:
                serving_cell = 'A'
                logs['handoffs'].append({'pos': pos, 'rss': rss_a})
                print(f"Handoff B -> A at position x={pos[0]:.2f} m")

        logs['rss_a'].append(rss_a)
        logs['rss_b'].append(rss_b)
        logs['rss_diff'].append(rss_b - rss_a) # Log the difference
        logs['serving_cell'].append(1 if serving_cell == 'A' else 2)

    return user_positions, logs

# Part 3

def plot_dashboard(user_positions, logs, params):
    
    if user_positions is None or logs is None:
        return
        
    user_x_positions = [p[0] for p in user_positions]
    hyst = params['hysteresis_margin_db']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot 1
    ax1.plot(user_x_positions, logs['rss_a'], label='RSS Cell A', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(user_x_positions, logs['rss_b'], label='RSS Cell B', color='red', alpha=0.7, linewidth=1)
    ax1.set_ylabel('RSS (dBm)')
    ax1.set_title('1. Received Signal Strength (with Log-Normal Shadowing)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Plot 2
    ax2.plot(user_x_positions, logs['rss_diff'], color='purple', label='RSS_B - RSS_A', alpha=0.8)
    ax2.axhline(y=hyst, color='red', linestyle=':', label=f'Handoff Thresh (A->B): +{hyst}dB')
    ax2.axhline(y=-hyst, color='blue', linestyle=':', label=f'Handoff Thresh (B->A): -{hyst}dB')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Delta RSS (dB)')
    ax2.set_title('2. Handoff Decision Logic (Difference + Hysteresis)')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Mark handoffs
    for ho in logs['handoffs']:
        ax2.axvline(x=ho['pos'][0], color='green', linestyle='--')

    # Plot 3
    ax3.plot(user_x_positions, logs['serving_cell'], 'k.-', drawstyle='steps-post', label='Serving Cell')
    ax3.set_xlabel('User Position along X-axis (meters)')
    ax3.set_ylabel('Cell ID')
    ax3.set_yticks([1, 2])
    ax3.set_yticklabels(['Cell A', 'Cell B'])
    ax3.set_ylim(0.5, 2.5)
    ax3.set_title('3. Active Connection Status')
    
    # Mark handoffs on serving cell plot
    for i, ho in enumerate(logs['handoffs']):
        ax3.axvline(x=ho['pos'][0], color='green', linestyle='--', label='Handoff Event' if i == 0 else "")
    
    ax3.legend()
    plt.tight_layout()
    plt.show()

def plot_spatial_map(user_positions, logs, params):

    if user_positions is None: return

    bs_a = params['bs_a_pos']
    bs_b = params['bs_b_pos']
    
    pos_arr = np.array(user_positions)
    serving_arr = np.array(logs['serving_cell'])

    plt.figure(figsize=(10, 8))
    
    # Plot Base Stations
    plt.plot(bs_a[0], bs_a[1], marker='^', color='blue', markersize=15, label='Base Station A', markeredgecolor='black')
    plt.plot(bs_b[0], bs_b[1], marker='^', color='red', markersize=15, label='Base Station B', markeredgecolor='black')

    # Plot User Path (Color coded by serving cell)

    path_a = pos_arr[serving_arr == 1]

    path_b = pos_arr[serving_arr == 2]

    if len(path_a) > 0:
        plt.scatter(path_a[:, 0], path_a[:, 1], color='blue', s=10, label='User (Connected to A)')
    if len(path_b) > 0:
        plt.scatter(path_b[:, 0], path_b[:, 1], color='red', s=10, label='User (Connected to B)')

    # Plot Handoff Events
    for ho in logs['handoffs']:
        plt.scatter(ho['pos'][0], ho['pos'][1], color='lime', s=150, edgecolors='black', label='Handoff Location', zorder=10)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title('Spatial Map: User Path & Network Coverage')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Main

if __name__ == "__main__":
    
    # 1. Get parameters
    simulation_params = get_simulation_config()
    
    # 2. Run simulation
    positions, results_logs = run_simulation(simulation_params)
    
    # 3. Plotting
    print("\nGenerating Figure 1: Simulation Dashboard...")
    plot_dashboard(positions, results_logs, simulation_params)
    
    print("Generating Figure 2: Spatial Map...")
    plot_spatial_map(positions, results_logs, simulation_params)