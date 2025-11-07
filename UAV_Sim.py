import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle 

# --- Part 1: Helper & Utility Functions ---

def getCoordinate(prompt):
    while True:
        try:
            x = float(input(f"Enter {prompt} x-coordinate (meters): "))
            y = float(input(f"Enter {prompt} y-coordinate (meters): "))
            return np.array([x, y])
        except ValueError:
            print("Invalid input. Please enter numeric values.")

def getFloat(prompt):
    while True:
        try:
            value = float(input(f"{prompt}"))
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def calculate_3d_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

def get_random_2d_direction():
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle), 0])

def RSS(distance_3d, p_tx, n, d0):
    
    if distance_3d < d0:
        distance_3d = d0
        
    path_loss = 10 * n * np.log10(distance_3d / d0)
    rss = p_tx - path_loss 
    return rss

def update_uav_position(current_pos, current_dir, params, uav_prefix):

    speed = params[f'{uav_prefix}_speed']
    anchor = params[f'{uav_prefix}_anchor']
    radius = params[f'{uav_prefix}_radius']
    
    velocity_vector = current_dir * speed
    next_pos = current_pos + velocity_vector * params['time_step']

    next_pos_2d = np.array([next_pos[0], next_pos[1]])
    dist_from_anchor = np.linalg.norm(next_pos_2d - anchor)
    
    if dist_from_anchor > radius:
        new_dir = get_random_2d_direction()
        return current_pos, new_dir
    else:
        if np.random.rand() < 0.05: # 5% chance to change direction
            new_dir = get_random_2d_direction()
            return next_pos, new_dir
        else:
            return next_pos, current_dir

# --- Part 2: Get Simulation Parameters ---

def getParameters():
    
    print("\nEnter simulation parameters")
    
    print("\n--- UAV-A (BS) Setup 🛰️ ---")
    uav_a_anchor = getCoordinate("UAV-A Anchor Position")
    uav_a_radius = getFloat("Enter UAV-A patrol radius (meters): ")
    uav_a_alt = getFloat("Enter UAV-A altitude (meters): ")
    uav_a_speed = getFloat("Enter UAV-A speed (m/s): ")

    print("\n--- UAV-B (BS) Setup 🛰️ ---")
    uav_b_anchor = getCoordinate("UAV-B Anchor Position")
    uav_b_radius = getFloat("Enter UAV-B patrol radius (meters): ")
    uav_b_alt = getFloat("Enter UAV-B altitude (meters): ")
    uav_b_speed = getFloat("Enter UAV-B speed (m/s): ")
    
    print("\n--- User Mobility Setup 🚶 ---")
    start_pos_2d = getCoordinate("User Start Position")
    end_pos_2d = getCoordinate("User End Position")
    user_velocity = getFloat("Enter user velocity in meters/second (decimal): ")
    start_pos = np.append(start_pos_2d, 0)
    end_pos = np.append(end_pos_2d, 0)
    
    print("\n--- Signal Propagation Setup ---")
    p_tx_dbm = getFloat("Enter transmit power in dBm (decimal): ")
    path_loss_n = getFloat("Enter path loss exponent (decimal): ")
    
    print("\n--- Handoff Logic Setup ---")
    hysteresis_margin_db = getFloat("Enter hysteresis margin in dB (decimal): ")
    
    print("\n--- Simulation Time Step ---")
    time_step = getFloat("Enter simulation time step in seconds (decimal): ")

    params = {
        'uav_a_anchor': uav_a_anchor, 'uav_a_radius': uav_a_radius,
        'uav_a_altitude': uav_a_alt, 'uav_a_speed': uav_a_speed,
        'uav_b_anchor': uav_b_anchor, 'uav_b_radius': uav_b_radius,
        'uav_b_altitude': uav_b_alt, 'uav_b_speed': uav_b_speed,
        'start_pos': start_pos, 'end_pos': end_pos, 'user_velocity': user_velocity,
        'p_tx_dbm': p_tx_dbm, 'path_loss_n': path_loss_n,
        'hysteresis_margin_db': hysteresis_margin_db,
        'time_step': time_step, 'ref_distance': 1.0 
    }
    
    print("\nAll parameters received. Starting simulation... \n")
    return params

# --- Part 3: Simulation Core ---

def Simulate(params):
    
    start_pos = params['start_pos']
    end_pos = params['end_pos']
    user_velocity = params['user_velocity']
    time_step = params['time_step']

    path_vector = end_pos - start_pos
    path_length = np.linalg.norm(path_vector)
    
    if path_length == 0 or user_velocity == 0:
        print("Error: Path length or velocity is zero. Simulation cannot run.")
        return None, None
        
    sim_time = path_length / user_velocity
    time_points = np.arange(0, sim_time, time_step)
    user_positions = [start_pos + t * user_velocity * path_vector / path_length for t in time_points]

    bs_a_pos = np.append(params['uav_a_anchor'], params['uav_a_altitude'])
    bs_b_pos = np.append(params['uav_b_anchor'], params['uav_b_altitude'])
    bs_a_dir = get_random_2d_direction()
    bs_b_dir = get_random_2d_direction()

    logs = {
        'rss_a': [], 'rss_b': [], 'rss_diff': [], 
        'serving_cell': [], 'handoffs': [],
        'bs_a_path': [], 'bs_b_path': []
    }
    
    initial_dist_a = calculate_3d_distance(user_positions[0], bs_a_pos)
    initial_dist_b = calculate_3d_distance(user_positions[0], bs_b_pos)
    serving_cell = 'A' if initial_dist_a < initial_dist_b else 'B'

    for user_pos in user_positions:
        
        bs_a_pos, bs_a_dir = update_uav_position(bs_a_pos, bs_a_dir, params, 'uav_a')
        bs_b_pos, bs_b_dir = update_uav_position(bs_b_pos, bs_b_dir, params, 'uav_b')
        logs['bs_a_path'].append(bs_a_pos)
        logs['bs_b_path'].append(bs_b_pos)

        dist_a = calculate_3d_distance(user_pos, bs_a_pos)
        dist_b = calculate_3d_distance(user_pos, bs_b_pos)

        rss_a = RSS(dist_a, params['p_tx_dbm'], params['path_loss_n'], params['ref_distance'])
        rss_b = RSS(dist_b, params['p_tx_dbm'], params['path_loss_n'], params['ref_distance'])

        if serving_cell == 'A':
            if rss_b > rss_a + params['hysteresis_margin_db']:
                serving_cell = 'B'
                logs['handoffs'].append({'pos': user_pos, 'rss': rss_b})
                print(f"Handoff A -> B at user pos x={user_pos[0]:.2f}, y={user_pos[1]:.2f} m")
                
        elif serving_cell == 'B':
            if rss_a > rss_b + params['hysteresis_margin_db']:
                serving_cell = 'A'
                logs['handoffs'].append({'pos': user_pos, 'rss': rss_a})
                print(f"Handoff B -> A at user pos x={user_pos[0]:.2f}, y={user_pos[1]:.2f} m")

        logs['rss_a'].append(rss_a)
        logs['rss_b'].append(rss_b)
        logs['rss_diff'].append(rss_b - rss_a) 
        logs['serving_cell'].append(1 if serving_cell == 'A' else 2)

    return user_positions, logs

# --- Part 4: Visualization ---

def plotResults(user_positions, logs, params):
    
    if user_positions is None: return
        
    user_x_positions = [p[0] for p in user_positions]
    hysteresis = params['hysteresis_margin_db']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    ax1.plot(user_x_positions, logs['rss_a'], label='RSS from UAV-A', color='blue')
    ax1.plot(user_x_positions, logs['rss_b'], label='RSS from UAV-B', color='red')
    ax1.set_ylabel('RSS (dBm) 📶')
    ax1.set_title('Handoff Simulation Analysis (UAV Base Stations)')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    for ho in logs['handoffs']:
        ax1.axvline(x=ho['pos'][0], color='green', linestyle='--')
        ax1.plot(ho['pos'][0], ho['rss'], 'go', markersize=10, 
                 label=f'Handoff Event at x={ho["pos"][0]:.0f}m')
    ax1.legend()

    ax2.plot(user_x_positions, logs['rss_diff'], label='RSS_B - RSS_A', color='purple')
    ax2.axhline(y=hysteresis, color='red', linestyle=':', 
                label=f'Handoff Threshold (A->B) = {hysteresis} dB')
    ax2.axhline(y=-hysteresis, color='blue', linestyle=':', 
                label=f'Handoff Threshold (B->A) = {-hysteresis} dB')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_ylabel('RSS Difference (dB)')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    for ho in logs['handoffs']:
        ax2.axvline(x=ho['pos'][0], color='green', linestyle='--')
    ax2.legend()
    
    ax3.plot(user_x_positions, logs['serving_cell'], 'k.-', drawstyle='steps-post', label='Serving Cell')
    ax3.set_xlabel('User Position along X-axis (meters)')
    ax3.set_ylabel('Serving Cell')
    ax3.set_yticks([1, 2])
    ax3.set_yticklabels(['UAV-A', 'UAV-B'])
    ax3.set_ylim(0.5, 2.5)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    for ho in logs['handoffs']:
        ax3.axvline(x=ho['pos'][0], color='green', linestyle='--', label='_nolegend_')
        ax3.plot(ho['pos'][0], 1.5, 'go', markersize=10, 
                 label=f'Handoff at x={ho["pos"][0]:.0f}m')
    ax3.legend(loc='best')
    
    plt.tight_layout()
    plt.show()


def plotSpatialMap(params, user_positions, logs):
    if user_positions is None: return
        
    positions = np.array(user_positions)
    serving_cell = np.array(logs['serving_cell'])
    cell_a_user_path = positions[serving_cell == 1][:, 0:2]
    cell_b_user_path = positions[serving_cell == 2][:, 0:2]
    
    uav_a_path_np = np.array(logs['bs_a_path'])[:, 0:2]
    uav_b_path_np = np.array(logs['bs_b_path'])[:, 0:2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if len(cell_a_user_path) > 0:
        ax.plot(cell_a_user_path[:, 0], cell_a_user_path[:, 1], 'b.', 
                markersize=2, label='User (Served by A)')
    if len(cell_b_user_path) > 0:
        ax.plot(cell_b_user_path[:, 0], cell_b_user_path[:, 1], 'r.', 
                markersize=2, label='User (Served by B)')

    # UAV-A
    anchor_a = params['uav_a_anchor']
    radius_a = params['uav_a_radius']
    ax.plot(anchor_a[0], anchor_a[1], 'bx', markersize=10, label='UAV-A Anchor')
    patrol_zone_a = Circle(anchor_a, radius_a, color='blue', fill=False, linestyle='--')
    ax.add_patch(patrol_zone_a)
    
    # UAV-B
    anchor_b = params['uav_b_anchor']
    radius_b = params['uav_b_radius']
    ax.plot(anchor_b[0], anchor_b[1], 'rx', markersize=10, label='UAV-B Anchor')
    patrol_zone_b = Circle(anchor_b, radius_b, color='red', fill=False, linestyle='--')
    ax.add_patch(patrol_zone_b)

    ax.plot(uav_a_path_np[:, 0], uav_a_path_np[:, 1], 'b:', 
            linewidth=0.5, label='UAV-A Path')
    ax.plot(uav_b_path_np[:, 0], uav_b_path_np[:, 1], 'r:', 
            linewidth=0.5, label='UAV-B Path')
            
    for ho in logs['handoffs']:
        ho_pos = ho['pos'] # This is the [x, y, z] of the *user*
        ax.plot(ho_pos[0], ho_pos[1], 'go', markersize=12, 
                label=f'Handoff at ({ho_pos[0]:.0f}, {ho_pos[1]:.0f})m', 
                markeredgecolor='black')

    ax.set_xlabel('X-coordinate (meters)')
    ax.set_ylabel('Y-coordinate (meters)')
    ax.set_title('Spatial Map of UAV Handoff 🛰️🚶')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box') # Makes X and Y scales equal
    
    plt.show()

# --- Part 5: Main Execution ---

if __name__ == "__main__":
    
    parameters = getParameters()
    
    positions, results_logs = Simulate(parameters)
    
    if positions:
        print("\nDisplaying Figure 1: Time-Series Analysis...")
        plotResults(positions, results_logs, parameters)
        
        print("Displaying Figure 2: Spatial Map Analysis...")
        plotSpatialMap(parameters, positions, results_logs)
    else:
        print("Simulation did not run. No results to plot.")