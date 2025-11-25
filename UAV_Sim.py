import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ==========================================
# 1. CONFIGURATION BLOCK
# ==========================================
CONFIG = {
    # UAV-A (Base Station A)
    'uav_a': {
        'anchor': np.array([0, 0]),
        'radius': 200,      # meters
        'altitude': 120,    # meters
        'speed': 15         # m/s
    },
    
    # UAV-B (Base Station B)
    'uav_b': {
        'anchor': np.array([500, 0]),
        'radius': 200,      # meters
        'altitude': 120,    # meters
        'speed': 15         # m/s
    },

    # User Mobility
    'user': {
        'start': np.array([0, -300, 0]),
        'end':   np.array([500, 300, 0]),
        'velocity': 5       # m/s
    },

    # Radio & Handoff
    'radio': {
        'tx_power': 30,     # dBm
        'path_loss_n': 2.5, # Environment exponent
        'hysteresis': 3.0   # dB
    },

    # Simulation Settings
    'time_step': 0.5        # seconds (Lower = higher resolution)
}

# ==========================================
# 2. PHYSICS & RADIO FUNCTIONS
# ==========================================

def get_rss(pos1, pos2, tx_power, n, d0=1.0):
    """Calculates RSS based on 3D distance and Log-Distance Path Loss."""
    dist = np.linalg.norm(pos1 - pos2)
    dist = max(dist, d0) # Avoid log(0)
    return tx_power - 10 * n * np.log10(dist / d0)

def random_direction():
    """Returns a random 2D unit vector (x, y, 0)."""
    theta = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(theta), np.sin(theta), 0])

def update_uav_position(pos, direction, anchor, radius, speed, dt):
    """
    Updates UAV position. Bounces off the patrol radius boundary.
    Returns: (new_position, new_direction)
    """
    # Propose new position
    next_pos = pos + direction * speed * dt
    
    # Check 2D distance from anchor (ignore altitude for boundary check)
    dist_from_anchor = np.linalg.norm(next_pos[:2] - anchor)
    
    if dist_from_anchor > radius:
        # Hit boundary: Stay put this step, pick new random direction inwards
        # (Simple bounce logic)
        to_center = anchor - pos[:2]
        to_center = to_center / np.linalg.norm(to_center) # Normalize
        
        # Mix vector towards center + random noise for realism
        new_dir_2d = to_center + np.random.uniform(-0.5, 0.5, 2)
        new_dir_2d = new_dir_2d / np.linalg.norm(new_dir_2d)
        
        return pos, np.array([new_dir_2d[0], new_dir_2d[1], 0])
    
    # Randomly change direction occasionally (Brownian-like motion)
    if np.random.rand() < 0.05:
        return next_pos, random_direction()
        
    return next_pos, direction

# ==========================================
# 3. SIMULATION ENGINE
# ==========================================

def run_simulation(cfg):
    # Setup User Path
    u_start, u_end = cfg['user']['start'], cfg['user']['end']
    u_vel = cfg['user']['velocity']
    dt = cfg['time_step']
    
    path_vec = u_end - u_start
    total_dist = np.linalg.norm(path_vec)
    total_time = total_dist / u_vel
    steps = int(total_time / dt)
    
    # Generate all user positions at once (Vectorized)
    t_vals = np.linspace(0, 1, steps)
    user_path = u_start + np.outer(t_vals, path_vec)

    # Initialize UAVs
    ua_pos = np.append(cfg['uav_a']['anchor'], cfg['uav_a']['altitude'])
    ub_pos = np.append(cfg['uav_b']['anchor'], cfg['uav_b']['altitude'])
    ua_dir, ub_dir = random_direction(), random_direction()

    # Data Containers
    logs = {
        'rss_a': [], 'rss_b': [], 'rss_diff': [], 
        'serving': [], 'handoffs': [], 
        'ua_path': [], 'ub_path': []
    }

    # Initial Connection
    d_a = np.linalg.norm(user_path[0] - ua_pos)
    d_b = np.linalg.norm(user_path[0] - ub_pos)
    serving = 'A' if d_a < d_b else 'B'

    # Time Stepping Loop
    for u_pos in user_path:
        # 1. Move UAVs
        ua_pos, ua_dir = update_uav_position(
            ua_pos, ua_dir, cfg['uav_a']['anchor'], 
            cfg['uav_a']['radius'], cfg['uav_a']['speed'], dt
        )
        ub_pos, ub_dir = update_uav_position(
            ub_pos, ub_dir, cfg['uav_b']['anchor'], 
            cfg['uav_b']['radius'], cfg['uav_b']['speed'], dt
        )
        
        logs['ua_path'].append(ua_pos.copy())
        logs['ub_path'].append(ub_pos.copy())

        # 2. Calculate Signals
        rss_a = get_rss(u_pos, ua_pos, cfg['radio']['tx_power'], cfg['radio']['path_loss_n'])
        rss_b = get_rss(u_pos, ub_pos, cfg['radio']['tx_power'], cfg['radio']['path_loss_n'])
        
        # 3. Handoff Decision
        hyst = cfg['radio']['hysteresis']
        
        if serving == 'A' and rss_b > rss_a + hyst:
            serving = 'B'
            logs['handoffs'].append((u_pos.copy(), rss_b))
        elif serving == 'B' and rss_a > rss_b + hyst:
            serving = 'A'
            logs['handoffs'].append((u_pos.copy(), rss_a))

        # 4. Logging
        logs['rss_a'].append(rss_a)
        logs['rss_b'].append(rss_b)
        logs['rss_diff'].append(rss_b - rss_a)
        logs['serving'].append(1 if serving == 'A' else 2)

    return user_path, logs

# ==========================================
# 4. VISUALIZATION FUNCTIONS
# ==========================================

def plot_time_series(user_path, logs, cfg):
    """Plots RSS, RSS Difference, and Serving Cell over time/distance."""
    x_axis = user_path[:, 0] # Use X coordinate for X-axis
    hyst = cfg['radio']['hysteresis']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: RSS Levels
    ax = axes[0]
    ax.plot(x_axis, logs['rss_a'], 'b-', label='RSS UAV-A', alpha=0.8)
    ax.plot(x_axis, logs['rss_b'], 'r-', label='RSS UAV-B', alpha=0.8)
    ax.set_ylabel('RSS (dBm)')
    ax.set_title('Signal Strength & Handoff Analysis')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Plot 2: RSS Difference
    ax = axes[1]
    ax.plot(x_axis, logs['rss_diff'], 'k-', label='RSS_B - RSS_A', lw=1)
    ax.axhline(hyst, color='red', ls=':', label=f'Hyst (+{hyst}dB)')
    ax.axhline(-hyst, color='blue', ls=':', label=f'Hyst (-{hyst}dB)')
    ax.fill_between(x_axis, -hyst, hyst, color='gray', alpha=0.1)
    ax.set_ylabel('Delta RSS (dB)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Plot 3: Serving Cell
    ax = axes[2]
    ax.step(x_axis, logs['serving'], where='post', color='black', lw=2)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['UAV-A', 'UAV-B'])
    ax.set_ylabel('Active Connection')
    ax.set_xlabel('User X Position (m)')
    ax.grid(True, alpha=0.3)

    # Annotate Handoffs on all plots
    for i, ax in enumerate(axes):
        for pos, _ in logs['handoffs']:
            ax.axvline(pos[0], color='green', ls='--', alpha=0.6)
            if i == 0: # Only label on top plot to avoid clutter
                ax.text(pos[0], ax.get_ylim()[0], 'HO', color='green', 
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_spatial_map(user_path, logs, cfg):
    """Plots the 2D Top-Down view of the simulation."""
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 1. Plot Patrol Zones
    c_a = Circle(cfg['uav_a']['anchor'], cfg['uav_a']['radius'], 
                 color='blue', alpha=0.1, label='Zone A')
    c_b = Circle(cfg['uav_b']['anchor'], cfg['uav_b']['radius'], 
                 color='red', alpha=0.1, label='Zone B')
    ax.add_patch(c_a)
    ax.add_patch(c_b)

    # 2. Plot UAV Paths
    ua_path = np.array(logs['ua_path'])
    ub_path = np.array(logs['ub_path'])
    plt.plot(ua_path[:, 0], ua_path[:, 1], 'b:', alpha=0.5, lw=1, label='UAV-A Path')
    plt.plot(ub_path[:, 0], ub_path[:, 1], 'r:', alpha=0.5, lw=1, label='UAV-B Path')

    # 3. Plot User Path (Color coded by serving cell)
    # We scatter plot the points to easily color-code segments
    serving_arr = np.array(logs['serving'])
    
    # Points served by A
    mask_a = serving_arr == 1
    plt.scatter(user_path[mask_a, 0], user_path[mask_a, 1], 
                c='blue', s=10, label='User (on A)')
    
    # Points served by B
    mask_b = serving_arr == 2
    plt.scatter(user_path[mask_b, 0], user_path[mask_b, 1], 
                c='red', s=10, label='User (on B)')

    # 4. Plot Handoff Points
    if logs['handoffs']:
        ho_coords = np.array([x[0] for x in logs['handoffs']])
        plt.scatter(ho_coords[:, 0], ho_coords[:, 1], 
                    c='lime', edgecolors='black', s=100, zorder=10, label='Handoff Event')

    # Formatting
    plt.title('Spatial Map: UAV Handoff Simulation')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.axis('equal') # CRITICAL for map views
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Initializing UAV Handoff Simulation...")
    
    # Run
    path_data, log_data = run_simulation(CONFIG)
    
    # Visualize
    print(f"Simulation Complete. Total steps: {len(path_data)}")
    print(f"Handoffs occurred: {len(log_data['handoffs'])}")
    
    plot_time_series(path_data, log_data, CONFIG)
    plot_spatial_map(path_data, log_data, CONFIG)