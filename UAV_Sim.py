import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Part 1

CONFIG = {

    'uav_a': {
        'anchor': np.array([0, 0]),
        'radius': 200,      
        'altitude': 120,    
        'speed': 15         
    },
    
    'uav_b': {
        'anchor': np.array([500, 0]),
        'radius': 200,      
        'altitude': 120,    
        'speed': 15         
    },
    
    'user': {
        'start': np.array([0, -300, 0]),
        'end':   np.array([500, 300, 0]),
        'velocity': 5       
    },

    'radio': {
        'tx_power': 30,         
        'path_loss_n': 2.5,     
        'hysteresis': 5.0,      
        'ref_distance': 1.0,    
        'frequency_hz': 2.4e9,  
        'shadowing_std_db': 0.1 
    },

    'time_step': 0.5        
}


# Part 2

def get_rss(pos1, pos2, tx_power, n, d0, frequency, shadow_std):
  
    dist = np.linalg.norm(pos1 - pos2)
    dist = max(dist, d0)

    c = 3e8 
    wavelength = c / frequency

    pl_ref = 20 * np.log10((4 * np.pi * d0) / wavelength)

    dist_loss = 10 * n * np.log10(dist / d0)

    shadowing = np.random.normal(0, shadow_std)

    path_loss = pl_ref + dist_loss + shadowing
    
    return tx_power - path_loss

def random_direction():

    theta = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(theta), np.sin(theta), 0])

def update_uav_position(pos, direction, anchor, radius, speed, dt):
    
    next_pos = pos + direction * speed * dt
    
    dist_from_anchor = np.linalg.norm(next_pos[:2] - anchor)
    
    if dist_from_anchor > radius:

        to_center = anchor - pos[:2]
        to_center = to_center / np.linalg.norm(to_center) 
        
        new_dir_2d = to_center + np.random.uniform(-0.5, 0.5, 2)
        new_dir_2d = new_dir_2d / np.linalg.norm(new_dir_2d)
        
        return pos, np.array([new_dir_2d[0], new_dir_2d[1], 0])
    
    if np.random.rand() < 0.05:
        return next_pos, random_direction()
        
    return next_pos, direction

# Part 3

def run_simulation(cfg):

    u_start, u_end = cfg['user']['start'], cfg['user']['end']
    u_vel = cfg['user']['velocity']
    dt = cfg['time_step']
    
    path_vec = u_end - u_start
    total_dist = np.linalg.norm(path_vec)
    total_time = total_dist / u_vel
    steps = int(total_time / dt)
    
    t_vals = np.linspace(0, 1, steps)
    user_path = u_start + np.outer(t_vals, path_vec)

    ua_pos = np.append(cfg['uav_a']['anchor'], cfg['uav_a']['altitude'])
    ub_pos = np.append(cfg['uav_b']['anchor'], cfg['uav_b']['altitude'])
    ua_dir, ub_dir = random_direction(), random_direction()

    logs = {
        'rss_a': [], 'rss_b': [], 'rss_diff': [], 
        'serving': [], 'handoffs': [], 
        'ua_path': [], 'ub_path': []
    }

    r_conf = cfg['radio']
    
    rss_a_init = get_rss(user_path[0], ua_pos, r_conf['tx_power'], r_conf['path_loss_n'], 
                         r_conf['ref_distance'], r_conf['frequency_hz'], 0)
    rss_b_init = get_rss(user_path[0], ub_pos, r_conf['tx_power'], r_conf['path_loss_n'], 
                         r_conf['ref_distance'], r_conf['frequency_hz'], 0)
    
    serving = 'A' if rss_a_init > rss_b_init else 'B'

    for u_pos in user_path:

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

        rss_a = get_rss(u_pos, ua_pos, r_conf['tx_power'], r_conf['path_loss_n'], 
                        r_conf['ref_distance'], r_conf['frequency_hz'], r_conf['shadowing_std_db'])
        rss_b = get_rss(u_pos, ub_pos, r_conf['tx_power'], r_conf['path_loss_n'], 
                        r_conf['ref_distance'], r_conf['frequency_hz'], r_conf['shadowing_std_db'])
        
        hyst = r_conf['hysteresis']
        
        if serving == 'A' and rss_b > rss_a + hyst:
            serving = 'B'
            logs['handoffs'].append((u_pos.copy(), rss_b))
        elif serving == 'B' and rss_a > rss_b + hyst:
            serving = 'A'
            logs['handoffs'].append((u_pos.copy(), rss_a))

        logs['rss_a'].append(rss_a)
        logs['rss_b'].append(rss_b)
        logs['rss_diff'].append(rss_b - rss_a)
        logs['serving'].append(1 if serving == 'A' else 2)

    return user_path, logs

# Part 4 

def plot_time_series(user_path, logs, cfg):
    
    x_axis = user_path[:, 0] 
    hyst = cfg['radio']['hysteresis']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: RSS Levels
    ax = axes[0]
    ax.plot(x_axis, logs['rss_a'], 'b-', label='RSS UAV-A', alpha=0.6, lw=1)
    ax.plot(x_axis, logs['rss_b'], 'r-', label='RSS UAV-B', alpha=0.6, lw=1)
    ax.set_ylabel('RSS (dBm)')
    ax.set_title('Signal Strength with Shadowing (UAV Network)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Plot 2: RSS Difference
    ax = axes[1]
    ax.plot(x_axis, logs['rss_diff'], 'k-', label='RSS_B - RSS_A', lw=1, alpha=0.8)
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

    for i, ax in enumerate(axes):
        first_ho = True
        for pos, _ in logs['handoffs']:
            label = 'Handoff' if (first_ho and i==0) else None
            ax.axvline(pos[0], color='green', ls='--', alpha=0.6, label=label)
            first_ho = False
    
    if len(logs['handoffs']) > 0:
        axes[0].legend()

    plt.tight_layout()
    plt.show()

def plot_spatial_map(user_path, logs, cfg):

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    c_a = Circle(cfg['uav_a']['anchor'], cfg['uav_a']['radius'], 
                 color='blue', alpha=0.1, label='Zone A')
    c_b = Circle(cfg['uav_b']['anchor'], cfg['uav_b']['radius'], 
                 color='red', alpha=0.1, label='Zone B')
    ax.add_patch(c_a)
    ax.add_patch(c_b)

    ua_path = np.array(logs['ua_path'])
    ub_path = np.array(logs['ub_path'])
    plt.plot(ua_path[:, 0], ua_path[:, 1], 'b:', alpha=0.5, lw=1, label='UAV-A Path')
    plt.plot(ub_path[:, 0], ub_path[:, 1], 'r:', alpha=0.5, lw=1, label='UAV-B Path')

    serving_arr = np.array(logs['serving'])
    
    mask_a = serving_arr == 1
    plt.scatter(user_path[mask_a, 0], user_path[mask_a, 1], 
                c='blue', s=10, label='User (on A)')
    
    mask_b = serving_arr == 2
    plt.scatter(user_path[mask_b, 0], user_path[mask_b, 1], 
                c='red', s=10, label='User (on B)')

    if logs['handoffs']:
        ho_coords = np.array([x[0] for x in logs['handoffs']])
        plt.scatter(ho_coords[:, 0], ho_coords[:, 1], 
                    c='lime', edgecolors='black', s=100, zorder=10, label='Handoff Event')

    plt.title('Spatial Map: UAV Handoff with Shadowing')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.axis('equal')
    plt.grid(True, ls='--', alpha=0.5)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.tight_layout()
    plt.show()

# Main

if __name__ == "__main__":
    print("Initializing UAV Handoff Simulation (with Shadowing)...")
    
    path_data, log_data = run_simulation(CONFIG)
    
    print(f"Simulation Complete. Total steps: {len(path_data)}")
    print(f"Handoffs occurred: {len(log_data['handoffs'])}")
    
    plot_time_series(path_data, log_data, CONFIG)
    plot_spatial_map(path_data, log_data, CONFIG)