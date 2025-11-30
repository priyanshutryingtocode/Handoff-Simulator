import numpy as np
import matplotlib.pyplot as plt

# Part 1

def config():
    return {
       
        'bs_a_pos': np.array([0.0, 0.0]),
        'bs_b_pos': np.array([2000.0, 0.0]),
        'start_pos': np.array([0.0, 200.0]),
        'end_pos':   np.array([2000.0, 200.0]),
        
        'velocity': 20.0,      # m/s
        'tx_power': 40.0,      # dBm
        'n': 2.8,              # Path loss exponent
        'd0': 1.0,             # Reference distance
        'freq': 2.4e9,         # 2.4 GHz
        'shadow_std': 0.5,     # Noise (Standard Deviation in dB)
        
        'hysteresis': 5.0,     # dB
        'dt': 0.5              # Time step
    }

# Part 2

def get_rss(dist, cfg):

    d = max(dist, cfg['d0'])
    
    c = 3e8
    wavelength = c / cfg['freq']
    pl_ref = 20 * np.log10((4 * np.pi * cfg['d0']) / wavelength)

    pl_dist = 10 * cfg['n'] * np.log10(d / cfg['d0'])
    
    shadowing = np.random.normal(0, cfg['shadow_std'])
    
    path_loss = pl_ref + pl_dist + shadowing
    return cfg['tx_power'] - path_loss

# Part 3

def run_simulation(cfg):

    path_vec = cfg['end_pos'] - cfg['start_pos']
    total_dist = np.linalg.norm(path_vec)
    duration = total_dist / cfg['velocity']
    time_steps = np.arange(0, duration, cfg['dt'])
    
    positions = [cfg['start_pos'] + t * cfg['velocity'] * (path_vec / total_dist) for t in time_steps]
    
    data = {
        'x': [], 'rss_a': [], 'rss_b': [], 'diff': [], 
        'cell': [], 'handoff_locs': []
    }
    
    cfg_init = cfg.copy(); cfg_init['shadow_std'] = 0
    d_a_init = np.linalg.norm(positions[0] - cfg['bs_a_pos'])
    d_b_init = np.linalg.norm(positions[0] - cfg['bs_b_pos'])
    serving = 'A' if get_rss(d_a_init, cfg_init) > get_rss(d_b_init, cfg_init) else 'B'
    
    # Main Loop
    for pos in positions:

        da = np.linalg.norm(pos - cfg['bs_a_pos'])
        db = np.linalg.norm(pos - cfg['bs_b_pos'])

        val_a = get_rss(da, cfg)
        val_b = get_rss(db, cfg)
        
        margin = cfg['hysteresis']
        
        if serving == 'A' and val_b > val_a + margin:
            serving = 'B'
            data['handoff_locs'].append(pos[0])
            print(f"Handoff A->B at x={pos[0]:.1f}m")
            
        elif serving == 'B' and val_a > val_b + margin:
            serving = 'A'
            data['handoff_locs'].append(pos[0])
            print(f"Handoff B->A at x={pos[0]:.1f}m")
            
        # Log Data
        data['x'].append(pos[0])
        data['rss_a'].append(val_a)
        data['rss_b'].append(val_b)
        data['diff'].append(val_b - val_a)
        data['cell'].append(1 if serving == 'A' else 2)
        
    return data

# Part 4
def plot_results(data, cfg):
    x = data['x']
    h = cfg['hysteresis']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: RSS Levels
    ax1.plot(x, data['rss_a'], 'b', alpha=0.6, lw=1.5, label='RSS Cell A')
    ax1.plot(x, data['rss_b'], 'r', alpha=0.6, lw=1.5, label='RSS Cell B')
    ax1.set_ylabel('RSS (dBm)', fontsize=10)
    ax1.set_title('1. Signal Strength (with Shadowing)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Decision Metric
    ax2.plot(x, data['diff'], 'purple', lw=1.5, label='RSS_B - RSS_A')
    ax2.fill_between(x, -h, h, color='gray', alpha=0.15, label='Hysteresis Band')
    ax2.axhline(h, c='r', ls=':', lw=1.5, label=f'Threshold (+{h}dB)')
    ax2.axhline(-h, c='b', ls=':', lw=1.5, label=f'Threshold (-{h}dB)')
    ax2.axhline(0, c='k', lw=0.5)
    ax2.set_ylabel('Diff (dB)', fontsize=10)
    ax2.set_title('2. Handoff Decision Logic', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Serving Cell
    ax3.step(x, data['cell'], where='post', c='k', lw=2)
    ax3.set_yticks([1, 2])
    ax3.set_yticklabels(['Cell A', 'Cell B'], fontsize=10)
    ax3.set_ylabel('Serving Cell', fontsize=10)
    ax3.set_title('3. Connection Status', fontsize=12, fontweight='bold')
    ax3.set_xlabel('User Position X (m)', fontsize=12)
    
    handoff_label_added = False
    for ho_x in data['handoff_locs']:
        label = 'Handoff Event' if not handoff_label_added else None
        
        for ax in [ax1, ax2, ax3]:
            lbl = label if ax == ax1 else None
            ax.axvline(ho_x, c='g', ls='--', alpha=0.8, lw=1.5, label=lbl)
            
        handoff_label_added = True

    ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    
    config = config()
    results = run_simulation(config)
    plot_results(results, config)