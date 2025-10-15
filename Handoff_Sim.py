import numpy as np
import matplotlib.pyplot as plt

#Part 1
#Functions for User Input

def getCoordinate(prompt):
    
    x = float(input(f"Enter {prompt} x-coordinate (meters): "))
    y = float(input(f"Enter {prompt} y-coordinate (meters): "))
    return np.array([x, y])


def getFloat(prompt):

    value = float(input(f"{prompt}"))
    return value


def getParameters():
    
    print("\nEnter simulation parameters")
    
    print("\nBase Station (BS) Setup \n")
    
    bs_a_pos = getCoordinate("BS-A Position")
    bs_b_pos = getCoordinate("BS-B Position")
    
    print("\nUser Mobility Setup \n")
    
    start_pos = getCoordinate("User Start Position")
    end_pos = getCoordinate("User End Position")
    user_velocity = getFloat("Enter user velocity in meters/second (decimal): ")
    
    print("\nSignal Propagation Setup \n")
    
    p_tx_dbm = getFloat("Enter transmit power in dBm (decimal): ")
    path_loss_n = getFloat("Enter path loss exponent (decimal): ")
    
    print("\nHandoff Logic Setup \n")
    
    hysteresis_margin_db = getFloat("Enter hysteresis margin in dB (decimal): ")
    
    print("\nSimulation Time Step \n")
    
    time_step = getFloat("Enter simulation time step in seconds (decimal): ")

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
    
    print("\nAll parameters received. Starting simulation... \n")
    
    return params

#Part 2
#Simulation Logic

def RSS(distance, p_tx, n, d0):
    
    if distance < d0:
        distance = d0
        
    path_loss = 10 * n * np.log10(distance / d0)
    rss = p_tx - path_loss 
    return rss

def Simulate(params):
    
    bs_a_pos = params['bs_a_pos']
    bs_b_pos = params['bs_b_pos']
    start_pos = params['start_pos']
    end_pos = params['end_pos']
    user_velocity = params['user_velocity']
    time_step = params['time_step']

    path_vector = end_pos - start_pos
    path_length = np.linalg.norm(path_vector)
        
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

        rss_a = RSS(dist_a, params['p_tx_dbm'], params['path_loss_n'], params['ref_distance'])
        rss_b = RSS(dist_b, params['p_tx_dbm'], params['path_loss_n'], params['ref_distance'])

        # Handoff logic
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

#Part 3
#Visualization

def plotResults(user_positions, logs):
        
    user_x_positions = [p[0] for p in user_positions]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8)) 

    ax.plot(user_x_positions, logs['serving_cell'], 'k.-', drawstyle='steps-post', label='Serving Cell')
    ax.set_xlabel('User Position along X-axis (meters)')
    ax.set_ylabel('Serving Cell')
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Cell A', 'Cell B'])
    ax.set_ylim(0.5, 2.5)
    ax.set_title('Cell Handoff Events')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for ho in logs['handoffs']:

        ax.axvline(x=ho['pos'], color='green', linestyle='--', label='_nolegend_')
        
        ax.plot(ho['pos'], 1.5, 'go', markersize=10, label=f'Handoff Event at x={ho["pos"]:.0f}m')

    ax.legend()
    plt.show()


#Main

if __name__ == "__main__":
    
    # 1. Get parameters from user
    parameters = getParameters()
    
    # 2. Run the simulation
    positions, results_logs = Simulate(parameters)
    
    # 3. Plot the results
    plotResults(positions, results_logs)