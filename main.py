import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="Handoff Simulator", layout="wide", page_icon="📡")

st.markdown("""
<style>
    /* 1. Dark Mode Dashboard Cards */
    div[data-testid="stMetric"] {
        background-color: #262730; /* Dark Grey Background */
        border: 1px solid #464b5f;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] > label {
        color: #d0d0d0 !important; /* Dimmed label text */
    }
    
    /* 2. UAV Status Badges */
    .badge-blue {
        background-color: #2980b9;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-red {
        background-color: #c0392b;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }

    /* 3. General Typography */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 4. Sidebar spacing */
    .css-1d391kg {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

ENV_PRESETS = {
    "Free Space (Line of Sight)": 2.0,
    "Open Rural Area": 2.5,
    "Suburban Area": 3.0,
    "Dense Urban City": 4.0
}


# LOGIC: SYSTEM 1 (STATIC HIGHWAY)


def run_static_sim(cfg):
    path_vec = cfg['end_pos'] - cfg['start_pos']
    dist_total = np.linalg.norm(path_vec)
    positions = np.linspace(0, dist_total, int(dist_total/cfg['step_size']))
    
    x_vals, rss_a, rss_b, serving_cell = [], [], [], []
    handoff_locs = []
    
    curr_serving = 'A'
    
    for x in positions:
        pos = cfg['start_pos'] + (x/dist_total)*path_vec

        da = np.linalg.norm(pos - cfg['bs_a_pos'])
        db = np.linalg.norm(pos - cfg['bs_b_pos'])

        def get_dbm(d):
            pl = 20*np.log10(4*np.pi*1.0/(3e8/2.4e9)) + 10*cfg['n']*np.log10(max(d,1)/1.0)
            return cfg['tx_power'] - pl + np.random.normal(0, cfg['shadow_std'])
            
        val_a = get_dbm(da)
        val_b = get_dbm(db)
        
        if curr_serving == 'A' and val_b > val_a + cfg['hysteresis']:
            curr_serving = 'B'; handoff_locs.append(x)
        elif curr_serving == 'B' and val_a > val_b + cfg['hysteresis']:
            curr_serving = 'A'; handoff_locs.append(x)
            
        x_vals.append(x); rss_a.append(val_a); rss_b.append(val_b)
        serving_cell.append(1 if curr_serving == 'A' else 2)
        
    return pd.DataFrame({'Distance': x_vals, 'RSS_A': rss_a, 'RSS_B': rss_b, 'Serving': serving_cell}), handoff_locs

# LOGIC: SYSTEM 2 (UAV SWARM)

def run_uav_sim_data(cfg):
    steps = int(cfg['duration'] / cfg['dt'])
    t_vals = np.linspace(0, 1, steps)
    user_vec = cfg['end_pos'] - cfg['start_pos']
    user_path = cfg['start_pos'] + np.outer(t_vals, user_vec)
    
    ua_pos = np.array([0.0, 0.0, 100.0])
    ub_pos = np.array([500.0, 0.0, 100.0])
    
    data = {"ua_x": [], "ua_y": [], "ub_x": [], "ub_y": [], "u_x": [], "u_y": [], "color": []}
    
    curr_serving = 'A'
    handoff_count = 0
    
    for i, u_pos in enumerate(user_path):

        ua_pos[:2] += np.random.normal(0, cfg['uav_jitter'], 2)
        ub_pos[:2] += np.random.normal(0, cfg['uav_jitter'], 2)
        
        dist_a = np.linalg.norm(u_pos - ua_pos)
        dist_b = np.linalg.norm(u_pos - ub_pos)
        pl_const = 20*np.log10(4*np.pi*1.0/(3e8/2.4e9))
        rss_a = cfg['tx_power'] - (pl_const + 10*cfg['n']*np.log10(max(dist_a,1)) + np.random.normal(0, cfg['noise']))
        rss_b = cfg['tx_power'] - (pl_const + 10*cfg['n']*np.log10(max(dist_b,1)) + np.random.normal(0, cfg['noise']))
        
        if curr_serving == 'A' and rss_b > rss_a + cfg['hysteresis']:
            curr_serving = 'B'; handoff_count += 1
        elif curr_serving == 'B' and rss_a > rss_b + cfg['hysteresis']:
            curr_serving = 'A'; handoff_count += 1
            
        data['ua_x'].append(ua_pos[0]); data['ua_y'].append(ua_pos[1])
        data['ub_x'].append(ub_pos[0]); data['ub_y'].append(ub_pos[1])
        data['u_x'].append(u_pos[0]); data['u_y'].append(u_pos[1])
        data['color'].append('blue' if curr_serving == 'A' else 'red')

    return pd.DataFrame(data), handoff_count

# PAGE 1: HIGHWAY RENDERER

def render_static_page():
    st.header("🛣️ Highway Handoff Analytics")
    st.markdown("Analysis of signal stability for a user moving linearly between two fixed towers.")
    st.divider()
    
    # Sidebar 
    with st.sidebar:
        st.subheader("🌍 Environment")
        env_type = st.selectbox("Terrain Preset", list(ENV_PRESETS.keys()), index=2)
        n_val = ENV_PRESETS[env_type]
        st.caption(f"Path Loss Exponent (n) = {n_val}")
        
        st.subheader("📡 Network Config")
        hyst = st.slider("Hysteresis Margin (dB)", 0.0, 10.0, 5.0)
        noise = st.slider("Signal Noise (dB)", 0.0, 8.0, 2.0)
        
        st.subheader("🚗 Mobility")
        step_size = st.slider("Resolution (m)", 1, 10, 1)

    # Execution
    cfg = {
        'bs_a_pos': np.array([0,0]), 'bs_b_pos': np.array([2000,0]),
        'start_pos': np.array([0,0]), 'end_pos': np.array([2000,0]),
        'n': n_val, 'hysteresis': hyst, 'shadow_std': noise, 'tx_power': 40, 'step_size': step_size
    }
    
    df, handoffs = run_static_sim(cfg)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Handoff Events", len(handoffs))
    col2.metric("Mean RSS (Tower A)", f"{df['RSS_A'].mean():.1f} dBm")
    col3.metric("Mean RSS (Tower B)", f"{df['RSS_B'].mean():.1f} dBm")
    col4.metric("Avg Signal Quality", f"{df[['RSS_A', 'RSS_B']].max(axis=1).mean():.1f} dBm")
    
    st.markdown("---")

    # Main Dashboard Plot
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Signal Strength & Noise", "Handoff Decision Logic", "Serving Cell Status"),
                        row_heights=[0.5, 0.25, 0.25])

    # Plot 1: RSS
    fig.add_trace(go.Scatter(x=df['Distance'], y=df['RSS_A'], name='Tower A', line=dict(color='#2980b9', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Distance'], y=df['RSS_B'], name='Tower B', line=dict(color='#c0392b', width=1)), row=1, col=1)
    
    # Plot 2: Differential
    diff = df['RSS_B'] - df['RSS_A']
    fig.add_trace(go.Scatter(x=df['Distance'], y=diff, name='Delta (B-A)', line=dict(color='#8e44ad', width=1.5)), row=2, col=1)
    # Hysteresis Bands
    fig.add_hrect(y0=-hyst, y1=hyst, row=2, col=1, fillcolor="gray", opacity=0.1, line_width=0, annotation_text="Hysteresis", annotation_position="top right")

    # Plot 3: Connection State
    fig.add_trace(go.Scatter(x=df['Distance'], y=df['Serving'], name='Connection', 
                             line_shape='hv', line=dict(color='#2c3e50', width=3)), row=3, col=1)

    # Handoff Markers
    for hx in handoffs:
        fig.add_vline(x=hx, line_dash="dash", line_color="#27ae60", opacity=0.8)

    fig.update_layout(height=800, hovermode="x unified", template="plotly_white")
    fig.update_yaxes(title_text="RSS (dBm)", row=1, col=1)
    fig.update_yaxes(title_text="Diff (dB)", row=2, col=1)
    fig.update_yaxes(tickvals=[1,2], ticktext=["Tower A", "Tower B"], row=3, col=1)
    fig.update_xaxes(title_text="Distance (meters)", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# PAGE 2: UAV RENDERER 

def render_uav_page():
    st.header("🚁 Drone Swarm Topography")
    st.markdown("Real-time spatial tracking of user connectivity in a dynamic UAV mesh network.")
    st.divider()

    # Sidebar Controls
    with st.sidebar:
        st.subheader("Controls")
        
        with st.expander("Mobility Parameters", expanded=True):
            uav_jitter = st.slider("Drone Stability", 0.0, 5.0, 2.0, 
                                 help="Higher values mean drones drift more (e.g., high wind).")
            
        with st.expander("Radio Environment", expanded=True):
            hyst = st.slider("Hysteresis (dB)", 0.0, 10.0, 5.0)
            noise = st.slider("RF Noise Floor", 0.0, 5.0, 1.0)
        
        if st.button("🔄 Regenerate Simulation", type="primary", use_container_width=True):
            st.session_state['uav_seed'] = np.random.randint(0,1000) 

    # Execution
    cfg = {
        'start_pos': np.array([0,-300,0]), 'end_pos': np.array([500,300,0]),
        'duration': 40, 'dt': 0.1, 'uav_jitter': uav_jitter, 
        'n': 2.5, 'tx_power': 30, 'hysteresis': hyst, 'noise': noise
    }
    
    df, count = run_uav_sim_data(cfg)
    
    # HUD Metrics (Top Row)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Handoffs", count, delta="Events")
    c2.metric("Network Nodes", "2 UAVs + 1 User")
    c3.metric("Avg Drone Drift", f"±{uav_jitter} m")
    c4.metric("Sim Duration", "40.0 s")

    # Main Visualization
    col_main, col_legend = st.columns([4, 1])
    
    with col_main:
        fig = go.Figure()
        
        # 1. UAV Clouds 
        fig.add_trace(go.Scatter(x=df['ua_x'], y=df['ua_y'], mode='markers', 
                               marker=dict(color='#3498db', size=3, opacity=0.15), name='Drone A Path'))
        fig.add_trace(go.Scatter(x=df['ub_x'], y=df['ub_y'], mode='markers', 
                               marker=dict(color='#e74c3c', size=3, opacity=0.15), name='Drone B Path'))
        
        # 2. User Path 
        mask_a = df['color'] == 'blue'
        fig.add_trace(go.Scatter(x=df[mask_a]['u_x'], y=df[mask_a]['u_y'], mode='markers', 
                               marker=dict(color='#2980b9', size=6), name='Connected to A'))
        
        mask_b = df['color'] == 'red'
        fig.add_trace(go.Scatter(x=df[mask_b]['u_x'], y=df[mask_b]['u_y'], mode='markers', 
                               marker=dict(color='#c0392b', size=6), name='Connected to B'))

        # 3. Anchors 
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', 
                               marker=dict(color='#3498db', symbol='cross-thin', size=12, line=dict(width=2)), 
                               name='Anchor A'))
        fig.add_trace(go.Scatter(x=[500], y=[0], mode='markers', 
                               marker=dict(color='#e74c3c', symbol='cross-thin', size=12, line=dict(width=2)), 
                               name='Anchor B'))

        fig.update_layout(
            height=600,
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="East-West Position (m)", gridcolor='#444'),
            yaxis=dict(title="North-South Position (m)", gridcolor='#444', scaleanchor="x", scaleratio=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_legend:
        st.markdown("### 📡 Live Feed")
        st.caption("Connection Status Stream")
        
        st.markdown(f"""
        <div style="background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #464b5f;">
            <div style="margin-bottom: 10px;">
                <span class="badge-blue">UAV A</span><br>
                <small style="color: #bbb;">Primary Node</small>
            </div>
            <div style="margin-bottom: 10px;">
                <span class="badge-red">UAV B</span><br>
                <small style="color: #bbb;">Secondary Node</small>
            </div>
            <hr style="border-color: #444;">
            <div style="margin-top: 10px;">
                <b>System Status:</b><br>
                <span style="color: #2ecc71;">● Online</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if count > 0:
            st.markdown("#### Recent Events")
            st.markdown(f"⚠️ **{count} Handoffs** detected during flight path.")


# MAIN APP ROUTER

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module", ["Highway (Static)", "Drone Swarm (Dynamic)"])

if page == "Highway (Static)":
    render_static_page()
else:
    render_uav_page()