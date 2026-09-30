import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import chirp, spectrogram

st.set_page_config(
    page_title="AquaPulse",
    page_icon="🌊",
    layout="wide"
)

FS = 10_000_000

# -----------------------------
# CUSTOM STYLE
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 0;
}
.subtitle {
    font-size: 1.1rem;
    opacity: 0.75;
    margin-top: -8px;
}
.card {
    padding: 18px;
    border: 1px solid rgba(120,120,120,0.25);
    border-radius: 16px;
    margin-bottom: 12px;
}
.small {
    font-size: 0.9rem;
    opacity: 0.75;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# ADAPTATION ENGINE
# -----------------------------
def choose_config(depth, turbidity, temperature, salinity, battery, mission):

    config = {
        "f0": 150000,
        "f1": 350000,
        "duration": 0.002,
        "amplitude": 0.60,
        "waveform": "LFM Chirp",
        "window": "Hann",
        "reasons": []
    }

    if turbidity > 70:
        config["f0"] = 100000
        config["f1"] = 220000
        config["duration"] = 0.003
        config["amplitude"] = 0.80
        config["reasons"].append(
            "High turbidity detected: lower-frequency band selected."
        )

    elif turbidity < 30:
        config["f0"] = 300000
        config["f1"] = 500000
        config["duration"] = 0.0015
        config["amplitude"] = 0.50
        config["reasons"].append(
            "Clear-water condition: higher-frequency band selected for finer detail."
        )

    else:
        config["reasons"].append(
            "Moderate turbidity: balanced frequency range maintained."
        )

    if depth > 300:
        config["f0"] = min(config["f0"], 120000)
        config["f1"] = min(config["f1"], 250000)
        config["duration"] += 0.001
        config["amplitude"] = min(config["amplitude"] + 0.10, 1.0)
        config["reasons"].append(
            "Deep-water operation: longer pulse and lower-frequency operation."
        )

    if mission == "High Resolution":
        config["waveform"] = "LFM Chirp"
        config["f0"] = max(config["f0"], 250000)
        config["f1"] = 500000
        config["duration"] = min(config["duration"], 0.002)
        config["reasons"].append(
            "Mission priority is resolution: wide high-frequency chirp selected."
        )

    elif mission == "Balanced":
        config["waveform"] = "LFM Chirp"
        config["reasons"].append(
            "Balanced mission: range, resolution and energy are jointly considered."
        )

    elif mission == "Long Range":
        config["waveform"] = "Geometric Sweep"
        config["f0"] = 100000
        config["f1"] = min(config["f1"], 250000)
        config["duration"] = max(config["duration"], 0.003)
        config["reasons"].append(
            "Long-range mission: lower-frequency geometric sweep selected."
        )

    elif mission == "Eco":
        config["waveform"] = "Phase-Coded Pulse"
        config["amplitude"] *= 0.60
        config["duration"] *= 0.75
        config["reasons"].append(
            "Eco mission: transmit amplitude and pulse duration reduced."
        )

    if battery < 20:
        config["amplitude"] *= 0.60
        config["duration"] *= 0.70
        config["reasons"].append(
            "Battery below 20%: emergency energy-saving override active."
        )

    config["amplitude"] = float(
        np.clip(config["amplitude"], 0.10, 1.0)
    )

    return config

# -----------------------------
# WAVEFORM GENERATION
# -----------------------------
def generate_waveform(config):

    duration = config["duration"]
    t = np.arange(0, duration, 1 / FS)

    if config["waveform"] == "LFM Chirp":
        signal = chirp(
            t,
            f0=config["f0"],
            f1=config["f1"],
            t1=duration,
            method="linear"
        )

    elif config["waveform"] == "Geometric Sweep":
        signal = chirp(
            t,
            f0=config["f0"],
            f1=config["f1"],
            t1=duration,
            method="logarithmic"
        )

    else:
        fc = (config["f0"] + config["f1"]) / 2

        code = np.array([
            1,1,1,1,1,
            -1,-1,
            1,1,
            -1,
            1,
            -1,
            1
        ])

        signal = np.zeros(len(t))
        samples_per_chip = len(t) // len(code)

        for i, bit in enumerate(code):
            start = i * samples_per_chip

            if i == len(code) - 1:
                end = len(t)
            else:
                end = start + samples_per_chip

            signal[start:end] = (
                bit *
                np.sin(2 * np.pi * fc * t[start:end])
            )

    signal *= np.hanning(len(signal))
    signal *= config["amplitude"]

    return t, signal

def energy_proxy(amplitude, duration):
    return amplitude**2 * duration

# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    '<div class="main-title">AquaPulse</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Environment-Aware • Mission-Aware • Energy-Aware Software-Defined Sonar Transmitter</div>',
    unsafe_allow_html=True
)

st.caption(
    "Functional software prototype for adaptive AUV sonar waveform generation"
)

st.divider()

# -----------------------------
# SCENARIO PRESETS
# -----------------------------
st.subheader("Quick Demo Scenarios")

p1, p2, p3 = st.columns(3)

if p1.button("Clear Shallow Reef", use_container_width=True):
    st.session_state.depth = 30
    st.session_state.turbidity = 10
    st.session_state.temperature = 25
    st.session_state.salinity = 35
    st.session_state.battery = 90
    st.session_state.mission = "High Resolution"

if p2.button("Muddy Deep Water", use_container_width=True):
    st.session_state.depth = 400
    st.session_state.turbidity = 85
    st.session_state.temperature = 15
    st.session_state.salinity = 35
    st.session_state.battery = 70
    st.session_state.mission = "Long Range"

if p3.button("Low Battery Mission", use_container_width=True):
    st.session_state.depth = 120
    st.session_state.turbidity = 40
    st.session_state.temperature = 20
    st.session_state.salinity = 35
    st.session_state.battery = 12
    st.session_state.mission = "Eco"

# defaults
defaults = {
    "depth": 100,
    "turbidity": 40,
    "temperature": 20,
    "salinity": 35,
    "battery": 80,
    "mission": "Balanced"
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Operating Conditions")

depth = st.sidebar.slider(
    "Depth (m)", 0, 500,
    key="depth"
)

turbidity = st.sidebar.slider(
    "Turbidity (%)", 0, 100,
    key="turbidity"
)

temperature = st.sidebar.slider(
    "Temperature (C)", 0, 40,
    key="temperature"
)

salinity = st.sidebar.slider(
    "Salinity (PSU)", 0, 40,
    key="salinity"
)

battery = st.sidebar.slider(
    "Battery (%)", 0, 100,
    key="battery"
)

mission = st.sidebar.selectbox(
    "Mission Mode",
    [
        "High Resolution",
        "Balanced",
        "Long Range",
        "Eco"
    ],
    key="mission"
)

# -----------------------------
# PROCESS
# -----------------------------
config = choose_config(
    depth,
    turbidity,
    temperature,
    salinity,
    battery,
    mission
)

t, signal = generate_waveform(config)

f0 = config["f0"]
f1 = config["f1"]
center = (f0 + f1) / 2
bandwidth = f1 - f0

# -----------------------------
# INPUT / OUTPUT SUMMARY
# -----------------------------
left, right = st.columns([1, 1.4])

with left:
    st.markdown("### Environment & Mission")

    st.write(f"**Depth:** {depth} m")
    st.write(f"**Turbidity:** {turbidity}%")
    st.write(f"**Temperature:** {temperature} C")
    st.write(f"**Salinity:** {salinity} PSU")
    st.write(f"**Battery:** {battery}%")
    st.write(f"**Mission:** {mission}")

with right:
    st.markdown("### Adaptive Decision Engine")

    for reason in config["reasons"]:
        st.write("• " + reason)

st.divider()

# -----------------------------
# METRICS
# -----------------------------
st.subheader("Selected Transmit Configuration")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Waveform", config["waveform"])
m2.metric("Start Frequency", f"{f0/1000:.0f} kHz")
m3.metric("Stop Frequency", f"{f1/1000:.0f} kHz")
m4.metric("Bandwidth", f"{bandwidth/1000:.0f} kHz")

m5, m6, m7, m8 = st.columns(4)

m5.metric("Center Frequency", f"{center/1000:.0f} kHz")
m6.metric("Pulse Duration", f"{config['duration']*1000:.2f} ms")
m7.metric("Amplitude", f"{config['amplitude']*100:.0f}%")
m8.metric("Sampling Rate", "10 MS/s")

# -----------------------------
# WAVEFORM
# -----------------------------
st.subheader("Generated Transmit Waveform")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t * 1000,
        y=signal,
        mode="lines",
        name="Adaptive Waveform"
    )
)

fig.update_layout(
    xaxis_title="Time (ms)",
    yaxis_title="Normalized Amplitude",
    height=380
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# FFT
# -----------------------------
fft_data = np.fft.rfft(signal)
freq = np.fft.rfftfreq(len(signal), 1 / FS)
magnitude = np.abs(fft_data)

st.subheader("Frequency Spectrum")

fft_fig = go.Figure()

fft_fig.add_trace(
    go.Scatter(
        x=freq / 1000,
        y=magnitude,
        mode="lines"
    )
)

fft_fig.update_layout(
    xaxis_title="Frequency (kHz)",
    yaxis_title="Magnitude",
    xaxis_range=[0, 700],
    height=380
)

st.plotly_chart(
    fft_fig,
    use_container_width=True
)

# -----------------------------
# SPECTROGRAM
# -----------------------------
st.subheader("Time-Frequency Spectrogram")

f, ts, Sxx = spectrogram(
    signal,
    FS,
    nperseg=1024,
    noverlap=900
)

mask = f <= 600000

spec_fig = go.Figure(
    data=go.Heatmap(
        x=ts * 1000,
        y=f[mask] / 1000,
        z=10 * np.log10(Sxx[mask] + 1e-12)
    )
)

spec_fig.update_layout(
    xaxis_title="Time (ms)",
    yaxis_title="Frequency (kHz)",
    height=430
)

st.plotly_chart(
    spec_fig,
    use_container_width=True
)

# -----------------------------
# ENERGY
# -----------------------------
st.subheader("Energy-Aware Operation")

adaptive_energy = energy_proxy(
    config["amplitude"],
    config["duration"]
)

baseline_energy = energy_proxy(
    0.8,
    0.003
)

change = (
    (baseline_energy - adaptive_energy)
    / baseline_energy
) * 100

e1, e2, e3 = st.columns(3)

e1.metric(
    "Adaptive Energy Index",
    f"{adaptive_energy:.6f}"
)

e2.metric(
    "Fixed Baseline Index",
    f"{baseline_energy:.6f}"
)

if change >= 0:
    e3.metric(
        "Relative Reduction",
        f"{change:.1f}%"
    )
else:
    e3.metric(
        "Additional Energy",
        f"{abs(change):.1f}%"
    )

st.caption(
    "Current energy metric is a simulation proxy proportional to amplitude squared times pulse duration. "
    "The physical prototype will use INA226 measurements."
)

# -----------------------------
# ARCHITECTURE
# -----------------------------
st.divider()

st.subheader("System Architecture")

st.code("""
Environmental Sensors
        |
        v
ADC + Sensor Fusion
        |
        v
Adaptive Decision Engine
 Environment + Mission + Battery
        |
        v
Waveform Generator
 LFM / Geometric / Phase-Coded
        |
        v
Hann Window
        |
        v
Timer + DMA
        |
        v
High-Speed DAC
        |
        v
Analog LPF + Amplifier
        |
        v
Physical Sonar Output
        |
   +----+----+
   |         |
Scope     ADC Loopback
             |
             v
      Signal Health Check
""")

# -----------------------------
# INNOVATION
# -----------------------------
st.subheader("Key Innovations")

i1, i2, i3 = st.columns(3)

with i1:
    st.markdown("#### Adaptive Intelligence")
    st.write(
        "Waveform changes using environment, mission objective and battery condition."
    )

with i2:
    st.markdown("#### Energy-Aware Transmission")
    st.write(
        "Pulse amplitude and duration are reduced when full transmit energy is unnecessary."
    )

with i3:
    st.markdown("#### Closed-Loop Validation")
    st.write(
        "Final hardware includes ADC loopback and power monitoring for real output verification."
    )

st.divider()

if battery < 20:
    st.warning(
        "LOW BATTERY: Energy-saving override is active."
    )
else:
    st.success(
        "SYSTEM READY: Adaptive transmitter operating normally."
    )

st.caption(
    "AquaPulse | SIH Software Prototype | Adaptive Software-Defined Sonar Transmitter"
)
