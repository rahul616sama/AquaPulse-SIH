import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import chirp, spectrogram

st.set_page_config(
    page_title="AquaPulse",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# THEME / LAYOUT
# ------------------------------------------------------------
st.markdown("""
<style>
:root{
    --bg:#04111f;
    --panel:#071c31;
    --panel2:#0a2741;
    --line:#0d4f7e;
    --cyan:#23c8ff;
    --blue:#1687ff;
    --green:#38e587;
    --text:#eaf7ff;
    --muted:#8fb4cc;
}
html, body, [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 75% 10%, rgba(20,100,170,.15), transparent 32%),
      linear-gradient(180deg,#03101c 0%,#061728 100%);
    color:var(--text);
}
[data-testid="stHeader"]{background:transparent;}
.block-container{
    max-width:1500px;
    padding-top:.75rem;
    padding-bottom:1.1rem;
}
h1,h2,h3,p{color:var(--text);}
.ap-header{
    display:flex;
    align-items:center;
    gap:14px;
    margin-bottom:2px;
}
.ap-logo{
    width:54px;height:54px;border-radius:14px;
    display:flex;align-items:center;justify-content:center;
    font-size:30px;
    border:1px solid #0e5f94;
    background:linear-gradient(180deg,#0b2e4c,#061827);
}
.ap-title{
    font-size:2.9rem;font-weight:900;line-height:.92;
    letter-spacing:-.04em;
}
.ap-title span{color:var(--cyan);}
.ap-sub{color:#b8d5e7;font-size:.92rem;margin-top:5px;}
.panel{
    border:1px solid var(--line);
    background:linear-gradient(180deg,rgba(8,31,52,.96),rgba(4,21,37,.96));
    border-radius:12px;
    padding:12px 14px;
    box-shadow:inset 0 0 22px rgba(22,135,255,.03);
}
.panel-title{
    font-size:.86rem;font-weight:800;color:#dff6ff;
    margin-bottom:8px;
}
.mini-label{
    color:var(--muted);font-size:.72rem;text-transform:uppercase;
    letter-spacing:.05em;
}
.value-line{
    display:flex;justify-content:space-between;gap:10px;
    border-bottom:1px solid rgba(50,125,170,.22);
    padding:5px 0;font-size:.82rem;
}
.value-line:last-child{border-bottom:none;}
.decision{
    color:#d9f6ff;font-size:.82rem;line-height:1.38;margin:5px 0;
}
.decision::before{content:"✓ ";color:var(--green);font-weight:900;}
.status-good{
    border:1px solid #1aa65f;
    background:linear-gradient(180deg,rgba(13,91,58,.22),rgba(3,35,27,.36));
    border-radius:12px;
    padding:18px 20px;
    min-height:112px;
    display:flex;
    flex-direction:column;
    justify-content:center;
}
.status-bad{
    border:1px solid #b74949;
    background:linear-gradient(180deg,rgba(110,30,30,.22),rgba(45,12,12,.36));
    border-radius:12px;
    padding:18px 20px;
    min-height:112px;
    display:flex;
    flex-direction:column;
    justify-content:center;
}
.status-main{font-size:1.18rem;font-weight:900;color:var(--green);}
.status-bad .status-main{color:#ff7d7d;}
.status-sub{font-size:.78rem;color:#b7cedd;margin-top:3px;}
div[data-testid="stMetric"]{
    border:1px solid #0d4f7e;
    background:linear-gradient(180deg,#09243c,#071a2d);
    border-radius:9px;
    padding:6px 8px;
}
div[data-testid="stMetricLabel"] p{font-size:.68rem;color:#9ec4d9;}
div[data-testid="stMetricValue"]{font-size:1.15rem;color:#dff8ff;}
.stButton>button{
    border:1px solid #0d5f96;
    border-radius:8px;
    background:#082139;
    color:#e7f8ff;
    font-weight:700;
    min-height:38px;
}
.stButton>button:hover{
    border-color:#27cfff;
    color:white;
    background:#0a3153;
}
[data-baseweb="select"]>div{
    background:#071c31!important;
    border-color:#0d4f7e!important;
}
[data-testid="stSlider"]{margin-top:-4px;margin-bottom:-7px;}
[data-testid="stExpander"]{
    border:1px solid #0d4f7e!important;
    background:#06182a!important;
    border-radius:9px!important;
}
hr{border-color:rgba(30,110,160,.25);}
.small-note{color:#7faac3;font-size:.7rem;margin-top:4px;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
PROFILES = {
    "Long Range": {"fc":150.0, "bw":100.0, "vpk":28.0, "z":70.0},
    "Balanced": {"fc":300.0, "bw":180.0, "vpk":24.0, "z":60.0},
    "High Resolution": {"fc":430.0, "bw":120.0, "vpk":18.0, "z":50.0},
    "Eco": {"fc":300.0, "bw":180.0, "vpk":24.0, "z":60.0},
}

WEIGHTS = {
    "High Resolution": (0.20,0.60,0.20),
    "Balanced": (0.35,0.35,0.30),
    "Long Range": (0.60,0.20,0.20),
    "Eco": (0.20,0.15,0.65),
}

DEFAULTS = {
    "depth":100,
    "turbidity":10,
    "temperature":12,
    "salinity":35,
    "battery":80,
    "mission":"Balanced",
    "req_range":120,
    "req_res":0.20,
}

for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_preset(name):
    presets = {
        "reef": dict(depth=40,turbidity=8,temperature=24,salinity=35,battery=90,
                     mission="High Resolution",req_range=80,req_res=0.08),
        "deep": dict(depth=600,turbidity=15,temperature=8,salinity=35,battery=70,
                     mission="Long Range",req_range=200,req_res=0.50),
        "eco": dict(depth=120,turbidity=12,temperature=18,salinity=35,battery=18,
                    mission="Eco",req_range=100,req_res=0.35),
    }
    for k,v in presets[name].items():
        st.session_state[k] = v

def sound_speed(T,S,D):
    return (1448.96 + 4.591*T - 5.304e-2*T*T + 2.374e-4*T*T*T
            + 1.340*(S-35) + 1.630e-2*D + 1.675e-7*D*D
            - 1.025e-2*T*(S-35) - 7.139e-13*T*D*D*D)

def absorption(f,T,S,Dm,pH=8.0):
    D = Dm/1000.0
    f1 = 0.78*math.sqrt(max(S,1)/35.0)*math.exp(T/26.0)
    f2 = 42.0*math.exp(T/17.0)
    A = 0.106*math.exp((pH-8)/0.56)
    B = 0.52*(1+T/43.0)*(S/35.0)*math.exp(-D/6.0)
    C = 0.00049*math.exp(-(T/27.0 + D/17.0))
    f2s = f*f
    return A*f1*f2s/(f1*f1+f2s) + B*f2*f2s/(f2*f2+f2s) + C*f2s

def energy_j(amp,dur_ms,p):
    vrms = amp*p["vpk"]/math.sqrt(2)
    power = (vrms*vrms)/max(p["z"],1)
    return (power/0.82)*(dur_ms/1000.0)

def optimize(env,profile):
    fc_lo = profile["fc"]-profile["bw"]/2
    fc_hi = profile["fc"]+profile["bw"]/2
    fcs=[100,150,200,250,300,350,400,450,500]
    bws=[40,80,120,180]
    durs=[0.5,1.0,2.0,3.0]
    amps=[0.3,0.5,0.7,0.9]
    candidates=[]
    reject={"band":0,"resolution":0,"range":0,"energy":0}
    # 120 Wh nominal pack, 20% reserve, mission ping count assumption.
    remaining_wh = 120.0*env["battery"]/100.0
    reserve_wh = 24.0
    ping_plan = 20000
    budget = max(0.0,remaining_wh-reserve_wh)*3600/max(ping_plan,1)

    for fc in fcs:
        for bw in bws:
            for dur in durs:
                for amp in amps:
                    lo,hi=fc-bw/2,fc+bw/2
                    if lo<fc_lo or hi>fc_hi:
                        reject["band"]+=1
                        continue
                    res=env["c"]/(2*bw*1000.0)
                    if res>env["req_res"]:
                        reject["resolution"]+=1
                        continue
                    a=absorption(fc,env["T"],env["S"],env["D"])
                    tl=20*math.log10(max(env["range"],1))+a*(env["range"]/1000)
                    pg=10*math.log10(max(bw*1000*(dur/1000),1))
                    amp_db=20*math.log10(max(amp,0.01))
                    margin=190+amp_db-2*tl-25-70-2-0.02*env["turbidity"]+12+pg-10
                    if margin<0:
                        reject["range"]+=1
                        continue
                    e=energy_j(amp,dur,profile)
                    if e>budget:
                        reject["energy"]+=1
                        continue
                    candidates.append(dict(fc=fc,bw=bw,dur=dur,amp=amp,res=res,margin=margin,energy=e,alpha=a))

    if not candidates:
        return None,reject,budget,[]

    max_margin=max(x["margin"] for x in candidates)
    max_energy=max(x["energy"] for x in candidates)
    min_res=min(x["res"] for x in candidates)
    wr,wq,we=WEIGHTS[env["mission"]]
    for x in candidates:
        rhat=max(0,min(1,x["margin"]/max(max_margin,1e-9)))
        qhat=max(0,min(1,min_res/max(x["res"],1e-9)))
        ehat=max(0,min(1,x["energy"]/max(max_energy,1e-9)))
        x["score"]=wr*rhat+wq*qhat-we*ehat
    candidates.sort(key=lambda x:x["score"],reverse=True)
    return candidates[0],reject,budget,candidates

def waveform(choice):
    if not choice:
        return np.array([]),np.array([])
    fs=10_000_000
    dur_s=choice["dur"]/1000.0
    t=np.arange(int(fs*dur_s))/fs
    f0=(choice["fc"]-choice["bw"]/2)*1000
    f1=(choice["fc"]+choice["bw"]/2)*1000
    y=chirp(t,f0=f0,f1=f1,t1=dur_s,method="linear")
    y*=np.hanning(len(y))*choice["amp"]
    return t,y

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="ap-header">
  <div class="ap-logo">🌊</div>
  <div>
    <div class="ap-title">Aqua<span>Pulse</span></div>
    <div class="ap-sub">Environment-Aware · Mission-Aware · Energy-Aware Software-Defined Sonar Transmitter</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# QUICK DEMO ROW
# ------------------------------------------------------------
q0,q1,q2,q3 = st.columns([.9,1,1,1])
with q0:
    st.markdown('<div class="panel-title" style="padding-top:10px">Quick Demo Scenarios</div>', unsafe_allow_html=True)
if q1.button("🌿 Clear Shallow Reef", use_container_width=True):
    apply_preset("reef"); st.rerun()
if q2.button("🌊 Deep-Water Survey", use_container_width=True):
    apply_preset("deep"); st.rerun()
if q3.button("🔋 Low Battery Mission", use_container_width=True):
    apply_preset("eco"); st.rerun()

# ------------------------------------------------------------
# MAIN DASHBOARD
# ------------------------------------------------------------
controls, body = st.columns([1.0,3.55], gap="small")

with controls:
    st.markdown('<div class="panel-title">Operating Conditions</div>', unsafe_allow_html=True)
    depth=st.slider("Depth (m)",0,1000,step=10,key="depth")
    turbidity=st.slider("Turbidity",0,100,step=1,key="turbidity")
    temperature=st.slider("Temperature (°C)",-2,35,step=1,key="temperature")
    salinity=st.slider("Salinity (PSU)",25,40,step=1,key="salinity")
    battery=st.slider("Battery (%)",0,100,step=1,key="battery")
    mission=st.selectbox("Mission Mode",list(WEIGHTS.keys()),key="mission")
    with st.expander("Mission target"):
        req_range=st.slider("Range (m)",20,500,step=10,key="req_range")
        req_res=st.slider("Resolution (m)",0.05,1.00,step=0.05,key="req_res")

with body:
    c=sound_speed(temperature,salinity,depth)
    profile=PROFILES[mission]
    env=dict(T=temperature,S=salinity,D=depth,turbidity=turbidity,battery=battery,
             mission=mission,range=req_range,req_res=req_res,c=c)
    choice,reject,budget,candidates=optimize(env,profile)

    top1,top2,top3=st.columns([1.05,1.15,1.35],gap="small")

    with top1:
        st.markdown('<div class="panel-title">🌊 Environment & Mission</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="panel">
          <div class="value-line"><span>Depth</span><b>{depth} m</b></div>
          <div class="value-line"><span>Turbidity</span><b>{turbidity}</b></div>
          <div class="value-line"><span>Temperature</span><b>{temperature} °C</b></div>
          <div class="value-line"><span>Salinity</span><b>{salinity} PSU</b></div>
          <div class="value-line"><span>Battery</span><b>{battery}%</b></div>
          <div class="value-line"><span>Mission</span><b>{mission}</b></div>
        </div>
        """,unsafe_allow_html=True)

    with top2:
        st.markdown('<div class="panel-title">⚙ Adaptive Decision Engine</div>', unsafe_allow_html=True)
        if choice:
            reasons=[
                f"Profile limits transmission to {profile['fc']-profile['bw']/2:.0f}–{profile['fc']+profile['bw']/2:.0f} kHz.",
                f"{mission} weighting selects the highest-scoring valid candidate.",
                f"Chosen ping meets {req_range} m modeled range and {req_res:.2f} m resolution targets."
            ]
        else:
            reasons=[
                "No candidate satisfies every hard constraint.",
                "Relax range/resolution or increase available battery.",
                "Transmission is inhibited instead of forcing an invalid waveform."
            ]
        st.markdown('<div class="panel">'+''.join([f'<div class="decision">{r}</div>' for r in reasons])+'</div>',unsafe_allow_html=True)

    with top3:
        st.markdown('<div class="panel-title">〽 Selected Transmit Configuration</div>', unsafe_allow_html=True)
        if choice:
            a,b,c1,d=st.columns(4)
            a.metric("Waveform","LFM Chirp")
            b.metric("Start",f"{choice['fc']-choice['bw']/2:.0f} kHz")
            c1.metric("Stop",f"{choice['fc']+choice['bw']/2:.0f} kHz")
            d.metric("Bandwidth",f"{choice['bw']:.0f} kHz")
            e,f,g,h=st.columns(4)
            e.metric("Centre",f"{choice['fc']:.0f} kHz")
            f.metric("Pulse",f"{choice['dur']:.1f} ms")
            g.metric("Amplitude",f"{choice['amp']*100:.0f}%")
            h.metric("Sample Rate","10 MS/s")
        else:
            st.error("TX INHIBIT — no valid waveform")

    t,sig=waveform(choice)

    ch1,ch2,ch3=st.columns([1.15,.95,1.05],gap="small")
    with ch1:
        st.markdown('<div class="panel-title">〰 Generated Transmit Waveform</div>', unsafe_allow_html=True)
        fig=go.Figure()
        if sig.size:
            stride=max(1,len(sig)//3000)
            fig.add_trace(go.Scatter(x=t[::stride]*1000,y=sig[::stride],mode="lines",line=dict(color="#37c7ff",width=1.5)))
        fig.update_layout(height=255,template="plotly_dark",paper_bgcolor="#071c31",plot_bgcolor="#071c31",
                          margin=dict(l=38,r=10,t=8,b=35),xaxis_title="Time (ms)",yaxis_title="Amplitude",
                          font=dict(size=10,color="#bfe8ff"),showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    with ch2:
        st.markdown('<div class="panel-title">▥ Frequency Spectrum</div>', unsafe_allow_html=True)
        fig=go.Figure()
        if sig.size:
            spec=np.abs(np.fft.rfft(sig))
            freq=np.fft.rfftfreq(len(sig),1/10_000_000)
            mask=freq<=600000
            fig.add_trace(go.Scatter(x=freq[mask]/1000,y=spec[mask],mode="lines",line=dict(color="#ffb11b",width=2)))
        fig.update_layout(height=255,template="plotly_dark",paper_bgcolor="#071c31",plot_bgcolor="#071c31",
                          margin=dict(l=38,r=10,t=8,b=35),xaxis_title="Frequency (kHz)",yaxis_title="Magnitude",
                          font=dict(size=10,color="#bfe8ff"),showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    with ch3:
        st.markdown('<div class="panel-title">▦ Time-Frequency Spectrogram</div>', unsafe_allow_html=True)
        if sig.size:
            nper=min(1024,max(128,len(sig)//16))
            fsp,tsp,Sxx=spectrogram(sig,10_000_000,nperseg=nper,noverlap=int(nper*.75))
            mask=fsp<=600000
            fig=go.Figure(go.Heatmap(x=tsp*1000,y=fsp[mask]/1000,z=10*np.log10(Sxx[mask]+1e-12),
                                     colorscale="Turbo",showscale=False))
        else:
            fig=go.Figure()
        fig.update_layout(height=255,template="plotly_dark",paper_bgcolor="#071c31",plot_bgcolor="#071c31",
                          margin=dict(l=38,r=10,t=8,b=35),xaxis_title="Time (ms)",yaxis_title="Frequency (kHz)",
                          font=dict(size=10,color="#bfe8ff"))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    en,status=st.columns([2.1,1],gap="small")
    with en:
        st.markdown('<div class="panel-title">🔋 Energy-Aware Operation</div>', unsafe_allow_html=True)
        if choice:
            fixed_energy=energy_j(.8,2.0,profile)
            adaptive=choice["energy"]
            delta=(fixed_energy-adaptive)/max(fixed_energy,1e-9)*100
            e1,e2,e3=st.columns(3)
            e1.metric("Adaptive Energy",f"{adaptive*1000:.2f} mJ")
            e2.metric("Fixed Baseline",f"{fixed_energy*1000:.2f} mJ")
            e3.metric("Relative Change",f"{delta:+.1f}%")
            st.markdown('<div class="small-note">Electrical comparison only. Underwater efficiency is not measured here.</div>',unsafe_allow_html=True)
        else:
            st.info("No valid transmission, so energy output is inhibited.")

    with status:
        if choice:
            st.markdown("""
            <div class="status-good">
              <div class="status-main">✓ SYSTEM READY</div>
              <div class="status-sub">Adaptive transmitter operating normally.</div>
            </div>
            """,unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-bad">
              <div class="status-main">TX INHIBIT</div>
              <div class="status-sub">Mission or hardware limits are not satisfied.</div>
            </div>
            """,unsafe_allow_html=True)

st.caption("AquaPulse · SIH software prototype · modeled acoustic performance only; real range/efficiency require transducer and tank validation.")
