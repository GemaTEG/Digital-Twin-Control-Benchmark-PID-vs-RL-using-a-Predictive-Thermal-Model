<div align="center">
  <h1>Digital Twin Control Benchmark (PID/PI vs RL) using a Predictive Thermal Model</h1>
  <p>
    A simulation framework that turns a trained thermal predictor into a <b>digital twin</b> and benchmarks
    <b>classical control</b> (PI/PID) against <b>Reinforcement Learning</b> (RL) for temperature regulation.
  </p>

  <p>
    <b>Base thermal model (upstream repo):</b><br/>
    <a href="https://github.com/GemaTEG/Predictive-Thermal-Model-of-Cooling-Controller">
      https://github.com/GemaTEG/Predictive-Thermal-Model-of-Cooling-Controller
    </a>
  </p>
</div>
<hr/>

<h2>1) What this repository is</h2>
<p>
  This project uses a previously trained machine-learning thermal model as a <b>plant surrogate</b> (digital twin)
  to simulate a TEC-based cooling controller. The goal is to compare:
</p>
<ul>
  <li><b>PI/PID-style control</b> (baseline controller with anti-windup and output limits)</li>
  <li><b>Reinforcement Learning control</b> (example: Soft Actor-Critic / SAC) trained in a Gymnasium environment</li>
</ul>
<p>
  Both controllers attempt to track a temperature setpoint while minimizing control effort and avoiding aggressive changes.
</p>

<hr/>

<h2>2) Relationship to the upstream thermal-model repository</h2>
<p>
  This repository is intended to be a <b>fork / derivative work</b> that focuses on <b>closed-loop control simulation</b>.
  The predictive thermal model itself was trained and documented in the upstream repository:
</p>
<ul>
  <li>
    <a href="https://github.com/GemaTEG/Predictive-Thermal-Model-of-Cooling-Controller">
      Predictive-Thermal-Model-of-Cooling-Controller
    </a>
  </li>
</ul>

<p>
  <b>Attribution note:</b> this repo reuses the trained model and/or the training approach from the upstream repo.
  If you copy files (e.g., exported model artifacts), keep the upstream license and credits intact.
</p>

<hr/>

<h2>3) Core idea: Thermal model as a Digital Twin</h2>
<p>
  Instead of running experiments on physical hardware, we simulate the thermal plant using a learned predictor
  (e.g., XGBoost) that estimates the next temperature state given the current state and control input (TEC current).
</p>

<p>
  This enables rapid and repeatable experiments for:
</p>
<ul>
  <li>Controller tuning (PI/PID gains, saturation, anti-windup)</li>
  <li>RL training (policy optimization in simulation)</li>
  <li>Fair comparison under identical disturbances / setpoints / constraints</li>
</ul>

<hr/>

<h2>4) What’s inside</h2>

<h3>4.1 Gymnasium environment (Digital Twin)</h3>
<p>
  The simulation is implemented as a <b>Gymnasium</b> environment for RL training and evaluation.
  The environment uses the predictive model to roll the temperature forward step-by-step.
</p>

<p><b>Action</b></p>
<ul>
  <li>
    Single continuous action: <b>normalized TEC current</b> in <code>[-1, +1]</code>,
    scaled internally to a physical current range (e.g., <code>[0, 8] A</code>).
  </li>
</ul>

<p><b>Observation</b></p>
<p>
  A compact state vector including temperatures, setpoint, error, and control history (lags), for example:
</p>
<ul>
  <li><code>T_aux</code> (aux/cold-plate temperature)</li>
  <li><code>T_cpu</code> (optional, if a CPU predictor model is available)</li>
  <li><code>setpoint</code></li>
  <li><code>error</code></li>
  <li>current and lagged currents (to capture dynamics)</li>
  <li>optional squared current features and lagged squared currents</li>
  <li><code>step_fraction</code> (progress through the episode)</li>
</ul>

<p><b>Reward shaping</b></p>
<p>
  The reward encourages accurate tracking while discouraging excessive current and abrupt changes (slew):
</p>
<pre><code>reward = -abs(error)
         - (small penalty on current magnitude)
         - (small penalty on current slew-rate)</code></pre>

<p><b>Termination / truncation</b></p>
<ul>
  <li>Episodes truncate after a fixed horizon (<code>max_steps</code>).</li>
  <li>Safety termination can trigger if temperatures go out of a valid range.</li>
</ul>

<h3>4.2 Baseline controller (PI/PID-style)</h3>
<p>
  A firmware-style PI baseline is included with:
</p>
<ul>
  <li>Output clamping (min/max current)</li>
  <li>Integral clamping</li>
  <li>Anti-windup back-calculation (to avoid integrator runaway)</li>
</ul>

<h3>4.3 RL controller</h3>
<p>
  RL training can be performed using <b>Stable-Baselines3</b>. An example setup uses:
</p>
<ul>
  <li><b>SAC</b> (Soft Actor-Critic) with an MLP policy</li>
  <li>Experience replay buffer</li>
  <li>Continuous action space</li>
</ul>

<hr/>

<h2>5) Model artifacts (how the digital twin is loaded)</h2>
<p>
  The environment expects exported artifacts (example filenames):
</p>
<ul>
  <li><code>model.pkl</code> and <code>scaler.pkl</code> for the main temperature predictor</li>
  <li><code>model_cpu.pkl</code> and <code>scaler_cpu.pkl</code> (optional) for CPU temperature prediction</li>
</ul>

<p>
  Export example (Joblib):
</p>
<pre><code>import joblib
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")</code></pre>

<p>
  You can either:
</p>
<ol>
  <li>Train/export these artifacts in the upstream repo and copy them here, or</li>
  <li>Train/export inside this repo if you include the training notebook/scripts.</li>
</ol>

<hr/>

<h2>6) Quickstart</h2>

<h3>6.1 Create a Python environment</h3>
<pre><code>python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate</code></pre>

<h3>6.2 Install dependencies</h3>
<pre><code>pip install -U pip
pip install numpy pandas scikit-learn joblib xgboost
pip install gymnasium stable-baselines3 matplotlib</code></pre>

<h3>6.3 Put model artifacts in place</h3>
<p>
  Copy or generate:
</p>
<ul>
  <li><code>model.pkl</code>, <code>scaler.pkl</code></li>
  <li>(optional) <code>model_cpu.pkl</code>, <code>scaler_cpu.pkl</code></li>
</ul>

<h3>6.4 Run the simulation / notebook</h3>
<p>
  Open the notebook and run cells end-to-end:
</p>
<pre><code>jupyter notebook</code></pre>

<p>
  Notebook example (in this repo): <code>gym env for simulation.ipynb</code>
</p>

<hr/>

<h2>7) Experiments & evaluation</h2>
<p>
  The recommended evaluation compares controllers over identical scenarios:
</p>
<ul>
  <li>Same initial temperatures</li>
  <li>Same setpoint schedule (fixed or randomized within a range)</li>
  <li>Same action limits (current min/max)</li>
  <li>Same episode length</li>
</ul>

<p><b>Suggested metrics</b></p>
<ul>
  <li><b>Tracking error</b>: MAE / RMSE of <code>(T_aux - setpoint)</code></li>
  <li><b>Overshoot / undershoot</b>: peak deviation from setpoint</li>
  <li><b>Settling time</b>: time to enter and stay within a tolerance band</li>
  <li><b>Control effort</b>: average current and/or integral of |current|</li>
  <li><b>Slew</b>: average |Δcurrent| per step</li>
</ul>

<hr/>

<h2>8) Example: RL training (SAC)</h2>
<p>
  A minimal Stable-Baselines3 SAC training loop looks like:
</p>

<pre><code>from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

check_env(env, warn=True)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=256,
    gamma=0.99,
    train_freq=1,
)

model.learn(total_timesteps=200_000)
model.save("sac_tec_env")</code></pre>

<hr/>

<h2>9) Notes & assumptions</h2>
<ul>
  <li>
    This is a <b>simulation benchmark</b>. Real-world performance depends on modeling fidelity and hardware constraints.
  </li>
  <li>
    Reward design matters. If the RL agent behaves aggressively, increase current/slew penalties or adjust episode setup.
  </li>
  <li>
    If you add noise/disturbances, evaluate robustness for both PID and RL under the same conditions.
  </li>
</ul>

<hr/>

<h2>10) License</h2>
<p>
  This repository reuses ideas and/or artifacts from the upstream project. Please ensure you:
</p>
<ul>
  <li>Keep the upstream license terms if you copied code or model artifacts</li>
  <li>Preserve attribution (link to the original repository)</li>
</ul>
<p>
  If the upstream repo includes a license file, copy it here and follow its requirements.
</p>

<hr/>

<h2>11) Citation / credit</h2>
<p>
  If you use this work, please credit:
</p>
<ul>
  <li>
    Upstream thermal model repository:
    <a href="https://github.com/GemaTEG/Predictive-Thermal-Model-of-Cooling-Controller">
      GemaTEG/Predictive-Thermal-Model-of-Cooling-Controller
    </a>
  </li>
  <li>
    This repo (digital twin control benchmark): (add your new repo link here)
  </li>
</ul>

<hr/>

<div align="center">
  <p>
    <b>Maintainer:</b> (your name / handle)<br/>
    <b>Contact:</b> (optional)
  </p>
</div>










