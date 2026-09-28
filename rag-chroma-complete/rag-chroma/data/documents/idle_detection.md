# Atlas Kitchen — Idle detection prototype
This is a fictional example; its constants are demonstration settings.

The prototype uses an LSTM autoencoder trained on known idle periods. The input window contains 30 consecutive samples at a one-second sampling interval. Features include temperature minus setpoint, recent temperature change, appliance state, and heater relay state.

Low reconstruction error means the window resembles the learned idle pattern. The threshold is the 99th percentile of reconstruction errors on the idle training set. An idle status is confirmed only after five continuous minutes of qualifying behavior.

Training and evaluation use separate chronological periods. Feature scalers are fitted only on training data. This avoids using future evaluation information during training. The prototype detects idle behavior but does not quantify energy savings.
