import numpy as np
import matplotlib.pyplot as plt
 
# Parameters for the Thevenin battery model
C_nom = 2.2  # Nominal capacity in Ah
R0 = 0.01  # Ohmic resistance (Ohms)
R1 = 0.015  # Resistance of RC circuit (Ohms)
C1 = 2400  # Capacitance (Farads)
dt = 1  # Time step (seconds)
 
# SOC-OCV Polynomial Fit (2nd-order polynomial)
SOC_data = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
OCV_data = np.array([2.8, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2])
p = np.polyfit(SOC_data, OCV_data, 5)  # Fit a 2nd order polynomial to the SOC-OCV data
 
# Current profile (discharge of -1A constant current)
time = np.arange(0, 3600 + dt, dt)  # Simulating for 1 hour
I = -1 * np.ones(time.shape)  # Constant discharge current of -1A
V_true = np.zeros(time.shape)  # True terminal voltage
SOC_true = np.zeros(time.shape)  # True SOC
SOC_est = np.zeros(time.shape)  # Estimated SOC
 
# Kalman filter parameters
Q = np.array([[1e-5, 0], [0, 1e-4]])  # Process noise covariance
R_meas = 0.01  # Measurement noise covariance
P = np.eye(2)  # Initial error covariance (Identity matrix)
 
# Initialize states
SOC = 1.0  # Initial true SOC (fully charged)
V_RC = 0.0  # Initial voltage across R1 and C1
SOC_estimate = SOC  # Initial SOC estimate
V_RC_estimate = 0.0  # Initial V_RC estimate
 
for k in range(1, len(time)):
    # True SOC and voltage update (based on Thevenin model)
    SOC = SOC - I[k] * dt / 3600 / C_nom  # Coulomb counting
    SOC = np.clip(SOC, 0, 1)  # Keep SOC within bounds
 
    # Polynomial-based OCV calculation
    Voc = np.polyval(p, SOC)  # Compute OCV based on the fitted polynomial
   
    # True terminal voltage (Thevenin model)
    V_RC = np.exp(-dt / (R1 * C1)) * V_RC + (1 - np.exp(-dt / (R1 * C1))) * I[k] * R1
    V_true[k] = Voc - I[k] * R0 - V_RC  # True terminal voltage
    SOC_true[k] = SOC  # Store true SOC
   
    # Kalman Filter Prediction
    A = np.array([[1, 0], [0, np.exp(-dt / (R1 * C1))]])
    B = np.array([-dt / C_nom, R1 * (1 - np.exp(-dt / (R1 * C1)))])
    x_pred = A @ np.array([SOC_estimate, V_RC_estimate]) + B * I[k]
    P_pred = A @ P @ A.T + Q
   
    # Measurement update
    C_matrix = np.array([1, -I[k] * R0])
    K = P_pred @ C_matrix.T / (C_matrix @ P_pred @ C_matrix.T + R_meas)
    V_measured = V_true[k] + 0.05 * np.random.randn()  # Noisy measurement
    x_est = x_pred + K * (V_measured - C_matrix @ x_pred)
    P = (np.eye(2) - np.outer(K, C_matrix)) @ P_pred
   
    # Update estimates
    SOC_estimate = x_est[0]
    V_RC_estimate = x_est[1]
    SOC_est[k] = SOC_estimate  # Store estimated SOC
 
# Plot true and estimated SOC
plt.figure()
plt.plot(time / 60, SOC_true, 'r-', label='True SOC', linewidth=2)
plt.plot(time / 60, SOC_est, 'g-', label='Estimated SOC', linewidth=2)
plt.xlabel('Time (minutes)')
plt.ylabel('State of Charge (SOC)')
plt.title('True SOC vs Estimated SOC')
plt.legend()
plt.grid(True)
plt.show()
 
# Plot OCV vs SOC
plt.figure()
plt.plot(SOC_data, OCV_data, 'b-o', label='Original Data', linewidth=2)
SOC_test = np.linspace(0, 1, 100)  # Generate test SOC values for polynomial plot
OCV_test = np.polyval(p, SOC_test)  # Compute OCV using the polynomial
plt.plot(SOC_test, OCV_test, 'r-', label='Polynomial Fit', linewidth=2)
plt.xlabel('State of Charge (SOC)')
plt.ylabel('Open Circuit Voltage (OCV)')
plt.title('SOC-OCV Curve with Polynomial Fit')
plt.legend()
plt.grid(True)
plt.show()