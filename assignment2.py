import numpy as np
import math 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R





#GRAVITATIONAL PARAMETER EARTH
mu_E = 3.986004418e14 #m^3/s^2
#HEIGHT OF S/C
h = 750e3 #m
#RADIUS OF EARTH
r_E = 6378.137e3 #m
#MEAN MOTION
n = (mu_E / (h + r_E)**3)**0.5 #1/s
print(n)
#SATELLITE INERTIA PROPERTIES
J_11 = 124.531 #kgm^2
J_22 = 124.586 #kgm^2
J_33 = 0.704 #kgm^2
#DISTURBANCE TORQUE
T_d = 0.0001 #Nm
Md = np.array((T_d, T_d, T_d))
#GAINS LINEAR CONTROL EULER
Kp = np.array((0.6, 0.5, 0.009))
Kd = np.array((9.95, 9.95, 0.05))

M_linear_list_eul = []
def dynamics(t,x):
	theta = x[0:3]
	theta_dot = x[3:6]
	#CONTROL LAW
	Mc = -Kp*(theta - theta_d_array) - Kd*theta_dot
	M_linear_list_eul.append(Mc)
	M = Mc + Md

	theta_ddot = np.zeros(3)
	#2ND ORDER LINEARIZED EQUATIONS
	theta_ddot[0] = (1 / J_11)*(n*theta_dot[2]*(J_11 - J_22 + J_33) - (n**2)*theta[0]*(J_22 - J_33) + M[0])

	theta_ddot[1] = (1 / J_22)*M[1]

	theta_ddot[2] = (1 / J_33)*(-n*theta_dot[0]*(J_11 - J_22 + J_33) - (n**2)*theta[2]*(J_22 - J_11) + M[2])

	return np.concatenate([theta_dot, theta_ddot])




# Initial state
initial_deg = 25 #degrees
initial_rad = math.radians(initial_deg) #radians
x0 = np.array((initial_rad, initial_rad, initial_rad, 0, 0, 0))
theta = x0[0:3]

segments = [
    (0,      0,    99.9,  999),
    (70,   100,    500,  4000),
    (-70,  500.1,  900,  3999),
    (0,    900.1, 1500,  5999)
]

# Lists to store results
all_t = []
all_y = []


for theta_d_degrees, t_start, t_end, num_points in segments:
    # Update desired attitude
    theta_d = np.deg2rad(theta_d_degrees)
    theta_d_array = np.array((theta_d, theta_d, theta_d))  # You might set this globally or however needed

    # Time evaluation points
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, num_points)

    # Solve the system
    sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval)

    # Store results
    all_t.append(sol.t)
    all_y.append(sol.y)

    # Update initial condition for time frame
    x0 = sol.y[:, -1]


full_t = np.concatenate(all_t)
full_y = np.hstack(all_y)

def dynamics_quaternions(t,x):
	#define quaternions with linear assumption q4 = 1
	q = np.array([x[0], x[1], x[2], 1])
	q_dot = x[3:6]
	#command matrix to convert q to qe
	Qc = np.array([
		[1, q_d_array[2], - q_d_array[1], - q_d_array[0]],
		[- q_d_array[2], 1, q_d_array[0], - q_d_array[1]],
		[q_d_array[1], - q_d_array[0], 1, - q_d_array[2]]
		], dtype=float)

	qe = np.matmul(Qc, q) 
	
	#linearized angular velocity values
	omega_1 = 2*q_dot[0] - 2*n*q[2]
	omega_2 = 2*q_dot[1] - 2*n
	omega_3 = 2*q_dot[2] - 2*n*q[0]
	omega = np.array([omega_1, omega_2, omega_3])
	#control law
	Mc = -Kp*qe - Kd*omega 

	M = Mc + Md

	q_ddot = np.zeros(3)
	#linearized 2nd order equations for quaternions
	q_ddot[0] = (1 / J_11)*(-n*q_dot[2]*(J_33 - J_22 + J_11) - (n**2)*q[0]*(J_22 - J_33) + M[0]/2)

	q_ddot[1] = (1 / J_22)*(M[1]/2)

	q_ddot[2] = (1 / J_33)*(n*q_dot[0]*(J_11 - J_22 + J_33) - (n**2)*q[2]*(J_22 - J_11) + M[2]/2)

	return np.concatenate([q_dot, q_ddot])

segments = [
    (0,      0,    99.9,  999),
    (70,   100,    500,  4000),
    (-70,  500.1,  900,  3999),
    (0,    900.1, 1500,  5999)
]

all_t_quat = []
all_y_quat = []
#gains for linearized quaternions
Kp = np.array((10, 8, 1))
Kd = np.array((30, 20, 2))


theta_deg = np.array([25, 25, 25])
theta_rad = np.deg2rad(theta_deg)

q_init = R.from_euler('zyx', [25, 25, 25], degrees=True).as_quat()
x0 = np.array([q_init[0], q_init[1], q_init[2], 0, 0, 0])
for theta_d_degrees, t_start, t_end, num_points in segments:
    #define desired attitude in quaternions
    q_init_d = R.from_euler('zyx', [theta_d_degrees, theta_d_degrees, theta_d_degrees], degrees=True).as_quat()
    q_d_array = np.array((q_init_d[0], q_init_d[1], q_init_d[2], q_init_d[3]))  # You might set this globally or however needed


    # Time evaluation points
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, num_points)

    # Solve the system
    sol = solve_ivp(dynamics_quaternions, t_span, x0, t_eval=t_eval)

    # Store results
    all_t_quat.append(sol.t)
    all_y_quat.append(sol.y)

    # Update initial condition for next time frame
    x0 = sol.y[:, -1]

#store linearized quaternion results
full_t_quat = np.concatenate(all_t_quat)
full_y_quat = np.hstack(all_y_quat)

#linearized euler angles in linearized system
plt.figure(figsize=(10, 6))
plt.plot(full_t, np.rad2deg(full_y[0]), label='θ1 (Roll)')
plt.plot(full_t, np.rad2deg(full_y[1]), label='θ2 (Pitch)')
plt.plot(full_t, np.rad2deg(full_y[2]), label='θ3 (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid()

plt.show(block=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(full_t, np.rad2deg(full_y[0]), label='θ1 (Roll)', color='tab:blue')
axs[0].set_ylabel('Roll (°)')
axs[0].legend()
axs[0].grid()

axs[1].plot(full_t, np.rad2deg(full_y[1]), label='θ2 (Pitch)', color='tab:orange')
axs[1].set_ylabel('Pitch (°)')
axs[1].legend()
axs[1].grid()

axs[2].plot(full_t, np.rad2deg(full_y[2]), label='θ3 (Yaw)', color='tab:green')
axs[2].set_ylabel('Yaw (°)')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('1plotlineul.png')
plt.show(block=False)


#linearized quaternions in linearized system
plt.figure(figsize=(10, 6))
plt.plot(full_t_quat, full_y_quat[0], label='q1')
plt.plot(full_t_quat, full_y_quat[1], label='q2')
plt.plot(full_t_quat, full_y_quat[2], label='q3')
plt.xlabel('Time (s)')
plt.ylabel('q')
plt.legend()
plt.grid()

plt.show(block=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(full_t, full_y_quat[0], label='q1', color='tab:blue')
axs[0].set_ylabel('q1')
axs[0].legend()
axs[0].grid()

axs[1].plot(full_t, full_y_quat[1], label='q2', color='tab:orange')
axs[1].set_ylabel('q2')
axs[1].legend()
axs[1].grid()

axs[2].plot(full_t, full_y_quat[2], label='q3', color='tab:green')
axs[2].set_ylabel('q3')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('1plotlinquat.png')
plt.show(block=False)



#NDI SECTION

#CONSTANTS
J_matrix = np.array([[J_11,0,0], [0, J_22, 0], [0, 0, J_33]], dtype=float)
J_inverse = np.array([[1/J_11, 0, 0], [0, 1/J_22, 0], [0, 0, 1/J_33]], dtype=float)

#gains for euler NDI
Kp = np.array([0.00889, 0.00889, 0.00889])
Kd = np.array([0.114, 0.114, 0.114])



def nonlinear_dynamics(t,x):
	theta = x[0:3]
	omega = x[3:6]

	#matrix that converts angular velocity to euler angles
	N_matrix = np.array([
		[1, np.sin(theta[0])*np.tan(theta[1]), np.cos(theta[0])*np.tan(theta[1])], 
		[0, np.cos(theta[0]), -np.sin(theta[0])],
		[0, np.sin(theta[0]) / np.cos(theta[1]), np.cos(theta[0]) / np.cos(theta[1])]
		], dtype=float)
	a2 = n * np.array([
    np.cos(theta[1]) * np.sin(theta[2]),
    np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2]) + np.cos(theta[0]) * np.cos(theta[2]),
    np.cos(theta[0]) * np.sin(theta[1]) * np.sin(theta[2]) - np.sin(theta[0]) * np.cos(theta[2])])

	theta_dot = N_matrix @ (omega - a2)
	e = theta - theta_d
	dot_e = theta_dot
	v = -Kp * e - Kd * dot_e



	Omega_matrix = np.array([
		[0, -omega[2], omega[1]],
		[omega[2], 0, -omega[0]],
		[-omega[1], omega[0], 0]
		], dtype=float)

	#defining elements of Jacobian matrix w.r.t. x
	D_11 = np.tan(theta[1])*(omega[1]*np.cos(theta[0]) - omega[2]*np.sin(theta[0]))
	D_12 = (1 / (np.cos(theta[1]))**2)*(omega[1]*np.sin(theta[0]) + omega[2]*np.cos(theta[0]))
	D_15 = np.sin(theta[0])*np.tan(theta[1])
	D_16 = np.cos(theta[0])*np.tan(theta[1])
	D_21 = -np.sin(theta[0])*omega[1] - np.cos(theta[0])*omega[2]
	D_31 = (1 / np.cos(theta[1]))*(omega[1]*np.cos(theta[0]) - omega[2]*np.sin(theta[0]))
	D_32 = (np.tan(theta[1]) / np.cos(theta[1]))*(omega[1]*np.sin(theta[0]) + omega[2]*np.cos(theta[0]))
	D_35 = np.sin(theta[0]) / np.cos(theta[1])
	D_36 = np.cos(theta[0]) / np.cos(theta[1])

	D_matrix = np.array([
		[D_11, D_12, 0, 1, D_15, D_16],
		[D_21, 0, 0, 0, np.cos(theta[0]), -np.sin(theta[0])],
		[D_31, D_32, 0, 0, D_35, D_36]
		], dtype=float)

	G_matrix = np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[1/J_11, 0, 0, 1/J_11, 0, 0],
		[0, 1/J_22, 0, 0, 1/J_22, 0],
		[0, 0, 1/J_33, 0, 0, 1/J_33],
		], dtype=float)
	G_partial_matrix = np.array([
		[0, 0, 0],
		[0, 0, 0],
		[0, 0, 0],
		[1/J_11, 0, 0],
		[0, 1/J_22, 0],
		[0, 0, 1/J_33],
		], dtype=float)

	#MATRIX MULTIPLICATION
	f_x_1 = np.matmul(N_matrix, omega)
	f_x_2 = - J_inverse @ Omega_matrix @ J_matrix @ omega
	f_x = np.concatenate([f_x_1, f_x_2])
	#need the Mc calculation, stack the first two matrices, and add the Mc/Md
	M_x = np.matmul(D_matrix, G_partial_matrix)
	l_x = D_matrix @ f_x  +  D_matrix @ G_partial_matrix @ Md
	M_c = np.linalg.solve(M_x, v - l_x)

	u = np.concatenate([Md, M_c])

	xdot = f_x + G_matrix @ u 
	return xdot 

# Initial state
initial_deg = 25 #degrees
initial_rad = math.radians(initial_deg)
x0 = np.array((initial_rad, initial_rad, initial_rad, 0, 0, 0))



segments = [
    (0,      0,    99.9,  999),
    (70,   100,    500,  4000),
    (-70,  500.1,  900,  3999),
    (0,    900.1, 1500,  5999)
]

# Lists to store results
all_t = []
all_y = []

for theta_d_degrees, t_start, t_end, num_points in segments:
    #command euler angles
    theta_d = np.deg2rad(theta_d_degrees)
    theta_d_array = np.array((theta_d, theta_d, theta_d))  
    # Time evaluation points
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, num_points)

    # Solve the system
    sol = solve_ivp(nonlinear_dynamics, t_span, x0, t_eval=t_eval)

    # Store results
    all_t.append(sol.t)
    all_y.append(sol.y)

    # Update initial condition for next time frame
    x0 = sol.y[:, -1]


full_t = np.concatenate(all_t)
full_y = np.hstack(all_y)

#plot for euler NDI
plt.figure(figsize=(10, 6))
plt.plot(full_t, np.rad2deg(full_y[0]), label='θ1 (Roll)')
plt.plot(full_t, np.rad2deg(full_y[1]), label='θ2 (Pitch)')
plt.plot(full_t, np.rad2deg(full_y[2]), label='θ3 (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('NDI euler Attitude Angles vs. Time')
plt.legend()
plt.grid()
plt.show(block=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(full_t, np.rad2deg(full_y[0]), label='θ1 (Roll)', color='tab:blue')
axs[0].set_ylabel('Roll (°)')
axs[0].legend()
axs[0].grid()

axs[1].plot(full_t, np.rad2deg(full_y[1]), label='θ2 (Pitch)', color='tab:orange')
axs[1].set_ylabel('Pitch (°)')
axs[1].legend()
axs[1].grid()

axs[2].plot(full_t, np.rad2deg(full_y[2]), label='θ3 (Yaw)', color='tab:green')
axs[2].set_ylabel('Yaw (°)')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('3plotNDIeul.png')
plt.show(block=False)


def nonlinear_dynamics_quat(t,x):
	q = x[0:4]

	omega = x[4:7]

	Omega_4_matrix = np.array([
		[0, omega[2], - omega[1], omega[0]], 
		[- omega[2], 0, omega[0], omega[1]],
		[omega[1], - omega[0] , 0, omega[2]],
		[- omega[0], - omega[1], - omega[2], 0]
		], dtype=float)

	#quaternion control matrix
	M_matrix = np.array([
		[qc[3], qc[2], -qc[1], -qc[0]], 
		[-qc[2], qc[3], qc[0], -qc[1]],
		[qc[1], -qc[0], qc[3], - qc[2]],
		[qc[0], qc[1], qc[2], qc[3]]
		], dtype=float)
	qe = M_matrix @ q 
	qedot = 0.5 * (M_matrix @ Omega_4_matrix @ q)
	v = -Kp * qe[:3] - Kd * qedot[:3]


	#3X3 skew matrix for dynamics
	Omega_matrix = np.array([
		[0, -omega[2], omega[1]],
		[omega[2], 0, -omega[0]],
		[-omega[1], omega[0], 0]
		], dtype=float)

	#Jacobian w.r.t. x
	D_matrix = np.array([
		[0, omega[2], - omega[1], omega[0], q[3], -q[2], q[1]],
		[- omega[2], 0, omega[0], omega[1], q[2], q[3], -q[0]],
		[omega[1], - omega[0], 0, omega[2], -q[1], q[0], q[3]],
		[- omega[0], - omega[1], - omega[2], 0, - q[0], - q[1], - q[2]]
		], dtype=float)

	G_matrix = np.array([
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[1/J_11, 0, 0, 1/J_11, 0, 0],
		[0, 1/J_22, 0, 0, 1/J_22, 0],
		[0, 0, 1/J_33, 0, 0, 1/J_33],
		], dtype=float)
	G_partial_matrix = np.array([
		[0, 0, 0],
		[0, 0, 0],
		[0, 0, 0],
		[0, 0, 0],
		[1/J_11, 0, 0],
		[0, 1/J_22, 0],
		[0, 0, 1/J_33],
		], dtype=float)

	#MATRIX MULTIPLICATION
	f_x_1 = 0.5*(Omega_4_matrix @ q) 
	f_x_2 = - J_inverse @ Omega_matrix @ J_matrix @ omega
	f_x = np.concatenate([f_x_1, f_x_2])
	
	M_x = 0.5*np.matmul(D_matrix[:3], G_partial_matrix)
	l_x = 0.5*(D_matrix @ f_x  +  D_matrix @ G_partial_matrix @ Md)

	M_c = np.linalg.solve(M_x, v - l_x[:3])

	u = np.concatenate([Md, M_c])

	xdot = f_x + G_matrix @ u 
	return xdot 



segments = [
    (0,      0,    99.9,  999),
    (70,   100,    500,  4000),
    (-70,  500.1,  900,  3999),
    (0,    900.1, 1500,  5999)
]

all_t_quat = []
all_y_quat = []
Kp = np.array([0.00889, 0.00889, 0.00889])
Kd = np.array([0.128, 0.128, 0.128])


q_init = R.from_euler('zyx', [25, 25, 25], degrees=True).as_quat()
x0 = np.array([q_init[0], q_init[1], q_init[2], q_init[3], 0, 0, 0])
for theta_d_degrees, t_start, t_end, num_points in segments:
    # Update desired angle globally or pass it in appropriately
    q_init_d = R.from_euler('zyx', [theta_d_degrees, theta_d_degrees, theta_d_degrees], degrees=True).as_quat()
    qc = np.array((q_init_d[0], q_init_d[1], q_init_d[2], q_init_d[3]))  # You might set this globally or however needed
    #omega_n = 4 / (0.707 *(t_end - t_start / 3))
    #Kp = omega_n**2
    #Kd = 2*0.707*omega_n

    # Time evaluation points
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, num_points)

    # Solve the system
    sol = solve_ivp(nonlinear_dynamics_quat, t_span, x0, t_eval=t_eval)

    # Store results
    all_t_quat.append(sol.t)
    all_y_quat.append(sol.y)

    # Update initial condition for next segment
    x0 = sol.y[:, -1]


full_t = np.concatenate(all_t_quat)
full_y = np.hstack(all_y_quat)

#plot quaternion NDI
plt.figure(figsize=(10, 6))
plt.plot(full_t, full_y[0], label='θ1 (Roll)')
plt.plot(full_t, full_y[1], label='θ2 (Pitch)')
plt.plot(full_t, full_y[2], label='θ3 (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('NDI quaternions vs. Time')
plt.legend()
plt.grid()
plt.show(block=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(full_t, full_y[0], label='q1', color='tab:blue')
axs[0].set_ylabel('q1')
axs[0].legend()
axs[0].grid()

axs[1].plot(full_t, full_y[1], label='q2', color='tab:orange')
axs[1].set_ylabel('q2')
axs[1].legend()
axs[1].grid()

axs[2].plot(full_t, full_y[2], label='q3', color='tab:green')
axs[2].set_ylabel('q3')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid()


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('3plotNDIquat.png')
plt.show(block=False)


#TIME SEPARATION NDI

#gains for ts euler ndi
Kp1 = np.array([0.0780, 0.0780,0.0780])
Kd1 = np.array([0.113, 0.113, 0.113])
Kp2 = np.array([0.113, 0.113, 0.113])

def nonlinear_dynamics_ts(t,x):
	theta = x[0:3]
	omega = x[3:6]

	N_matrix = np.array([
		[1, np.sin(theta[0])*np.tan(theta[1]), np.cos(theta[0])*np.tan(theta[1])], 
		[0, np.cos(theta[0]), -np.sin(theta[0])],
		[0, np.sin(theta[0]) / np.cos(theta[1]), np.cos(theta[0]) / np.cos(theta[1])]
		], dtype=float)

	Omega_matrix = np.array([
		[0, -omega[2], omega[1]],
		[omega[2], 0, -omega[0]],
		[-omega[1], omega[0], 0]
		], dtype=float)


	e = theta - theta_d_array
	e_dot = N_matrix @ omega
	v1 = - Kp1 * e - Kd1 * e_dot
	omegad = np.linalg.inv(N_matrix) @ v1

	v2 = - Kp2*(omega - omegad)
	term1 = J_inverse @ Omega_matrix @ J_matrix @ omega - J_inverse @ Md 
	Tc = np.matmul(J_matrix, v2 + term1)

	omegadot = - term1 + J_inverse @ Tc
	thetadot = N_matrix @ omega 

	return np.concatenate([thetadot, omegadot])




# Initial state
initial_deg = 25 #degrees
initial_rad = math.radians(initial_deg) # radians
x0 = np.array((initial_rad, initial_rad, initial_rad, 0, 0, 0))



segments = [
    (0,      0,    99.9,  999),
    (70,   100,    500,  4000),
    (-70,  500.1,  900,  3999),
    (0,    900.1, 1500,  5999)
]

# Lists to store results
all_t = []
all_y = []

for theta_d_degrees, t_start, t_end, num_points in segments:
    # euler command angles
    theta_d = np.deg2rad(theta_d_degrees)
    theta_d_array = np.array((theta_d, theta_d, theta_d)) 

    # Time evaluation points
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, num_points)

    # Solve the system
    sol = solve_ivp(nonlinear_dynamics_ts, t_span, x0, t_eval=t_eval)

    # Store results
    all_t.append(sol.t)
    all_y.append(sol.y)

    # Update initial condition for next time period
    x0 = sol.y[:, -1]


full_t = np.concatenate(all_t)
full_y = np.hstack(all_y)

#plot for time separation NDI for euler
plt.figure(figsize=(10, 6))
plt.plot(full_t, np.rad2deg(full_y[0]), label='θ1 (Roll)')
plt.plot(full_t, np.rad2deg(full_y[1]), label='θ2 (Pitch)')
plt.plot(full_t, np.rad2deg(full_y[2]), label='θ3 (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Time separation NDI Euler Attitude Angles vs. Time')
plt.legend()
plt.grid()
plt.show(block=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(full_t, np.rad2deg(full_y[0]), label='θ1 (Roll)', color='tab:blue')
axs[0].set_ylabel('Roll (°)')
axs[0].legend()
axs[0].grid()

axs[1].plot(full_t, np.rad2deg(full_y[1]), label='θ2 (Pitch)', color='tab:orange')
axs[1].set_ylabel('Pitch (°)')
axs[1].legend()
axs[1].grid()

axs[2].plot(full_t, np.rad2deg(full_y[2]), label='θ3 (Yaw)', color='tab:green')
axs[2].set_ylabel('Yaw (°)')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('3plotTSNDIeul.png')
plt.show()

#TIME SEPARATION NDI QUATERNIONS

#gains for ts quaternion NDI
Kp1 = np.array([0.0778, 0.0778, 0.0778])
Kp2 = np.array([0.114, 0.114, 0.114])

def nonlinear_dynamics_ts_quat(t,x):
	q = x[0:4]
	omega = x[4:7]

	Q_matrix = np.array([
		[q[3], -q[2], q[1], q[0]], 
		[q[2], q[3], -q[0], q[1]],
		[-q[1], q[0], q[3],  q[2]],
		[-q[0], -q[1], -q[2], q[3]]
		], dtype=float)
	Omega_4_matrix = np.array([
		[0, omega[2], - omega[1], omega[0]], 
		[- omega[2], 0, omega[0], omega[1]],
		[omega[1], - omega[0] , 0, omega[2]],
		[- omega[0], - omega[1], - omega[2], 0]
		], dtype=float)


	#quaternion control matrix
	M_matrix = np.array([
		[qc[3], qc[2], -qc[1], -qc[0]], 
		[-qc[2], qc[3], qc[0], -qc[1]],
		[qc[1], -qc[0], qc[3], - qc[2]],
		[qc[0], qc[1], qc[2], qc[3]]
		], dtype=float)


	qe = M_matrix @ q 
	qedot = 0.5 * (M_matrix @ Omega_4_matrix @ q)
	v1 = -Kp1 * qe[:3] 


	Omega_matrix = np.array([
		[0, -omega[2], omega[1]],
		[omega[2], 0, -omega[0]],
		[-omega[1], omega[0], 0]
		], dtype=float)


	M_x = np.linalg.inv(0.5*(M_matrix @ Q_matrix))

	omegad = M_x[:3, :3] @ v1

	v2 = - Kp2*(omega - omegad)

	term1 = J_inverse @ Omega_matrix @ J_matrix @ omega - J_inverse @ Md
	Tc = np.matmul(J_matrix, v2 + term1)

	omegadot = - term1 + J_inverse @ Tc
	qdot = 0.5*(Omega_4_matrix @ q)

	return np.concatenate([qdot , omegadot])

segments = [
    (0,      0,    99.9,  999),
    (70,   100,    500,  4000),
    (-70,  500.1,  900,  3999),
    (0,    900.1, 1500,  5999)
]

all_t_quat = []
all_y_quat = []
Kp = np.array([0.005, 0.004, 0.0032])
Kd = np.array([0.12, 0.12, 0.12])


q_init = R.from_euler('zyx', [25, 25, 25], degrees=True).as_quat()
x0 = np.array([q_init[0], q_init[1], q_init[2], q_init[3], 0, 0, 0])
for theta_d_degrees, t_start, t_end, num_points in segments:
    # Update desired angle globally or pass it in appropriately
    q_init_d = R.from_euler('zyx', [theta_d_degrees, theta_d_degrees, theta_d_degrees], degrees=True).as_quat()
    qc = np.array((q_init_d[0], q_init_d[1], q_init_d[2], q_init_d[3]))  # You might set this globally or however needed

    # Time evaluation points
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, num_points)

    # Solve the system
    sol = solve_ivp(nonlinear_dynamics_ts_quat, t_span, x0, t_eval=t_eval)

    # Store results
    all_t_quat.append(sol.t)
    all_y_quat.append(sol.y)

    # Update initial condition for next segment
    x0 = sol.y[:, -1]


full_t = np.concatenate(all_t_quat)
full_y = np.hstack(all_y_quat)

#plot quaternion NDI
plt.figure(figsize=(10, 6))
plt.plot(full_t, full_y[0], label='θ1 (Roll)')
plt.plot(full_t, full_y[1], label='θ2 (Pitch)')
plt.plot(full_t, full_y[2], label='θ3 (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Time Separation NDI quaternions vs. Time')
plt.legend()
plt.grid()
plt.show(block=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(full_t, full_y[0], label='q1', color='tab:blue')
axs[0].set_ylabel('q1')
axs[0].legend()
axs[0].grid()

axs[1].plot(full_t, full_y[1], label='q2', color='tab:orange')
axs[1].set_ylabel('q2')
axs[1].legend()
axs[1].grid()

axs[2].plot(full_t, full_y[2], label='q3', color='tab:green')
axs[2].set_ylabel('q3')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid()


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('3plotTSNDIquat.png')
plt.show(block=False)


#END TIME SEPARATION


#INDI 


#GAINS
Kp3 = np.array([0.0778, 0.0778, 0.0778])
Kd3 = np.array([0.7, 0.7, 0.7])

def INDI_dynamics(x, Mc):
	theta = x[0:3]
	omega = x[3:6]

	N_matrix = np.array([
		[1, np.sin(theta[0])*np.tan(theta[1]), np.cos(theta[0])*np.tan(theta[1])], 
		[0, np.cos(theta[0]), -np.sin(theta[0])],
		[0, np.sin(theta[0]) / np.cos(theta[1]), np.cos(theta[0]) / np.cos(theta[1])]
		], dtype=float)

	Omega_matrix = np.array([
		[0, -omega[2], omega[1]],
		[omega[2], 0, -omega[0]],
		[-omega[1], omega[0], 0]
		], dtype=float)
	theta_dot = N_matrix @ omega
	term1 = - J_inverse @ Omega_matrix @ J_matrix @ omega
	omega_dot = term1 + J_inverse @ Md + J_inverse @ Mc

	xdot = np.concatenate([theta_dot, omega_dot])

	return xdot 

segments = [
    (0,      0,    99.9),
    (70,   100,    500),
    (-70,  500.1,  900),
    (0,    900.1, 1500)
]

# Lists to store results
all_t = []
all_y = []


t_step = 0.1 #s
x = np.array((np.deg2rad(25),np.deg2rad(25),np.deg2rad(25),0,0,0))
theta_d = np.array((0,0,0))
omega_prev = np.array((0,0,0))
Mc_prev = np.array((0,0,0))
for theta_d_degrees, t_start, t_end in segments:
	t = t_start
	theta_d = np.deg2rad(theta_d_degrees)
	theta_d_array = np.array((theta_d, theta_d, theta_d))
	omega_prev = x[3:6].copy()

	while t < t_end:
		theta = x[0:3]
		omega = x[3:6]

		N_matrix = np.array([
			[1, np.sin(theta[0])*np.tan(theta[1]), np.cos(theta[0])*np.tan(theta[1])], 
			[0, np.cos(theta[0]), -np.sin(theta[0])],
			[0, np.sin(theta[0]) / np.cos(theta[1]), np.cos(theta[0]) / np.cos(theta[1])]
			], dtype=float)

		omega_d = -Kp3*(theta - theta_d_array)
		v =  - Kd3*(omega - omega_d)
		omega_dot0 = (omega - omega_prev) / t_step
		Delta_Mc = J_matrix @ (v - omega_dot0)
		Mc = Delta_Mc + Mc_prev

		sol = solve_ivp(lambda t, x: INDI_dynamics(x, Mc), [t, t + t_step], x)
		omega_prev = omega.copy()
		Mc_prev = Mc.copy() 
		x = sol.y[:,-1]
		t += t_step
		all_t.append(sol.t)
		all_y.append(sol.y)


# Optionally, concatenate all results
full_t = np.concatenate(all_t)
full_y = np.hstack(all_y)

plt.figure(figsize=(10, 6))
plt.plot(full_t, np.rad2deg(full_y[0]), label='θ1 (Roll)')
plt.plot(full_t, np.rad2deg(full_y[1]), label='θ2 (Pitch)')
plt.plot(full_t, np.rad2deg(full_y[2]), label='θ3 (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('INDI Euler Attitude Angles vs. Time')
plt.legend()
plt.grid()
plt.show(block=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(full_t, np.rad2deg(full_y[0]), label='θ1 (Roll)', color='tab:blue')
axs[0].set_ylabel('Roll (°)')
axs[0].legend()
axs[0].grid()

axs[1].plot(full_t, np.rad2deg(full_y[1]), label='θ2 (Pitch)', color='tab:orange')
axs[1].set_ylabel('Pitch (°)')
axs[1].legend()
axs[1].grid()

axs[2].plot(full_t, np.rad2deg(full_y[2]), label='θ3 (Yaw)', color='tab:green')
axs[2].set_ylabel('Yaw (°)')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('3plotINDIeul.png')
plt.show(block=False)



#GAINS
Kp = np.array([0.0778, 0.0778, 0.0778])
Kd = np.array([0.7, 0.7, 0.7])

def INDI_dynamics_quat(x, Mc):
	q = x[0:4]
	omega = x[4:7]

	Omega_4_matrix = np.array([
		[0, omega[2], - omega[1], omega[0]], 
		[- omega[2], 0, omega[0], omega[1]],
		[omega[1], - omega[0] , 0, omega[2]],
		[- omega[0], - omega[1], - omega[2], 0]
		], dtype=float)


	Omega_matrix = np.array([
		[0, -omega[2], omega[1]],
		[omega[2], 0, -omega[0]],
		[-omega[1], omega[0], 0]
		], dtype=float)
	q_dot = 0.5*(Omega_4_matrix @ q)
	term1 = - J_inverse @ Omega_matrix @ J_matrix @ omega
	omega_dot = term1 + J_inverse @ Md + J_inverse @ Mc

	xdot = np.concatenate([q_dot, omega_dot])

	return xdot 

segments = [
    (0,      0,    99.9),
    (70,   100,    500),
    (-70,  500.1,  900),
    (0,    900.1, 1500)
]

# Lists to store results
all_t = []
all_y = []


t_step = 0.1 #s
q_init = R.from_euler('zyx', [25, 25, 25], degrees=True).as_quat()
x = np.array([q_init[0], q_init[1], q_init[2], q_init[3], 0, 0, 0])
omega_prev = np.array((0,0,0))
Mc_prev = np.array((0,0,0))
for theta_d_degrees, t_start, t_end in segments:
	t = t_start
	q_init_d = R.from_euler('zyx', [theta_d_degrees, theta_d_degrees, theta_d_degrees], degrees=True).as_quat()
	qc = np.array((q_init_d[0], q_init_d[1], q_init_d[2], q_init_d[3]))  
	omega_prev = x[3:6].copy()

	while t < t_end:
		q = x[0:4]
		omega = x[4:7]

		#quaternion control matrix
		M_matrix = np.array([
			[qc[3], qc[2], -qc[1], -qc[0]], 
			[-qc[2], qc[3], qc[0], -qc[1]],
			[qc[1], -qc[0], qc[3], - qc[2]],
			[qc[0], qc[1], qc[2], qc[3]]
			], dtype=float)
		qe = M_matrix @ q 
		omega_d = -Kp * qe[:3]
		v =  - Kd *(omega - omega_d)


		omega_dot0 = (omega - omega_prev) / t_step
		Delta_Mc = J_matrix @ (v - omega_dot0)
		Mc = Delta_Mc + Mc_prev

		sol = solve_ivp(lambda t, x: INDI_dynamics_quat(x, Mc), [t, t + t_step], x)
		omega_prev = omega.copy()
		Mc_prev = Mc.copy() 
		x = sol.y[:,-1]
		t += t_step
		all_t.append(sol.t)
		all_y.append(sol.y)


# Optionally, concatenate all results
full_t = np.concatenate(all_t)
full_y = np.hstack(all_y)

plt.figure(figsize=(10, 6))
plt.plot(full_t, full_y[0], label='θ1 (Roll)')
plt.plot(full_t, full_y[1], label='θ2 (Pitch)')
plt.plot(full_t, full_y[2], label='θ3 (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('INDI quaternions vs. Time')
plt.legend()
plt.grid()
plt.show(block=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(full_t, full_y[0], label='q1', color='tab:blue')
axs[0].set_ylabel('q1')
axs[0].legend()
axs[0].grid()

axs[1].plot(full_t, full_y[1], label='q2', color='tab:orange')
axs[1].set_ylabel('q2')
axs[1].legend()
axs[1].grid()

axs[2].plot(full_t, full_y[2], label='q3', color='tab:green')
axs[2].set_ylabel('q3')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid()


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('3plotINDIquat.png')
plt.show(block=False)

#END INDI

"""
#LINEARIZED CONTROLLER IN NONLINEAR DYNAMICS

#GAINS
Kp = np.array((0.5, 0.4, 0.007))
Kd = np.array((9.95, 9.95, 0.05))
def nonlinear_dynamics_pd(t, x):
    theta = x[0:3]
    omega = x[3:6]

    # kinematics matrix (converts omega to theta_dot)
    N = np.array([
        [1, np.sin(theta[0]) * np.tan(theta[1]), np.cos(theta[0]) * np.tan(theta[1])],
        [0, np.cos(theta[0]), -np.sin(theta[0])],
        [0, np.sin(theta[0]) / np.cos(theta[1]), np.cos(theta[0]) / np.cos(theta[1])]
    ], dtype=float)

    theta_dot = N @ omega  

    # Linear PD controller (from linearized design)
    Mc = -Kp * (theta - theta_d_array) - Kd * theta_dot

    # Omega matrix
    Omega = np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ], dtype=float)

    # Nonlinear dynamics
    omega_dot = J_inverse @ (Mc + Md - Omega @ (J_matrix @ omega))
    theta_dot = N @ omega

    return np.concatenate([theta_dot, omega_dot])

# Initial state
initial_deg = 25 #degrees
initial_rad = math.radians(initial_deg) #radians
x0 = np.array((initial_rad, initial_rad, initial_rad, 0, 0, 0))
theta = x0[0:3]

segments = [
    (0,      0,    99.9,  999),
    (70,   100,    500,  4000),
    (-70,  500.1,  900,  3999),
    (0,    900.1, 1500,  5999)
]

# Lists to store results
all_t = []
all_y = []

for theta_d_degrees, t_start, t_end, num_points in segments:
    # Update desired attitude
    theta_d = np.deg2rad(theta_d_degrees)
    theta_d_array = np.array((theta_d, theta_d, theta_d))  # You might set this globally or however needed

    # Time evaluation points
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, num_points)

    # Solve the system
    sol = solve_ivp(nonlinear_dynamics_pd, t_span, x0, t_eval=t_eval)

    # Store results
    all_t.append(sol.t)
    all_y.append(sol.y)

    # Update initial condition for time frame
    x0 = sol.y[:, -1]


full_t = np.concatenate(all_t)
full_y = np.hstack(all_y)

plt.figure(figsize=(10, 6))
plt.plot(full_t, np.rad2deg(full_y[0]), label='θ1 (Roll)')
plt.plot(full_t, np.rad2deg(full_y[1]), label='θ2 (Pitch)')
plt.plot(full_t, np.rad2deg(full_y[2]), label='θ3 (Yaw)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid()
plt.savefig('1plotlineulnon.png')
plt.show(block=False)


def nonlinear_dynamics_quat_pd(t,x):
	q = x[0:4]

	omega = x[4:7]

	Omega_4_matrix = np.array([
		[0, omega[2], - omega[1], omega[0]], 
		[- omega[2], 0, omega[0], omega[1]],
		[omega[1], - omega[0] , 0, omega[2]],
		[- omega[0], - omega[1], - omega[2], 0]
		], dtype=float)

	#quaternion control matrix
	M_matrix = np.array([
		[qc[3], qc[2], -qc[1], -qc[0]], 
		[-qc[2], qc[3], qc[0], -qc[1]],
		[qc[1], -qc[0], qc[3], - qc[2]],
		[qc[0], qc[1], qc[2], qc[3]]
		], dtype=float)
	qe = M_matrix @ q 
	qedot = 0.5 * (M_matrix @ Omega_4_matrix @ q)
	M_c = -Kp * qe[:3] - Kd * omega


	#3X3 skew matrix for dynamics
	Omega_matrix = np.array([
		[0, -omega[2], omega[1]],
		[omega[2], 0, -omega[0]],
		[-omega[1], omega[0], 0]
		], dtype=float)

	#MATRIX MULTIPLICATION
	f_x_1 = 0.5*(Omega_4_matrix @ q) 
	f_x_2 = - J_inverse @ Omega_matrix @ J_matrix @ omega + J_inverse @ Md + J_inverse @ M_c
	xdot = np.concatenate([f_x_1, f_x_2])


	return xdot 

segments = [
    (0,      0,    99.9,  999),
    (70,   100,    500,  4000),
    (-70,  500.1,  900,  3999),
    (0,    900.1, 1500,  5999)
]

all_t_quat = []
all_y_quat = []
Kp = np.array([0.005, 0.004, 0.0032])
Kd = np.array([0.12, 0.12, 0.12])


q_init = R.from_euler('zyx', [25, 25, 25], degrees=True).as_quat()
x0 = np.array([q_init[0], q_init[1], q_init[2], q_init[3], 0, 0, 0])
for theta_d_degrees, t_start, t_end, num_points in segments:
    # Update desired angle globally or pass it in appropriately
    q_init_d = R.from_euler('zyx', [theta_d_degrees, theta_d_degrees, theta_d_degrees], degrees=True).as_quat()
    qc = np.array((q_init_d[0], q_init_d[1], q_init_d[2], q_init_d[3]))  # You might set this globally or however needed

    # Time evaluation points
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, num_points)

    # Solve the system
    sol = solve_ivp(nonlinear_dynamics_quat_pd, t_span, x0, t_eval=t_eval)

    # Store results
    all_t_quat.append(sol.t)
    all_y_quat.append(sol.y)

    # Update initial condition for next segment
    x0 = sol.y[:, -1]


full_t = np.concatenate(all_t_quat)
full_y = np.hstack(all_y_quat)

#plot quaternion NDI
plt.figure(figsize=(10, 6))
plt.plot(full_t, full_y[0], label='q1')
plt.plot(full_t, full_y[1], label='q2')
plt.plot(full_t, full_y[2], label='q3')
plt.xlabel('Time (s)')
plt.ylabel('q')
plt.legend()
plt.grid()
plt.savefig('1plotlinquatnon.png')
plt.show(block=False)
"""