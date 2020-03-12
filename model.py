import numpy as np
from scipy.optimize import minimize

def model(x, D_x, D_y, big_lambda, small_lambda, C, v, v_w, tau, Dtau):
    s, p_x, p_y, d_x, d_y, H, H_min = x
    s_x = p_x * s
    s_y = p_y * s
    alpha_x = d_x/D_x
    alpha_y = d_y/D_y

    #Equations
    L = ((D_x * D_y)/(2 * s_x * s_y)) * (s_x + s_y) * (1 + alpha_x * alpha_y) + ((D_x * D_y)/(2 * s_x * s_y)) * (s_x - s_y) * (alpha_y - alpha_x)
    V = ((2 * alpha_x * D_x * D_y)/(s_y * H)) * (1 + (D_x/(2 * D_y)) * (1 - alpha_x)) + ((2 * alpha_y * D_x * D_y)/(s_x * H)) * (1 + (D_y/(2 * D_x)) * (1 - alpha_y)) 

    O_y_1 = (big_lambda * H * (1 + alpha_x) * (1 - alpha_y))/(4 * alpha_x * D_x)
    O_y_2 = ((big_lambda * H * np.power(1 - alpha_x, 2) * np.power(1 + alpha_y, 2))/32) + (big_lambda * s_y * H * (4 - np.power(1 + alpha_y, 2) * np.power(1 - alpha_x, 2) - 2 * np.power(alpha_x, 2) * np.power(alpha_y, 2)))/(8 * alpha_x * D_x)
    O_y = max(O_y_1, O_y_2)

    O_x_1 = (big_lambda * H * (1 + alpha_y) * (1 - alpha_x))/(4 * alpha_y * D_y)
    O_x_2 = ((big_lambda * H * np.power(1 - alpha_y, 2) * np.power(1 + alpha_x, 2))/32) + (big_lambda * s_x * H * (4 - np.power(1 + alpha_x, 2) * np.power(1 - alpha_y, 2) - 2 * np.power(alpha_x, 2) * np.power(alpha_y, 2)))/(8 * alpha_y * D_y)
    O_x = max(O_x_1, O_x_2)

    A = (((s_x + s_y)/4) + s/2)/v_w

    P_0 = ((s_x * D_x + s_y * D_y)/(2 * D_x * D_y)) * (1 + alpha_x * alpha_y) + ((s_x * D_x - s_y * D_y)/(2 * D_x * D_y)) * (alpha_y - alpha_x) - (alpha_x * alpha_y * s_x * s_y)/(D_x * D_y)
    P_1 = (s_y/(2 * D_x)) * (-alpha_y + np.power(alpha_y, 2) - 3 * alpha_x * alpha_y + alpha_x * np.power(alpha_y, 2)) + (s_x/(2 * D_y)) * (-alpha_x + np.power(alpha_x, 2) - (3 * alpha_x * alpha_y) + (alpha_y * np.power(alpha_x, 2))) + (1/2) * (1 - np.power(alpha_y, 2) - np.power(alpha_x, 2) + (4 * alpha_x * alpha_y) - np.power(alpha_x, 2) * np.power(alpha_y, 2)) + (s_x * s_y * alpha_x * alpha_y)/(D_x * D_y)
    P_2 = (1/2) * (1 - (4 * alpha_x * alpha_y) + np.power(alpha_x, 2) + np.power(alpha_y, 2) + np.power(alpha_x, 2) * np.power(alpha_y, 2)) - (s_y/(2 * D_x)) * np.power(1 - alpha_y, 2) * (1 + alpha_x) - (s_x/(2 * D_y)) * np.power(1 - alpha_x, 2) * (1 + alpha_y)

    W = ((H/(6 * alpha_x) * (1 - np.power(alpha_x, 3)) * (1 - alpha_y)/(1 - alpha_x)) + (H/(6 * alpha_y) * (1 - np.power(alpha_y, 3)) * (1 - alpha_x)/(1 - alpha_y)) + (alpha_x * alpha_y * (H/2))) * (1 + P_1) + (H/2) * P_2
    e_T = P_1 + 2 * P_2

    E = ((np.power(alpha_y, 2) * np.power(D_y, 2) + np.power(alpha_x, 2) * np.power(D_x, 2) + 4 * alpha_x * alpha_y * D_x * D_y)/(4 * (alpha_x * D_x + alpha_y * D_y)) + ((alpha_x * D_x + alpha_y * D_y)/(12 * alpha_x * alpha_y * D_x * D_y)) * (1 - (alpha_x * alpha_y)/2)) * (1 - np.power(alpha_x, 2) * np.power(alpha_y, 2)) + (1/3) * (alpha_x * D_x + alpha_y * D_y) * (np.power(alpha_x, 2) * np.power(alpha_y, 2)) + (1/4) * (D_x * (2 - 3 * alpha_x + np.power(alpha_x, 3) + D_y * (2 - 3 * alpha_y + np.power(alpha_y, 3))))

    reciprocal_v_c = ((1/v) + (tau/s)) + (1 + e_T) * (big_lambda/V) * Dtau # or (big_lambda/(V * Dtau))

    M = V * reciprocal_v_c
    T = E * reciprocal_v_c

    #z_a = pi_V * V + pi_M + M + pi_L * L
    #z_u = A + W + T + (delta/v_w) * e_T

    #return z_a + z_u

def main():
    #test = model([])
    #print(test)

if __name__ == "__main__":
    main()