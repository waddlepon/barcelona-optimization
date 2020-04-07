import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def model(x, D_x, D_y, big_lambda, small_lambda, C, v, v_w, tau, Dtau, delta, pi_V, pi_M, pi_L):
    s, p_x, p_y, d_x, d_y, H = x

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
    P_1 = (s_y/(2 * D_x)) * (-alpha_y + np.power(alpha_y, 2) - (3 * alpha_x * alpha_y) + alpha_x * np.power(alpha_y, 2)) + (s_x/(2 * D_y)) * (-alpha_x + np.power(alpha_x, 2) - (3 * alpha_x * alpha_y) + (alpha_y * np.power(alpha_x, 2))) + (1 - np.power(alpha_y, 2) - np.power(alpha_x, 2) + (4 * alpha_x * alpha_y) - np.power(alpha_x, 2) * np.power(alpha_y, 2))/2 + (s_x * s_y * alpha_x * alpha_y)/(D_x * D_y)
    P_2 = (1 - (4 * alpha_x * alpha_y) + np.power(alpha_x, 2) + np.power(alpha_y, 2) + np.power(alpha_x, 2) * np.power(alpha_y, 2))/2 - (s_y/(2 * D_x)) * np.power(1 - alpha_y, 2) * (1 + alpha_x) - (s_x/(2 * D_y)) * np.power(1 - alpha_x, 2) * (1 + alpha_y)

    W = ((H/(6 * alpha_x) * (1 - np.power(alpha_x, 3)) * (1 - alpha_y)/(1 - alpha_x)) + (H/(6 * alpha_y) * (1 - np.power(alpha_y, 3)) * (1 - alpha_x)/(1 - alpha_y)) + (alpha_x * alpha_y * (H/2))) * (1 + P_1) + (H/2) * P_2
    e_T = P_1 + 2 * P_2

    E_1 = ((np.power(alpha_y, 2) * np.power(D_y, 2) + np.power(alpha_x, 2) * np.power(D_x, 2) + 4 * alpha_x * alpha_y * D_x * D_y)/(4 * (alpha_x * D_x + alpha_y * D_y)) + ((alpha_x * D_x + alpha_y * D_y)/(12 * alpha_x * alpha_y * D_x * D_y)) * (1 - (alpha_x * alpha_y)/2)) * (1 - np.power(alpha_x, 2) * np.power(alpha_y, 2))
    E_2 = ((alpha_x * D_x + alpha_y * D_y) * (np.power(alpha_x, 2) * np.power(alpha_y, 2)))/3 
    E_3 = ((D_x * (2 - 3 * alpha_x + np.power(alpha_x, 3)) + D_y * (2 - 3 * alpha_y + np.power(alpha_y, 3))))/4
    E = E_1 + E_2 + E_3

    reciprocal_v_c = ((1/v) + (tau/s)) + (1 + e_T) * (big_lambda/V) * Dtau

    M = V * reciprocal_v_c
    T = E * reciprocal_v_c

    z_a = pi_V * V + pi_M * M + pi_L * L
    z_u = A + W + T + (delta/v_w) * e_T
    
    #debug
    '''
    print("L: " + str(L))
    print("V: " + str(V))
    print("O_y_1: " + str(O_y_1))
    print("O_y_2: " + str(O_y_2))
    print("O_y: " + str(O_y))
    print("O_x_1: " + str(O_x_1))
    print("O_x_2: " + str(O_x_2))
    print("O_x: " + str(O_x))
    print("A: " + str(A))
    print("P_0: " + str(P_0))
    print("P_1: " + str(P_1))
    print("P_2: " + str(P_2))
    print("W: " + str(W))
    print("e_T: " + str(e_T))
    print("E: " + str(E))
    print("1/v_c: " + str(reciprocal_v_c))
    print("M: " + str(M))
    print("T: " + str(T))
    print("z_a: " + str(z_a))
    print("z_u: " + str(z_u))
    '''

    return z_a + z_u

def main():
    #test case off barcelona
    '''
    test = model([0.65, 2.0, 1.0, 8.5, 4.25, 0.05], (10.0, 5.0, 45000.0, 20000.0, 150.0, 21.4, 2.0, 0.008611111111, 0.0004166666667, 0.03, 5.2, 60.2, 80.0))
    print(test)
    '''

    #constants
    D_x, D_y, big_lambda, small_lambda, C, v, v_w, tau, Dtau, delta, pi_V, pi_M, pi_L = 10.0, 5.0, 45000.0, 20000.0, 150.0, 21.4, 2.0, 0.008611111111, 0.0004166666667, 0.03, 5.2, 60.2, 80.0
    H_min = 0.05
    N = 13.07692308 

    #constraints
    def constraint_func(x):
        s, p_x, p_y, d_x, d_y, H = x
        s_x = p_x * s
        s_y = p_y * s
        alpha_x = d_x/D_x
        alpha_y = d_y/D_y

        O_y_1 = (big_lambda * H * (1 + alpha_x) * (1 - alpha_y))/(4 * alpha_x * D_x)
        O_y_2 = ((big_lambda * H * np.power(1 - alpha_x, 2) * np.power(1 + alpha_y, 2))/32) + (big_lambda * s_y * H * (4 - np.power(1 + alpha_y, 2) * np.power(1 - alpha_x, 2) - 2 * np.power(alpha_x, 2) * np.power(alpha_y, 2)))/(8 * alpha_x * D_x)
        O_y = max(O_y_1, O_y_2)

        O_x_1 = (big_lambda * H * (1 + alpha_y) * (1 - alpha_x))/(4 * alpha_y * D_y)
        O_x_2 = ((big_lambda * H * np.power(1 - alpha_y, 2) * np.power(1 + alpha_x, 2))/32) + (big_lambda * s_x * H * (4 - np.power(1 + alpha_x, 2) * np.power(1 - alpha_y, 2) - 2 * np.power(alpha_x, 2) * np.power(alpha_y, 2)))/(8 * alpha_y * D_y)
        O_x = max(O_x_1, O_x_2)

        return np.array([s, alpha_x - s_x/D_x, alpha_y - s_y/D_y, H - H_min, C - O_x, C - O_y, N - ((alpha_x * D_x)/s_x + (alpha_y * D_y)/s_y)])

    constraint = NonlinearConstraint(constraint_func, 0, np.inf)
    
    res = minimize(model, [0.65, 2.0, 1.0, 8.5, 4.25, 0.05], method="trust-constr", constraints=constraint, args=(D_x, D_y, big_lambda, small_lambda, C, v, v_w, tau, Dtau, delta, pi_V, pi_M, pi_L))

    s, p_x, p_y, d_x, d_y, H = res.x
    print("s: " + str(s))
    print("p_x: " + str(p_x))
    print("p_y: " + str(p_y))
    print("d_x: " + str(d_x))
    print("d_y: " + str(d_y))
    print("H: " + str(H))
    print("z: " + str(res.fun))

if __name__ == "__main__":
    main()