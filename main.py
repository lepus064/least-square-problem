from time import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg
import autograd.numpy as np
from autograd import jacobian
import imageio

if __name__ == '__main__':

    # parameters
    np.random.seed(0)
    ground_truth = np.array([1.2, 0.4, 2.5])
    initial_guess = np.array([0.5, 1.5, 1.5])
    test_func = lambda x, para: np.exp(para[0] * x * x + para[1] * x + para[2])
    iter_time = 50

    # ground truth curve for comparison
    curve_x = np.linspace(-1, 1, 1000)
    curve_y = test_func(curve_x, ground_truth)

    # generate noise data
    total = 300
    generated_x = np.random.rand(total)*2-1
    noise = np.random.standard_normal(total)/2
    generated_y = test_func(generated_x, ground_truth) + noise


    vector_func = lambda para: np.array(
        [y - test_func(x, para) for y, x in zip(generated_y, generated_x)])

    auto_J = jacobian(vector_func)

    def manual_J(para):
        f = test_func(generated_x, para)
        j0 = -generated_x * generated_x * f
        j1 = -generated_x * f
        j2 = -f
        return np.vstack([j0, j1, j2]).T

    # matrix size
    # para 3, 1
    # test_func 200, 1
    # Jacobian 200, 3
    # Hessian 3, 3

    # test_functions = [manual_J, auto_J]
    test_functions = [manual_J]
    test_result = []
    time_cost = []
    params_record = [initial_guess.copy()]
    residual_record = []

    for jacobian_func in test_functions:
        current_para = initial_guess
        start_time = time()
        for iter in range(iter_time):
            residual = vector_func(current_para)
            residual_record.append(np.linalg.norm(residual))
            Jx = jacobian_func(current_para)

            H = Jx.T @ Jx
            B = -Jx.T @ residual
            dx = scipy.linalg.solve(H, B)
            if(np.linalg.norm(dx) < 1e-8):
                break
            current_para += dx
            params_record.append(current_para.copy())

        time_cost.append(time() - start_time)
        print("time: ", time_cost[-1])
        test_result.append(current_para.copy())

    # plotting
    ims_names = []
    current_para_name = ['initial guess']+['current parameters' for _ in params_record]
    text_x = 270
    for i, para in enumerate(params_record):
        plt.title(r"$Target\ function:\ f\ (x)\ =\ e^{ax^2+bx+c}$",
          math_fontfamily='stixsans', size=14)
        plt.scatter(generated_x, generated_y, marker='.', label='data with noise', linewidths=0.5)
        plt.plot(curve_x, curve_y, color='r', label='ground truth')
        y_ = test_func(curve_x, para)
        plt.plot(curve_x, y_, color='g', label = 'current result')
        plt.legend(loc='upper left')
        plt.annotate('ground truth \na = 1.200, b = 0.400, c = 2.500',
            xy=(text_x, 385), xycoords='figure pixels')
        para_str = '\na = {0[0]:.3f}, b = {0[1]:.3f}, c = {0[2]:.3f}'.format(para)
        plt.annotate(current_para_name[i]+para_str,
            xy=(text_x, 345), xycoords='figure pixels')
        plt.annotate("total residual: {0:.3f}".format(residual_record[i]),
            xy=(text_x, 320), xycoords='figure pixels')
        img_name = str(i)+".png"
        plt.ylim(5, 60)
        plt.savefig(img_name)
        ims_names.append(img_name)
        plt.close()

    ims_names = [ims_names[0]]+ims_names
    
    # Build GIF
    with imageio.get_writer('example.gif', mode='I', duration=0.5) as writer:
        for filename in ims_names:
            image = imageio.imread(filename)
            writer.append_data(image)