from numpy import *


def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return (new_b, new_m)


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def run():
    # step1:- collect data
    points = genfromtxt('CSV_DATA/data.csv', delimiter=',', dtype='int64')

    # step2:-  define our hyperparamters
    # how fast our model converge
    learning_rate = 0.001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # step3:- train our model
    print("Starting gradient descent at b = {}, m = {}, error = {}".format(initial_b, initial_m,
                                                                           compute_error_for_line_given_points(
                                                                               initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {} iterations b = {}, m = {}, error = {}".format(num_iterations, b, m,
                                                                   compute_error_for_line_given_points(b, m, points)))


if __name__ == '__main__':
    run()
