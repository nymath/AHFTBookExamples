import typing

import numpy as np
from scipy.special import gamma, hyp1f1
from scipy.optimize import root
import matplotlib.pyplot as plt

__all__ = [
    "default_params",
    "F_plus", "F_plus_grad", "F_minus", "F_minus_grad",
    "optimal_threshold_exit_long", "optimal_threshold_exit_short",
    "u_optimal_threshold_exit_long", "u_optimal_threshold_exit_short",
    "H_plus", "H_minus", "u_H_plus", "u_H_minus",
    "figure_11_4_0", "figure_11_4_1",
]

default_params = {
    "rho": 0.01,
    "kappa": 0.5,
    "sigma": 0.5,
    "theta": 0,
    "c": 0.01,
}


def F_plus(epsilon, rho, kappa, sigma, theta):
    return 2 ** (-1 + rho / (2 * kappa)) * \
        (gamma(rho / (2 * kappa)) * hyp1f1(rho / (2 * kappa), 1 / 2, (kappa * (epsilon - theta) ** 2) / sigma ** 2) +
         2 * np.sqrt(kappa / sigma ** 2) * (epsilon - theta) * gamma((kappa + rho) / (2 * kappa)) *
         hyp1f1((kappa + rho) / (2 * kappa), 3 / 2, (kappa * (epsilon - theta) ** 2) / sigma ** 2))


def F_plus_grad(epsilon, rho, kappa, sigma, theta):
    return 2 ** (1 / 2 + 1 / 2 * (-1 + rho / kappa)) * np.sqrt(kappa / sigma ** 2) * \
        (gamma((kappa + rho) / (2 * kappa)) * hyp1f1((kappa + rho) / (2 * kappa), 1 / 2,
                                                     (kappa * (epsilon - theta) ** 2) / sigma ** 2) +
         2 * np.sqrt(kappa / sigma ** 2) * (epsilon - theta) * gamma(1 + rho / (2 * kappa)) *
         hyp1f1(1 + rho / (2 * kappa), 3 / 2, (kappa * (epsilon - theta) ** 2) / sigma ** 2))


def F_minus(epsilon, rho, kappa, sigma, theta):
    return 2 ** (-1 + rho / (2 * kappa)) * \
        (gamma(rho / (2 * kappa)) * hyp1f1(rho / (2 * kappa), 1 / 2, (kappa * (epsilon - theta) ** 2) / sigma ** 2) +
         2 * np.sqrt(kappa / sigma ** 2) * (-epsilon + theta) * gamma((kappa + rho) / (2 * kappa)) *
         hyp1f1((kappa + rho) / (2 * kappa), 3 / 2, (kappa * (epsilon - theta) ** 2) / sigma ** 2))


def F_minus_grad(epsilon, rho, kappa, sigma, theta):
    return -2 ** (1 / 2 + 1 / 2 * (-1 + rho / kappa)) * np.sqrt(kappa / sigma ** 2) * \
        (gamma((kappa + rho) / (2 * kappa)) * hyp1f1((kappa + rho) / (2 * kappa), 1 / 2,
                                                     (kappa * (epsilon - theta) ** 2) / sigma ** 2) +
         2 * np.sqrt(kappa / sigma ** 2) * (-epsilon + theta) * gamma(1 + rho / (2 * kappa)) *
         hyp1f1(1 + rho / (2 * kappa), 3 / 2, (kappa * (epsilon - theta) ** 2) / sigma ** 2))


def optimal_threshold_exit_long(x_0=1.5, rho=0.01, kappa=0.5, sigma=0.5, theta=1, c=0.01):
    aux_func = lambda x: (x - c) * F_plus_grad(x, rho, kappa, sigma, theta) - F_plus(x, rho, kappa, sigma, theta)
    return root(aux_func, x0=np.array(x_0))["x"]


def optimal_threshold_exit_short(x_0=-.5, rho=0.01, kappa=0.5, sigma=0.5, theta=1, c=0.01):
    aux_func = lambda x: (x + c) * F_minus_grad(x, rho, kappa, sigma, theta) - F_minus(x, rho, kappa, sigma, theta)
    return root(aux_func, x0=np.array(x_0))["x"]


u_optimal_threshold_exit_long = np.vectorize(optimal_threshold_exit_long)
u_optimal_threshold_exit_short = np.vectorize(optimal_threshold_exit_short)


def H_plus(epsilon, rho, kappa, sigma, theta, c):
    optimal_epsilon = optimal_threshold_exit_long(x_0=1.5, rho=rho, kappa=kappa, sigma=sigma, theta=theta, c=c)
    if epsilon < optimal_epsilon:
        result = F_plus(epsilon, rho, kappa, sigma, theta) / F_plus(optimal_epsilon, rho, kappa, sigma, theta) * (
                    optimal_epsilon - c)
    else:
        result = epsilon - c
    return result


def H_minus(epsilon, rho, kappa, sigma, theta, c):
    optimal_epsilon = optimal_threshold_exit_short(x_0=-.5, rho=rho, kappa=kappa, sigma=sigma, theta=theta, c=c)
    if epsilon > optimal_epsilon:
        result = - F_minus(epsilon, rho, kappa, sigma, theta) / F_minus(optimal_epsilon, rho, kappa, sigma, theta) * (
                    optimal_epsilon + c)
    else:
        result = -(epsilon + c)
    return result


def G(epsilon, rho, kappa, sigma, theta, c):
    pass


u_H_plus = np.vectorize(H_plus)
u_H_minus = np.vectorize(H_minus)


def figure_11_4_0(default_params: typing.Dict[str, typing.Union[float, np.ndarray]]):
    params = default_params.copy()
    fig, ax = plt.subplots()
    for k in [0.5, 1, 2, 4, 8]:
        params.update({"kappa": k})
        x_values = np.arange(0.1, 1.5, 0.1)
        ax.plot(x_values, u_H_plus(x_values, **params), label="kappa={:.2f}".format(k))
    ax.set_xlabel("epsilon")
    ax.set_ylabel("H_plus")
    ax.grid(True)
    ax.legend()
    # If fig.show() does not work, try plt.show()
    fig.show()


def figure_11_4_1(default_params: typing.Dict[str, typing.Union[float, np.ndarray]]):
    params = default_params.copy()
    fig, ax = plt.subplots()
    for k in [0.01, 0.02, 0.04, 0.08, 0.16]:
        params.update({"rho": k})
        x_values = np.arange(0.1, 1.5, 0.1)
        ax.plot(x_values, u_H_plus(x_values, **params), label="rho={:.2f}".format(k))
    ax.set_xlabel("epsilon")
    ax.set_ylabel("H_plus")
    ax.grid(True)
    ax.legend()
    # If fig.show() does not work, try plt.show()
    fig.show()


if __name__ == "__main__":
    kappa_values = np.array([0.5, 1.0, 2.0, 4.0])
    params = {
        "rho": 0.01,
        "kappa": 0.5,
        "sigma": 0.5,
        "theta": .0,
        "c": 0.01,
    }

    figure_11_4_1(params)
    roots = u_optimal_threshold_exit_long(x_0=1.5, rho=0.01, kappa=kappa_values, sigma=0.5, theta=1, c=0.01)
    print(roots)

    roots = u_optimal_threshold_exit_short(x_0=-.5, rho=0.01, kappa=kappa_values, sigma=0.5, theta=1, c=0.01)
    print(roots)
