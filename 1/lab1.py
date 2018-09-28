# encoding: utf-8
# -*- coding: utf-8 -*-
import argparse
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import *
from scipy.special import gamma, gammainc


def get_start_moment(*args):
    distribution_type = args[0]
    args = args[1:]
    if distribution_type == 'Exp':
        # Экспоненциальное распределение
        return [1 / args[0], 1 / args[0]]
    elif distribution_type == 'U':
        # Равномерное распределение
        return [(args[0] + args[1]) / 2, (args[1] - args[0]) / (2 * sqrt(3))]
    elif distribution_type == 'Г':
        # Гамма распределение
        return [args[0] * args[1], sqrt(args[0] * args[1])]
    elif distribution_type == 'TN':
        # Усеченное нормальное распределение
        c = 1 / (0.5 + math.erf(args[0] / args[1]))
        k = c / sqrt(2 * pi) * exp(-pow(args[0], 2) / (2 * pow(args[1], 2)))
        return [args[0] + k * args[1], args[1] * sqrt(1 + k * args[0] / args[1] - pow(k, 2))]
    elif distribution_type == 'R':
        # Релея распределение
        return [sqrt(pi / (4 * args[0])), sqrt((4 - pi) / (4 * args[0]))]
    elif distribution_type == 'W':
        # Вейбулла распределение
        return [args[1] * gamma(1 + 1 / args[0]),
                args[1] * sqrt(gamma(1 + 2 / args[0]) - gamma(1 + 1 / args[0]) ** 2)]
    elif distribution_type == 'N':
        # Нормальное распределение
        return [args[0], args[1]]
    return None


def p(t, *args):
    distribution_type = args[0]
    args = args[1:]
    if distribution_type == 'Exp':
        # Экспоненциальное распределение
        return array([exp(-args[0] * j) for j in t])
    elif distribution_type == 'U':
        # Равномерное распределение
        return array([1 if j < args[0] else \
                          (args[1] - j) / (args[1] - args[0]) if args[0] <= j <= args[1] else \
                              0 if j > args[1] else None for j in t])
    elif distribution_type == 'Г':
        # Гамма распределение
        return array([1 - gammainc(args[0], j / args[1]) / gamma(args[0]) for j in t])
    elif distribution_type == 'TN':
        # Усеченное нормальное распределение
        c = 1 / (0.5 + math.erf(args[0] / args[1]))
        return array([c * (0.5 - math.erf((j - args[0]) / args[1])) for j in t])
    elif distribution_type == 'R':
        # Релея распределение
        return array([exp(-args[0] * pow(j, 2)) for j in t])
    elif distribution_type == 'W':
        # Вейбулла распределение
        return array([exp(-pow(j / args[1], args[0])) for j in t])
    elif distribution_type == 'N':
        # Нормальное распределение
        return array([0.5 - math.erf((j - args[0]) / args[1]) for j in t])
    return None


def f(t, *args):
    distribution_type = args[0]
    args = args[1:]
    if distribution_type == 'Exp':
        # Экспоненциальное распределение
        return array([args[0] * exp(-args[0] * j) for j in t])
    elif distribution_type == 'U':
        # Равномерное распределение
        return array([1 / (args[1] - args[0]) if args[0] <= j <= args[1] \
                          else 0 if j < args[0] or j > args[1] \
            else None for j in t])
    elif distribution_type == 'Г':
        # Гамма распределение
        return array([pow(j, args[0] - 1) / (pow(args[1], args[0]) * gamma(args[0])) * exp(-j / args[1]) for j in t])
    elif distribution_type == 'TN':
        # Усеченное нормальное распределение
        c = 1 / (0.5 + math.erf(args[0] / args[1]))
        return array([(c / (args[1] * sqrt(2 * pi))) * exp(-pow(j - args[0], 2) / (2 * pow(args[1], 2))) for j in t])
    elif distribution_type == 'R':
        # Релея распределение
        return array([2 * args[0] * j * exp(-args[0] * pow(j, 2)) for j in t])
    elif distribution_type == 'W':
        # Вейбулла распределение
        return array(
            [((args[0] * pow(j, args[0] - 1)) / pow(args[1], args[0])) * exp(-pow(j / args[1], args[0])) for j in t])
    elif distribution_type == 'N':
        # Нормальное распределение
        return array([1 / (args[1] * sqrt(2 * pi)) * exp(-pow(j - args[0], 2) / (2 * pow(args[1], 2))) for j in t])
    return None


def parse_csv(variant=0):
    with open("variants.csv", encoding="UTF-8") as f:
        try:
            s = re.sub("[\t| ]", '', f.read()).split('\n')[variant].split(";")[1:]
            objects = []
            for i in s:
                parsed = re.sub("[\t| |\)]", '', i)
                parsed = re.split("[( | ,]", parsed)
                if '∙' in i:
                    parsed = parsed[0], *[float(k) for k in re.split("[∙|-]", parsed[1])]
                    parsed = parsed[0], parsed[1] * parsed[2] ** (- parsed[3] if '-' in i else parsed)
                else:
                    parsed = parsed[0], *[float(k) for k in parsed[1:]]
                objects.append(parsed)
            return objects
        except IndexError:
            parse_csv()
    return None


def plot(x, y, verbose):
    dpi = 300
    fig = plt.figure(dpi=dpi)
    mpl.rcParams.update({'font.size': 10})
    plt.axis([min(x), max(x), min(y), max(y)])
    plt.title('')
    plt.xlabel('t')
    plt.ylabel(verbose)
    plt.plot(x, y, color='blue', linewidth=1,
             label=verbose)
    plt.legend(loc='upper right')
    fig.savefig('{}/{}.png'.format(verbose[0], verbose))
    plt.close()


parser = argparse.ArgumentParser(description='Обработка вариантов по ЛР1')
parser.add_argument('-v', '--variant', type=int, default=0, help='Вариант')
inputs = parser.parse_args()
max_t, interval = 3000, 100
objects = parse_csv(inputs.variant - 1)
t = array([i for i in range(0, max_t + interval, interval)])
file = open("out.txt", "w", encoding='UTF-8')
lc = 0
for i, obj in enumerate(objects):
    moments = get_start_moment(*obj)
    P = p(t, *obj)
    F = f(t, *obj)
    L = F / P
    Q = 1 - P
    lc += L
    plot(t, P, "P_{}(t)".format(i + 1))
    plot(t, F, "f_{}(t)".format(i + 1))
    plot(t, L, "λ_{}(t)".format(i + 1))
    plot(t, Q, "Q_{}(t)".format(i + 1))
    file.write("Объект {}".format(i + 1))
    file.write('\n')
    file.write('\t'.join(["moments:", *["%.4f" % j for j in moments]]))
    file.write('\n')
    file.write('\t'.join(["t:", *["%.4f" % j for j in t]]))
    file.write('\n')
    file.write('\t'.join(["P:", *["%.4f" % j for j in P]]))
    file.write('\n')
    file.write('\t'.join(["F:", *["%.4f" % j for j in F]]))
    file.write('\n')
    file.write("\t".join(["λ:", *["%.4f" % j for j in L]]))
    file.write('\n')
    file.write("\t".join(["Q:", *["%.4f" % j for j in Q]]))
    file.write('\n')
plot(t, lc, "λ_с(t)")
file.write("\t".join(["λ_с:", *["%.4f" % j for j in lc]]))
file.close()
