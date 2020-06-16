
# Data io :
import json
import toml

# Numerical :
import numpy as np
import pandas as pd

# Plotting :
import matplotlib.pyplot as plt
import seaborn as sns

# Stats :
import scipy.stats as sts
import scipy.optimize as sopt

# System utilities :
import glob
import os
import sys

# Type annotations :
from typing import Dict, List, Any, NoReturn, Callable, Union, Optional

# Local import :
from customobjs import objdict


DEFAULT_CONFIG_FILE = "regression_config.toml"

"""
# First legacy block
# Leer el rango de densidades y la lista de directorios
inf, sup  = np.loadtxt("slopes.txt", max_rows=1, unpack=True)
directorios = np.loadtxt("slopes.txt", dtype=str, skiprows=1)
temps = np.loadtxt("slopes.txt", skiprows=1)
"""

def parse_config(filename: str) -> Dict[str,Any]:
    """
        Parse the config
    """
    if "toml" in filename:
        with open(filename, "r") as f:
            _config = toml.load(f, _dict=objdict)
    else:
        raise Exception("Invalid config file format, provide a .toml or .json file.")

    _config.data.update({
        "temperatures": list(map(float, _config.data.directories))
    })

    return _config
##

def exit_explain_usage():
    """
        Pretty self explanatory, isn't it ?
    """
    print(f"\n\nUsage: \n $ python {sys.argv[0]} config_file")
    print(f"Config file should be .toml format : https://github.com/toml-lang/toml\n\n")
    exit()
##


def main(config_file):

    # Cargar los datos
    config = parse_config(config_file)

    sup = config.regression.rho.sup
    inf = config.regression.rho.inf
    directorios = config.data.directories
    temps = config.data.temperatures
    regex = config.data.glob
    nfiles = config.data.datfiles

    slopes = []
    error = np.array([])

    for direct in directorios:
        lista_arch = sorted(glob.glob(f"{direct}/{regex}"))[:nfiles]

        # Un arreglo para todos los datos
        x_vals, y_vals_total = np.loadtxt(lista_arch[0], usecols=(0, 1), unpack=True)
        data = pd.read_csv(lista_arch[0], sep="\s+", usecols=(0, 1), names=["x", "y"])

        for i in lista_arch[1:]:
            # Abrir el archivo, solo importan las primeras dos columnas
            y_vals = pd.read_csv(i, sep="\s+", usecols=[1], names=["y"], squeeze=True)
            #__, y_vals = np.loadtxt(i, usecols=(0, 1), unpack=True)
            # Añadir sola la segunda columna para luego promediar
            y_vals_total += y_vals
            data.y += y_vals

        # Realizar el promedio de todos los datos
        y_vals = y_vals_total / len(lista_arch)
        data.y = data.y.apply(lambda x: x / len(lista_arch))
        #data["error"] =
        # Agregar el error por directorio
        error = np.append(error, y_vals.std())

        # Obtener indice del arreglo donde se va a truncar
        i, = np.where(x_vals == inf)
        j, = np.where(x_vals == sup)
        # Cortar x_vals según los límites impuestos
        x_vals = x_vals[i.item():j.item()]
        # Cortar y_vals según los límites
        y_vals = y_vals[i.item():j.item()]

        # Crear el modelo de regresión lineal
        m, intercept, r, p, err = sts.linregress(x_vals, y_vals)
        x_expect = x_vals*m + intercept
        slopes.append(m)
        # Mostrar datos estadísticos
        chisq, p = sts.chisquare(y_vals, f_exp=x_expect)

    # Graficar todo
    np.savetxt('data_slopes_woo.dat', np.c_[(temps, slopes, error)], delimiter=' ')
    mslope, intercept, r, p, err = sts.linregress(temps, slopes, )
    # Hacer esto una funci
    def x_slopes(x): return x*mslope + intercept
    # Imprimir el cero
    print(sopt.root(x_slopes, -0.1).x)
    x_vals = np.linspace(1.26, 1.37)
    print('Lista de errores', error)
    plt.plot(x_vals, x_slopes(x_vals))
    plt.errorbar(temps, slopes, yerr=error, fmt='o')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        if DEFAULT_CONFIG_FILE in os.listdir("."):
            print(f"Config file not specified, using default `{DEFAULT_CONFIG_FILE}`")
            main(DEFAULT_CONFIG_FILE)
        else:
            print(f"Config file not specified, using default `{DEFAULT_CONFIG_FILE}`...")
            print(f"Default `{DEFAULT_CONFIG_FILE}` not found in {os.path.abspath('.')}")
            exit_explain_usage()
    else:
        if "toml" in sys.argv[1]:
            main(sys.argv[1])
        else:
            exit_explain_usage()
##
