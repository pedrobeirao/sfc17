from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import dates
import pandas as pd
import numpy as np
from scipy import interp
from sfc17 import tools
import datetime
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')


def plot_results(df_0, df_1, engine_no, model_fit, xgp, var1, var_power,
                 var_mass):
    """
    Plot modeled mass fuel flow for 2 fuel types.
    :param df_0: Data frame with data for one fuel type.
    :param df_1: Data frame with data for one fuel type.
    :param engine_no: Engine number.
    :param model: Array with modeled SFC.
    :param xgp: Array with power values corresponding to modeled SFC.
    :param var1:
    :param var_power: Variable correspoding to power.
    :param var_mass: riable correspoding to mass fuel flow.
    :return:
    """
    if len(var1[var1.str.contains('iso')]) > 0:
        df_0_iso = tools.sfc_iso(df_0, engine_no)
        ff1_iso = df_0_iso * df_0[var_power]
        plt.scatter(df_0_sort[var_power],
                    ff1_iso, color='red', marker='.', alpha=0.1)
        df_1_iso = tools.tools.sfc_iso(df_1, engine_no)
        ff2_iso = df_1_iso * df_1[var_power]
        plt.scatter(df_1[var_power],
                    ff2_iso, color='red', marker='.', alpha=0.1)
        sfoc1_iso = model_fit['model.ted.rig:sfc1.iso@engine.main.' + engine_no]
        sfoc2_iso = model_fit['model.ted.rig:sfc2.iso@engine.main.' + engine_no]
        plt.plot(xgp, sfoc1_iso * xgp, color='navy')
        plt.plot(xgp, sfoc2_iso * xgp, color='navy')
        plt.xlim((0, np.max(xgp) + 1))
        plt.ylim((0, np.max(sfoc1_iso * xgp) + 500))
    else:
        plt.scatter(df_0[var_power],
                    df_0[var_mass], color='red', marker='.', alpha=0.1)
        plt.scatter(df_1[var_power],
                    df_1[var_mass], color='red', marker='.', alpha=0.1)
        ff1 = model_fit['model.ted.rig:sfc1@engine.main.' + engine_no]
        ff2 = model_fit['model.ted.rig:sfc2@engine.main.' + engine_no]
        plt.plot(xgp, ff1 * xgp, color='navy')
        plt.plot(xgp, ff2 * xgp, color='navy')
        plt.xlim((0, np.max(xgp) + 1))
        plt.ylim((0, np.max(ff1 * xgp) + 500))


def plot_results_sfc(df_0, df_1, engine_no, model, xgp, var1, var_power,
                     var_mass):
    """
    Plot modeled SFC for 2 fuel types.
    :param df_0: Data frame with data for one fuel type.
    :param df_1: Data frame with data for one fuel type.
    :param engine_no: Engine number.
    :param model: Array with modeled SFC.
    :param xgp: Array with power values corresponding to modeled SFC.
    :param var1:
    :param var_power: Variable correspoding to power.
    :param var_mass: riable correspoding to mass fuel flow.
    :return:
    """
    if len(var1[var1.str.contains('iso')]) > 0:
        df_0_iso = tools.sfc_iso(df_0, engine_no)
        plt.scatter(df_0[var_power],
                    df_0_iso, color='red', marker='.', alpha=0.1)
        df_1_iso = tools.tools.sfc_iso(df_1, engine_no)
        plt.scatter(df_1[var_power],
                    df_1_iso, color='red', marker='.', alpha=0.1)
        sfoc1_iso = model['model.ted.rig:sfc1.iso@engine.main.' + engine_no]
        sfoc2_iso = model['model.ted.rig:sfc2.iso@engine.main.' + engine_no]
        plt.plot(xgp, sfoc1_iso, color='navy')
        plt.plot(xgp, sfoc2_iso, color='navy')
        plt.xlim((0, np.max(xgp) + 1))
    else:
        plt.scatter(df_0[var_power],
                    df_0[var_mass] / df_0[var_power], color='red', marker='.',
                    alpha=0.1)
        plt.scatter(df_1[var_power],
                    df_1[var_mass] / df_1[var_power], color='red', marker='.',
                    alpha=0.1)
        ff1 = model['model.ted.rig:sfc1@engine.main.' + engine_no]
        ff2 = model['model.ted.rig:sfc2@engine.main.' + engine_no]
        plt.plot(xgp, ff1, color='navy')
        plt.plot(xgp, ff2, color='navy')
        plt.xlim((0, np.max(xgp) + 1))


def plot_results_single(df, engine_no, model, xgp, var1, var_power, var_mass):
    """
    Plot modeled mass fuel flow for 1 fuel type.
    :param df: Data frame with cleaned data.
    :param engine_no: Engine number.
    :param model: Array with modeled SFC.
    :param xgp: Array with power values corresponding to modeled SFC.
    :param var1:
    :param var_power: Variable correspoding to power.
    :param var_mass: riable correspoding to mass fuel flow.
    :return:
    """
    if len(var1[var1.str.contains('iso')]) > 0:
        df_iso = tools.sfc_iso(df, engine_no)
        ff_iso = df_iso * df[var_power]
        plt.scatter(df[var_power],
                    ff_iso, color='red', marker='.', alpha=0.1)
        sfoc_iso = model['model.ted.rig:sfc.iso@engine.main.' + engine_no]
        plt.plot(xgp, sfoc_iso * xgp, color='navy')
        plt.xlim((0, np.max(xgp) + 1))
        plt.ylim((0, np.max(sfoc_iso * xgp) + 500))
    else:
        plt.scatter(df[var_power],
                    df[var_mass], color='red', marker='.', alpha=0.1)
        sfoc = model['model.ted.rig:sfc@engine.main.' + engine_no]
        plt.plot(xgp, sfoc * xgp, color='navy')
        plt.xlim((0, np.max(xgp) + 1))
        plt.ylim((0, np.max(sfoc * xgp) + 500))


def plot_results_sfc_single(df, engine_no, model, xgp, var1, var_power,
                            var_mass):
    """
    Plot modeled SFC for 1 fuel type.
    :param df: Data frame with cleaned data.
    :param engine_no: Engine number.
    :param model: Array with modeled SFC.
    :param xgp: Array with power values corresponding to modeled SFC.
    :param var1:
    :param var_power: Variable correspoding to power.
    :param var_mass: riable correspoding to mass fuel flow.
    :return:
    """
    if len(var1[var1.str.contains('iso')]) > 0:
        df_iso = tools.sfc_iso(df, engine_no)
        plt.scatter(df[var_power],
                    df_iso, color='red', marker='.', alpha=0.1)
        sfoc_iso = model['model.ted.rig:sfc.iso@engine.main.' + engine_no]
        plt.plot(xgp, sfoc_iso, color='navy')
        plt.xlim((0, np.max(xgp) + 1))
    else:
        plt.scatter(df[var_power],
                    df[var_mass] / df[var_power], color='red', marker='.',
                    alpha=0.1)
        ff = model['model.ted.rig:sfc@engine.main.' + engine_no]
        plt.plot(xgp, ff, color='navy')
        plt.xlim((0, np.max(xgp) + 1))


def rmse_single(df, engine_no, model, xgp, var_power, var_mass):
    """
    Calculates RMSE for data with single model.
    :param df: Data frame with cleaned data.
    :param engine_no: Engine number.
    :param model: Array with modeled SFC.
    :param xgp: Array with power values corresponding to modeled SFC.
    :param var_power: Variable correspoding to power.
    :param var_mass: riable correspoding to mass fuel flow.
    :return rmse: RMSE value.
    """
    var1 = df.columns
    if len(var1[var1.str.contains('iso')]) > 0:
        df_iso = tools.sfc_iso(df, engine_no)
        sfc_data_nan = df_iso[~np.isnan(df_iso)]
        sfoc_iso = model['model.ted.rig:sfc.iso@engine.main.' + engine_no]
        sfc_1 = interp(np.sort(df_iso[var_power].dropna(), axis=0), xgp,
                       sfoc_iso[:, 0])
        mse = mean_squared_error(sfc_data_nan, sfc_1)
    else:
        sfc_data = df[var_mass].values / df[var_power].values
        ff = model['model.ted.rig:sfc@engine.main.' + engine_no]
        sfc_1 = interp(np.sort(df[var_power].dropna(), axis=0), xgp, ff)
        sfc_data_nan = sfc_data[~np.isnan(sfc_data)]
        sfc_1_inf = sfc_1[~np.isinf(sfc_1)]
        sfc_1_inf2 = sfc_1_inf[~np.isnan(sfc_1_inf)]
        sfc_data_nan2 = sfc_data_nan[~np.isnan(sfc_1_inf)]
        mse = mean_squared_error(sfc_data_nan2, sfc_1_inf2)
    rmse = np.sqrt(mse)
    return rmse


def rmse_double(df_0, df_1, engine_no, model, xgp, var_power, var_mass):
    """
    Calculates RMSE for data with two model components.
    :param df_0: Data frame with data for one fuel type.
    :param df_1: Data frame with data for one fuel type.
    :param engine_no: Engine number.
    :param model: Array with modeled SFC.
    :param xgp: Array with power values corresponding to modeled SFC.
    :param var_power: variable correspoding to power.
    :param var_mass: variable correspoding to mass fuel flow.
    :return rmse1, rmse2: RMSE values for each model.
    """
    var1 = df_0.columns
    if len(var1[var1.str.contains('iso')]) > 0:
        df_0_iso = tools.sfc_iso(df_0, engine_no)
        df_1_iso = tools.sfc_iso(df_1, engine_no)
        sfoc1_iso = model['model.ted.rig:sfc1.iso@engine.main.' + engine_no]
        sfoc2_iso = model['model.ted.rig:sfc2.iso@engine.main.' + engine_no]
        sfc_1 = interp(np.sort(df_0_iso[var_power].dropna(), axis=0), xgp,
                       sfoc1_iso[:, 0])
        sfc_2 = interp(np.sort(df_1_iso[var_power].dropna(), axis=0), xgp,
                       sfoc2_iso[:, 0])
        df_nan_1 = df_0_iso[~np.isnan(df_0_iso)]
        df_nan_2 = df_1_iso[~np.isnan(df_1_iso)]
        mse1 = mean_squared_error(df_nan_1, sfc_1)
        mse2 = mean_squared_error(df_nan_2, sfc_2)
    else:
        sfc_data_1 = df_0[var_mass].values / df_0[var_power].values
        sfc_data_2 = df_1[var_mass].values / df_1[var_power].values
        ff1 = model['model.ted.rig:sfc1@engine.main.' + engine_no]
        ff2 = model['model.ted.rig:sfc2@engine.main.' + engine_no]
        sfc_1 = interp(np.sort(df_0[var_power].dropna(), axis=0), xgp, ff1)
        sfc_2 = interp(np.sort(df_1[var_power].dropna(), axis=0), xgp, ff2)
        sfc_data_nan_1 = sfc_data_1[~np.isnan(sfc_data_1)]
        sfc_data_nan_2 = sfc_data_2[~np.isnan(sfc_data_2)]
        sfc_1_inf = sfc_1[~np.isinf(sfc_1)]
        sfc_1_inf2 = sfc_1_inf[~np.isnan(sfc_1_inf)]
        sfc_2_inf = sfc_2[~np.isinf(sfc_2)]
        sfc_2_inf2 = sfc_2_inf[~np.isnan(sfc_2_inf)]
        sfc_data_nan_1_2 = sfc_data_nan_1[~np.isnan(sfc_1_inf)]
        sfc_data_nan_2_2 = sfc_data_nan_2[~np.isnan(sfc_2_inf)]
        mse1 = mean_squared_error(sfc_data_nan_1_2, sfc_1_inf2)
        mse2 = mean_squared_error(sfc_data_nan_2_2, sfc_2_inf2)
    rmse1 = np.sqrt(mse1)
    rmse2 = np.sqrt(mse2)
    return rmse1, rmse2
