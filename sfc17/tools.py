import pandas as pd
import numpy as np
from sfc17 import models
from sfc17.ff_decomposition import decomp_fit
import matplotlib.pyplot as plt
from model_helpers.io import get_ship_wrapper
import requests
import urllib.request
from enidata.rest_data import RestDataService


def load_configs(file_path=None):
    """
    Load model parameters.
    :param file_path: Path to model parameter file.
    :return config: Dictionary with model parameters.
    """
    hfo_model = models.SfcCalc()
    mdo_model = models.SfcCalc()
    split_model = models.SplitFuelData()
    sfc_model = models.SfcModel(json_path=file_path, hfo_model=hfo_model,
                            mdo_model=mdo_model, split_model=split_model)
    state = sfc_model.get_state()
    config = state['params']  # get the init parameters
    return config


def engine_info(ship, verbose=True):
    """
    Loads info from the engine database
    :param ship: code number of the ship.
    :param verbose: Logical variable regulating the printing of the result
    :return return engineinfo: Data Frame containing engine info
    :return generatorinfo: Data Frame containing generator info
    :return enginepoolsinfo: Data Frame containing engine pools info
    """
    response = requests.get(
        'http://svn.eniram.fi/repos/dm/databases/mdb/' + ship + '.json')
    ship_data = response.json()

#    ship_data_df = pd.DataFrame.from_dict(ship_data)
    if len([s for s in ship_data if "engines" in s]) > 0:
        engine_data = ship_data['engines']
        engine_data_df = pd.DataFrame.from_dict(engine_data, orient = 'index')
        if len([s for s in engine_data_df.columns if
                "massFlowRate.fuel" in s]) > 0:
            if verbose:
                print("Mass flow rate present in data.")
        else:
            if verbose:
                print("Mass flow rate not present in data.")
        engineinfo = engine_data_df
        if verbose:
            print(engineinfo)
    else:
        engineinfo = 0
    if len([s for s in ship_data if "generators" in s]) > 0:
        generator_data = ship_data['generators']
        generator_data_df = pd.DataFrame.from_dict(generator_data,
                                                   orient='index')
        generatorinfo = generator_data_df
        if verbose:
            print(generatorinfo)
    else:
        generatorinfo = 0
    if len([s for s in ship_data if "enginepools" in s]) > 0:
        enginepools_data = ship_data['enginepools']
        enginepools_data_df = pd.DataFrame.from_dict(enginepools_data,
                                                     orient='index')
        enginepoolsinfo = enginepools_data_df
        if verbose:
            print(enginepoolsinfo)
    else:
        enginepoolsinfo = 0
    web_info = np.unique(ship_data['url.engines'])
    if verbose:
        print("More info in ", web_info)
    return engineinfo, generatorinfo, enginepoolsinfo


def variable_select(ship, engineinfo, generatorinfo):
    """
    Selects from engine and generator info the power and mass variables for all
    engines.
    :param engineinfo: Data frame with engine info.
    :param generatorinfo: Data frame with generator info.
    :return var_power: Variable correspoding to power.
    :return var_mass: Variable correspoding to mass fuel flow.
    """
    client_test = RestDataService(
        'http://test-rest02.skiff.eniram.fi:9600/hdf5rest', interface='dv4')
    var_ship = client_test.list_variables(ship)
    var = engineinfo.columns
    var_mass = np.array([])
    if 'massFlowRate.fuel' in var:
        var_mass = engineinfo['massFlowRate.fuel'].values
    elif len([s for s in var_ship if "lng" in s]) > 0:
        var_mass = [s for s in var_ship if
                    "massFlowRate.fuel.lng@engine.main" in s]
        var_mass = var_mass[1:]
    else:
        var_mass = []
        print("No mass fuel rate variable in machine database.")
    if len(generatorinfo['variable.power'].dropna()) > 0:
        var_power = generatorinfo['variable.power'].values
    else:
        var_power = 'power@' + generatorinfo.index
    return var_power, var_mass


def clean_data(df, var_power, var_mass):
    """
    Cleans the data from NaN and filters it to include only positive values.
    :param df: Data Frame with raw data loaded in "load_data".
    :param var_power: Variable correspoding to power.
    :param var_mass: Variable correspoding to mass fuel flow.
    :return df: Data frame with cleaned data.
    """
    df = df[[var_power, var_mass]].dropna()
    df = df[df[var_power] > 0]
    df = df[df[var_mass] > 0]
    #    df = df[df.columns[(~df.isin([-1])).any(axis=0)]]
    return df


def load_variables(j, engines, var_power, var_mass, sfoc_plc):
    var_non = 'noncanonical:count.sfc_notification.sfc_fuel_type.engine.main.'
    v = 'validation.aggregate:mode.ok{sfocPlc.1:massFlowRate.fuel@engine.main.'
    if sfoc_plc is True:
        vars = ['sfocPlc.1:' + var_power,
                'sfocPlc.1:' + var_mass,
                'sfocPlc.1:density.fuel.in@engine.main.' + engines[j],
                'sfocPlc.1:temperature.fuel.in@engine.main.' + engines[j],
                'validation.aggregate:mode.ok{power@generator.diesel.' +
                engines[j] + '}',
                'validation.aggregate:mode.ok{massFlowRate.fuel@engine.main.' +
                engines[j] + '}',
                var_non + engines[j], v + engines[j] + '}',
                'sfocPlc.1:temperature.chargeaircoolant@engine.main.' +
                engines[j],
                'sfocPlc.1:massFlowRate.fuel.mdoEquivalent@engine.main.' +
                engines[j],
                'sfocPlc.1:temperature.air@engineroom.1',
                'sfocPlc.1:temperature.air@engineroom.1']
    else:
        vars = [var_power, var_mass,
                'density.fuel.in@engine.main.' + engines[j],
                'temperature.fuel.in@engine.main.' + engines[j],
                'validation.aggregate:mode.ok{power@generator.diesel.' +
                engines[j] + '}',
                'validation.aggregate:mode.ok{massFlowRate.fuel@engine.main.' +
                engines[j] + '}',
                var_non + engines[j], v + engines[j] + '}',
                'massFlowRate.fuel.mdoEquivalent@engine.main.' + engines[j],
                'sfocPlc.1:temperature.chargeaircoolant@engine.main.' +
                engines[j],
                'sfocPlc.1:temperature.air@engineroom.1',
                'sfocPlc.1:temperature.air@engineroom.1']
    return vars

def plot_diagnostic(df_all, i, config):
    """
    :param df_all: Data Frame with cleaned data.
    :param i: Index correponding to engine number.
    :param config: Dictionary with the model parameters.
    """
    begin_date = config['begin']
    end_date = config['end']
    ship = config['ship_id']
    ed, gd, epd = engine_info(ship, verbose=False)
    var_power, var_mass = variable_select(ed, gd)
    plt.figure(figsize=(13, 9))
    plt.subplot(221)
    ax = df_all[var_power[i]].plot()
    ax.set_xlim(pd.Timestamp(begin_date), pd.Timestamp(end_date))
    plt.ylabel("Power (MW)")
    plt.subplot(222)
    ax = df_all[var_mass[i]].plot()
    ax.set_xlim(pd.Timestamp(begin_date), pd.Timestamp(end_date))
    plt.ylabel("Mass fuel flow (kg/h)")
    # plt.legend(loc='lower right')
    plt.subplot(223)
    plt.scatter(df_all[var_power[i]], df_all[var_mass[i]], alpha=0.1)
    plt.ylabel("Mass fuel flow (kg/h)")
    plt.xlabel("Power (MW)")
    plt.subplot(224)
    plt.scatter(df_all[var_power[i]],
                df_all[var_mass[i]] / df_all[var_power[i]], alpha=0.1)
    plt.ylabel("SFOC (MW/kg*h)")
    plt.xlabel("Power (MW)")
    plt.ylim((180, 300))
    plt.show()


def run_model(config, json_path=None):
    hfo_model = models.SfcCalc(**config)
    if config["split"] is True:
        split_model = models.SplitFuelData(**config)
        mdo_model = models.SfcCalc(**config)
    else:
        split_model = None
        mdo_model = None
    sfc_model = models.SfcModel(json_path=json_path, hfo_model=hfo_model,
                                mdo_model=mdo_model, split_model=split_model)
    return sfc_model


def decode_user_model(user_model):
    """
    :param user_model:
    :return:
    """
    out = dict()
    for term in user_model:
        out[term] = dict()
        varname = user_model[term]['varname']
        if varname is not None:
            varname_src = "'" + user_model[term]['varname'] + "'"
            func_str = user_model[term]['get_column'].replace('varname',
                                                              varname_src)
        else:
            func_str = user_model[term]['get_column']
        out[term]['get_column'] = eval(func_str)
        out[term]['varname'] = varname
    return out


def build_spec(config):
    """Currently supports one gp in a GAM.
    :param config:
    :return:
    """
    gp_dict = dict()
    for key_config in config:
        if key_config.startswith('gp_'):
            gp_dict[key_config.split('gp_')[1]] = config[key_config]
    return [gp_dict]


def make_df_gp(X, y, delta_y, var1, var2):
    """
    Prepare data for GP regression
    :param X: Array with Power data.
    :param y: array with mass fuel flow data.
    :param delta_y: array with residuals from the linear regression.
    :return df: A pandas data frame with power, fuel flow and residuals data.
    """
    df = pd.DataFrame()
    df[var1] = X
    df[var2] = y
    df['delta_y'] = delta_y
    return df


def sfc_iso(df, engine_no):
    """
    Correct the SFC curve using ISO standards
    :param df: Pandas Dataframe with the model.
    :param engine_no: Engine number.
    :return df_iso: Data frame with ISO-corrected data.
    """
    ref_temp_air_inlet = 298.0
    ref_temp_ca_cool_water = 298.0
    ref_press_ambient = 100000.0
    exponent_n = 1.2
    exponent_s = 1.0
    exponent_m = 0.7
    mech_eff = 0.9
    ref_lhv = 42.7
    site_lhv_hfo = 40.0
    #    site_lhv_mdo = 42.7
    df_iso = df[['sfocPlc.1:temperature.air@engineroom.1',
                 'sfocPlc.1:temperature.chargeaircoolant@engine.main.' +
                 engine_no,
                 'sfocPlc.1:pressure.air@engineroom.1']].dropna()
    site_temp_air_inlet = df_iso[
        'sfocPlc.1:temperature.air@engineroom.1'].values
    site_temp_ca_cool_water = df_iso[
        'sfocPlc.1:temperature.chargeaircoolant@engine.main.' +
        engine_no].values
    site_press_ambient = df_iso['sfocPlc.1:pressure.air@engineroom.1'].values
    k1 = (ref_temp_air_inlet / (
        site_temp_air_inlet[:, 0] + 273.15)) ** exponent_n
    k2 = (ref_temp_ca_cool_water /
          (site_temp_ca_cool_water + 273.15)) ** exponent_s
    k3 = ((site_press_ambient[:, 0] * 100.0) / ref_press_ambient) ** exponent_m

    k_tot = k1 * k2 * k3
    alfa = k_tot - 0.7 * (1.0 - k_tot) * (1.0 / mech_eff - 1.0)
    beta = k_tot / alfa
    sfc = df['massFlowRate.fuel@engine.main.' + engine_no].values / df[
        'power@generator.diesel.' + engine_no].values
    df_iso = (sfc * site_lhv_hfo) / (beta * ref_lhv)
    return df_iso
