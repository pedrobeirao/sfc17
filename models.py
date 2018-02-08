import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sfc17.tools import engine_info, variable_select, load_variables, \
    clean_data, \
    make_df_gp
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import json
import time
import os
from scipy import interp, cluster
from model_helpers.linear_models import SE
from model_helpers.simple import merge_dict
from sfc17.validation import plot_results, plot_results_sfc, rmse_double, \
    plot_results_single, plot_results_sfc_single, rmse_single
from sfc17.version import __version__
from model_helpers.helper_features import ParamsFeature
from model_helpers.base_models import AdditiveModelWithGP
from model_helpers.io import get_ship_wrapper


class SfcCalc(AdditiveModelWithGP):
    """
    Calculates linear regression and GP models.
    """
    DEFAULTS = {
        'location': 'http://rest.eniram.io/hdf5rest/',
        'ship_id': '1010-006',
        'begin': '2016-02-01',
        'end': '2016-03-01',
        'aggregation': 'first',
        'sfoc_plc': 0,
        'min_power': -0.1,
        'max_power': 10,
        'step_power': 0.1,
        'split': 'False',
        'scale_length': 4,
        'out_threshold': 140,
        't_grid': np.arange(-0.1, 10.1, 0.1),
        'obs': 'fuelflow',
        'obs_noise': .1,
        'fuel_type': 'None'
    }

    MODEL_SPEC = {'component.load':
                      {'get_column': lambda x: x[['load']].values,
                       # slope relaxed as a GP
                       'ind_column': 0,
                       'input_names': ['load'],
                       'gp':
                           {'input_grid': [np.linspace(0., 10., 100)],
                            'kernels': [SE],
                            'kernel_parameters': [[3., 1.e-04]],
                            'mean': lambda x: np.zeros((len(x), 1)),
                            'sigma': .1}},
                  'component.offset':
                      {'get_column': lambda x: np.ones((len(x), 1)),
                       'ind_column': 1,
                       'input_names': ['load'],
                       'gp': None}}

    def __init__(self, json_path=None, **kwargs):
        super(SfcCalc, self).__init__(model_spec=self.MODEL_SPEC, **kwargs)

        t = self["out_threshold"]
        l = self["scale_length"]
        min_power = self['min_power']
        max_power = self['max_power']
        xgp = np.arange(min_power, max_power + 0.1, 0.1)

        self.MODEL_SPEC['component.load']['gp']['kernel_parameters'] = \
            [[l, 1.e-04]]
        self.MODEL_SPEC['component.load']['gp']['input_grid'] = [xgp]
        self.regressor_model = None

        if json_path is not None:
            self.load(json_path)

    def rlinear_regression(self, X, y, t):
        """
        Performs robust linear regression.
        :param X: Array with X data coordinates.
        :param y: array with y data coordinates.
        :param t: Outlier detection threshold.
        :return rlr: Structure with regression model
        :return inlier_mask: Sparse array for filtering data for outliers.
        """
        self.regressor_model = RANSACRegressor(residual_threshold=t)
        self.regressor_model.fit(X, y)
        rlr = self.regressor_model
        inlier_mask = rlr.inlier_mask_
        return inlier_mask


class SplitFuelData(ParamsFeature):
    """
    Splits data into 2 different fuel types.
    """

    DEFAULTS = {
        'location': 'http://rest.eniram.io/hdf5rest/',
        'ship_id': '1010-006',
        'begin': '2016-02-01',
        'end': '2016-03-01',
        'aggregation': 'first',
        'sfoc_plc': 0,
        'min_power': -0.1,
        'max_power': 10,
        'step_power': 0.1,
        'split': 'False',
        'scale_length': 4,
        'out_threshold': 140,
        't_grid': np.arange(-0.1, 10.1, 0.1),
        'fuel_type': 'None'
    }

    def __init__(self, json_path=None, **kwargs):

        self.hfo = merge_dict({'test': 3}, kwargs)
        self.mdo = merge_dict({'test': 3}, kwargs)
        super(SplitFuelData, self).__init__(**kwargs)
        self.track_attributes('state_hfo', ('hfo',))
        self.track_attributes('state_mdo', ('mdo',))
        t = self["out_threshold"]

        self.agglo_model = AgglomerativeClustering(linkage="ward")
        self.track_attributes('agglo_model',
                              ('agglo_model', 'linkage'))
        self.regressor_model = RANSACRegressor(residual_threshold=t)
        self.track_attributes('regressor_model',
                              ('regressor_model', 'residual_threshold'))
        self.kmeans_model = KMeans(n_clusters=2)
        self.track_attributes('kmeans_model', ('kmeans_model', 'n_clusters'))
        ship = self['ship_id']
        ed, gd, epd = engine_info(ship, verbose=False)
        var_power, var_mass = variable_select(ship, ed, gd)
        self.variables = {'var_mass': var_mass, 'var_power': var_power}
        self.var_mass = var_mass
        self.var_power = var_power

        if json_path is not None:
            self.load(json_path)

    def dens_class(self, df, engine_no):
        """
        Labels fuel type (0 or 1) based on density and temperature.
        :param engine_no: Engine number.
        :param df: The data after cleaning and rebinning.
        :return y_pred: Array with predictions from clustering.
        :return df_res: Cleaned data frame with selected features.
        """
        vari = int(engine_no) - 1
        df_res = df[
            [self.var_power[vari], self.var_mass[vari],
             'density.fuel.in@engine.main.' + engine_no,
             'temperature.fuel.in@engine.main.' + engine_no]]
        df_res = df_res.dropna()
        self.kmeans_model = KMeans(n_clusters=2, random_state=0).fit(
            df_res['density.fuel.in@engine.main.' + engine_no].values.reshape(
                -1, 1))
        y_pred = self.kmeans_model.predict(
            df_res['density.fuel.in@engine.main.' + engine_no].values.reshape(
                -1, 1))
        return y_pred, df_res

    def residual_class(self, df, engine):
        """
        Labels fuel type (0 or 1) without density or temperature.
        :param df: The data after cleaning and rebinning.
        :param engine: Engine number.
        :return y_pred: Array with predictions from clustering.
        :return df_gp: Data frame with sorted data.
        """
        min_power = self['min_power']
        max_power = self['max_power']
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        l = self['scale_length']
        vari = int(engine) - 1
        t = self['out_threshold']
        df_res = df[[self.var_power[vari], self.var_mass[vari]]].dropna()
        df_sort = df_res.sort_values([self.var_power[vari]])
        x = df_res[self.var_power[vari]].values
        y = df_res[self.var_mass[vari]].values
        self.regressor_model = RANSACRegressor(residual_threshold=t)
        self.regressor_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        i_mask = self.regressor_model.inlier_mask_
        pred_y = self.regressor_model.predict(x.reshape(-1, 1))
        delta_y = y[i_mask].reshape(-1, 1) - pred_y[i_mask]
        df_gp = make_df_gp(x[i_mask], y[i_mask], delta_y,
                           self.var_power[vari],
                           self.var_mass[vari])
        df_gp.index = pd.DatetimeIndex(df_res[i_mask].index)
        data_train = pd.DataFrame()
        data_train['fuelflow'] = y[i_mask]
        data_train['load'] = x[i_mask]
        model = SfcCalc()
        model.train(data_train)
        data_test = pd.DataFrame()
        data_test['fuelflow'] = np.interp(xgp, np.sort(x[i_mask]), y[i_mask])
        data_test['load'] = xgp
        yp_grid = model.simulate_terms(data_test)
        y_gp = yp_grid['component.load'].values + \
               yp_grid['component.offset'].values
        yp = np.interp(np.sort(x[i_mask], axis=0), xgp, y_gp)
        residual_gp = data_train['fuelflow'].values - yp
        xratio = df_gp[self.var_power[vari]].values / np.std(
            df_gp[self.var_power[vari]].values)
        yratio = residual_gp / np.std(residual_gp)
        data = np.vstack((yratio, xratio)).T
        wdata = cluster.vq.whiten(data)
        connectivity = kneighbors_graph(wdata[:, 0].reshape(-1, 1),
                                        n_neighbors=10, include_self=False)
        connectivity = 0.5 * (connectivity + connectivity.T)
        self.agglo_model = AgglomerativeClustering(linkage="ward",
                                                   connectivity=connectivity,
                                                   n_clusters=2).fit_predict(
            wdata)
        y_pred = self.agglo_model
        return y_pred, df_gp

    def detect_cluster_type(self, df, engine_no):
        """
        Detects type of clustering algorithm.
        :param df: The data after cleaning and rebinning.
        :param engine_no: Engine number.
        :return cluster_type: String with cluster type
        """
        var = df.columns
        if len(var[var.str.contains(
                        'density.fuel.in@engine.main.' + engine_no)]) > 0:
            print("Using fuel density for clustering.")
            df_den = df['density.fuel.in@engine.main.' + engine_no].dropna()
            if np.mean(df_den.values) > 0:
                cluster_type = 'density_based'
            else:
                cluster_type = 'residual_based'
        else:
            print("Fuel density not available. Performing blind clustering.")
            cluster_type = 'residual_based'
        return cluster_type

    def run_split(self, df, engines, cluster_type=None):
        """
        Splits data based on fuel types.
        :param df: The data after cleaning and rebinning.
        :param engine_no: Engine number.
        :param cluster_type: String with cluster type.
        :return df_0: Data frame with data for one fuel type.
        :return df_1: Data frame with data for one fuel type.
        """
        min_power = self['min_power']
        max_power = self['max_power']
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        df0 = pd.DataFrame()
        df_split0 = pd.DataFrame()
        df_split1 = pd.DataFrame()
        for j in range(len(engines)):
            df_ft = pd.DataFrame()
            if cluster_type is None:
                cluster_type = self.detect_cluster_type(df, engines[j])
            if cluster_type == 'density_based':
                fuel_type, df_split = self.dens_class(df, engines[j])
                print(len(fuel_type), len(df_split))
            elif cluster_type == 'residual_based':
                fuel_type, df_split = self.residual_class(df, engines[j])
            else:
                raise ValueError(
                    'Invalid option: cluster_type={}'.format(cluster_type))
            vari = int(engines[j]) - 1
            sfc = df_split[self.var_mass[vari]] / df_split[self.var_power[vari]]
            if np.median(df_split[fuel_type == 0]) > np.median(
                    df_split[fuel_type == 1]):
                df_0 = df_split[fuel_type == 0]
                df_1 = df_split[fuel_type == 1]
            else:
                df_1 = df_split[fuel_type == 0]
                df_0 = df_split[fuel_type == 1]
            df_ft['fuel_type_' + engines[j]] = fuel_type
            df_ft['fuel_type_' + engines[j]] = df_ft[
                'fuel_type_' + engines[j]].replace([0, 1], ['HFO', 'MDO'])
            df_ft.index = df_split.index
            df_split0 = pd.concat([df_split0, df_split], axis=1)
            df0 = pd.concat([df0, df_ft], axis=1)

        df_split1 = pd.concat([df_split0, df0], axis=1)
        return df_split1


class SfcModel(ParamsFeature):
    """
    Class containing the SFC model.
    """
    DEFAULTS = {
        'location': 'http://rest.eniram.io/hdf5rest/',
        'ship_id': '1010-006',
        'begin': '2016-02-01',
        'end': '2016-03-01',
        'aggregation': 'first',
        'min_power': -0.1,
        'max_power': 10,
        'split': 'False',
        'fuel_type': 'None'
    }

    def __init__(self, json_path=None, hfo_model=None, mdo_model=None,
                 split_model=None, **kwargs):
        if json_path is None and hfo_model is None:
            raise ValueError('Must give either json_path to load, or submodels')

        super(SfcModel, self).__init__(**kwargs)
        self.hfo_model = hfo_model or SfcCalc()
        self.mdo_model = mdo_model or SfcCalc()
        self.split_model = split_model or SplitFuelData()
        for name in ['hfo_model', 'mdo_model', 'split_model']:
            self.track_interface(name,
                                 (name, 'get_state'),
                                 (name, 'set_state'))

        ship = self.hfo_model.params["ship_id"]
        ed, gd, epd = engine_info(ship, verbose=False)
        var_power, var_mass = variable_select(ship, ed, gd)
        self.var_power = var_power
        self.var_mass = var_mass
        self.variables = {'var_mass': var_mass, 'var_power': var_power}
        self.metadata = {'package.version': __version__}

        if json_path is not None:
            self.load(json_path)

    def load_data(self, engines):
        """
        Loads and cleans the data from the HDF5 REST database.
        :param engines: Array of engine numbers.
        :return df_all: Data frame with the cleaned data for the parameters
        defined in config.
        """
        sfoc_plc = self.hfo_model.params['sfoc_plc']
        ship = self.hfo_model.params['ship_id']
        begin_date = self.hfo_model.params['begin']
        end_date = self.hfo_model.params['end']
        ed, gd, epd = engine_info(ship, verbose=False)
        df_all = pd.DataFrame()
        print(self.hfo_model.params['ship_id'])
        var_power, var_mass = variable_select(ship, ed, gd)
        val = 'validator.aggregate:mode.ok{massFlowRate.fuel@engine.main.'
        val_sb = 'validator.aggregate:mode.ok{' \
                 'sfocPlc.1:massFlowRate.fuel@engine.main.'
        for j in range(len(engines)):
            vari = int(engines[j]) - 1
            variables = load_variables(j, engines, var_power[vari],
                                       var_mass[vari], sfoc_plc)
            df = get_ship_wrapper(
                location='http://rest.eniram.io/hdf5rest/',
                ship_id=ship, variables=variables,
                begin=begin_date, end=end_date)
            print("Loading data... engine " + engines[j])
            var = df.columns
            if len(var[var.str.contains(
                    'massFlowRate.fuel@engine.main.')]) > 0:
                if sfoc_plc == 1:
                    df_n = clean_data(df, 'sfocPlc.1:' + var_power[vari],
                                      'sfocPlc.1:' + var_mass[vari])
                else:
                    df_n = clean_data(df, var_power[vari], var_mass[vari])
                if len(var[var.str.contains(val)]) > 0:
                    df_valid = df_n[(df[val + engines[j] + '}'] == 1)]
                    if len(df_valid) > 0:
                        df_r = df_valid.resample('5min').median()
                        df_all = pd.concat([df_all, df_r], axis=1)
                    else:
                        df_r = df_n.resample('5min').median()
                        df_all = pd.concat([df_all, df_r], axis=1)
                elif len(var[var.str.contains(val_sb)]) > 0:
                    df_valid = df_n[(df[val_sb + engines[j] + '}'] == 1)]
                    if len(df_valid) > 0:
                        df_r = df_valid.resample('5min').median()
                        df_all = pd.concat([df_all, df_r], axis=1)
                    else:
                        df_r = df_n.resample('5min').median()
                        df_all = pd.concat([df_all, df_r], axis=1)
                else:
                    df_r = df_n.resample('5min').median()
                    df_all = pd.concat([df_all, df_r], axis=1)
                if len(df_n) == 0:
                    print("No data for engine " + engines[
                        j] + ".")
            elif (len(var[var.str.contains(
                    'volFlowRate.fuel@flowmeter.')]) > 0 and len(
                var[var.str.contains('density.fuel.in@engine.main.')])) > 0:
                print(
                    "No mass flow rate data for this period.")
            else:
                print("No data for this period.")
        if len(df_all) == 0:
            print("No data for this period")
        return df_all

    def plot_diagnostic(self, df, i):
        """
        Plots mass flow rate and power time series, as well as
        mass flow rate vs. power and SFC vs. power.
        :param df: Data Frame with cleaned data.
        :param i: Index correponding to engine number.
        """
        begin_date = self.hfo_model.params['begin']
        end_date = self.hfo_model.params['end']
        ship = self.hfo_model.params['ship_id']
        plt.figure(figsize=(13, 9))
        plt.subplot(221)
        ax = df[self.var_power[i]].plot()
        ax.set_xlim(pd.Timestamp(begin_date), pd.Timestamp(end_date))
        plt.ylabel("Power (MW)")
        plt.subplot(222)
        ax = df[self.var_mass[i]].plot()
        ax.set_xlim(pd.Timestamp(begin_date), pd.Timestamp(end_date))
        plt.ylabel("Mass fuel flow (kg/h)")
        # plt.legend(loc='lower right')
        plt.subplot(223)
        plt.scatter(df[self.var_power[i]], df[self.var_mass[i]], alpha=0.1)
        plt.ylabel("Mass fuel flow (kg/h)")
        plt.xlabel("Power (MW)")
        plt.subplot(224)
        plt.scatter(df[self.var_power[i]],
                    df[self.var_mass[i]] / df[self.var_power[i]], alpha=0.1)
        plt.ylabel("SFOC (MW/kg*h)")
        plt.xlabel("Power (MW)")
        plt.ylim((180, 300))
        plt.show()

    def calculate(self, x, y, engines, i, fuel_type):
        """
        Fitting the model
        :param x: Data frame with power data.
        :param y: Data frame with fuel flow data.
        :param engines: Array of engine numbers.
        :param i: Index correponding to engine number.
        :param fuel_type: Binary array indicating HFO (0) or MDO (1) fuel types.
        :return ff: Modeled fuel flow
        :return y_std_gp: Statndard deviation from GP fit.
        """
        vari = int(engines[i]) - 1
        x = x.values
        y = y.values
        ff = []
        y_std_gp = []
        if np.median(y) > 20000:
            y /= 100
        if fuel_type is 'hfo':
            min_power = self.hfo_model.params["min_power"]
            max_power = self.hfo_model.params["max_power"]
            l = self.hfo_model.params['scale_length']
            t = self.hfo_model.params['out_threshold']
            i_mask = self.hfo_model.rlinear_regression(
                x.reshape(-1, 1), y.reshape(-1, 1), t=t)
            xgp = np.arange(min_power, max_power + 0.1, 0.1)
            data_train = pd.DataFrame()
            data_train['fuelflow'] = y[i_mask]
            data_train['load'] = x[i_mask]
            self.hfo_model.train(data_train)
            data_test = pd.DataFrame()
            data_test['fuelflow'] = np.interp(xgp, x[i_mask], y[i_mask])
            data_test['load'] = xgp
            yp_grid = self.hfo_model.simulate_terms(data_test)
            y_gp = yp_grid['component.load'].values + \
                   yp_grid['component.offset'].values
        elif fuel_type is 'mdo':
            min_power = self.mdo_model.params["min_power"]
            max_power = self.mdo_model.params["max_power"]
            l = self.mdo_model.params['scale_length']
            t = self.mdo_model.params['out_threshold']
            i_mask = self.mdo_model.rlinear_regression(
                x.reshape(-1, 1), y.reshape(-1, 1), t=t)
            xgp = np.arange(min_power, max_power + 0.1, 0.1)
            data_train = pd.DataFrame()
            data_train['fuelflow'] = y[i_mask]
            data_train['load'] = x[i_mask]
            self.mdo_model.train(data_train)
            data_test = pd.DataFrame()
            data_test['fuelflow'] = np.interp(xgp, x[i_mask], y[i_mask])
            data_test['load'] = xgp
            yp_grid = self.mdo_model.simulate_terms(data_test)
            y_gp = yp_grid['component.load'].values + \
                   yp_grid['component.offset'].values
        return y_gp

    def detect_split_type(self, df):
        """
        Detects data split type if split is enabled.
        :param df: Data Frame with cleaned data.
        :return split_type: String with type of data splitting.
        """
        var = df.columns
        if len(var[var.str.contains(
                'count.sfc_notification.sfc_fuel_type.engine.main.')]) > 0:
            split_type = 'threshold_based'
        else:
            split_type = 'clustering_based'
        return split_type

    # def split_data(self, df, engines):
    #     var = 'noncanonical:count.sfc_notification.sfc_fuel_type.engine.main.'
    #     if split_type is None:
    #         split_type = self.detect_split_type(df)
    #     if split_type == 'clustering_based':
    #         df_0, df_1, df = self.split_model.run_split(df, engines,
    #                                                     cluster_type=None)
    #         print("Using clustering algorithms for fuel type split")
    #     elif split_type == 'threshold_based':
    #         print(
    #             "Using temperature and density thresholds for fuel type split")
    #         df_0 = df[df[var + engines[j]] == 1]
    #         df_1 = df[df[var + engines[j]] == 2]
    #     return df_0, df_1

    def fit_split(self, df, engines, j, fuel_type, split_type=None):
        """
        Fits the model on data for two fuel types, if split is enabled.
        :param df: Data Frame with cleaned data.
        :param engines: Array of engine numbers.
        :param j: Index number of a specific engine.
        :param split_type: Type of data splitting.
        :return model: Calculated SFC values for two fuel types.
        """
        model1 = pd.DataFrame()
        model2 = pd.DataFrame()
        model = pd.DataFrame()
        df_0 = pd.DataFrame()
        df_1 = pd.DataFrame()
        min_power = self.hfo_model.params["min_power"]
        max_power = self.hfo_model.params["max_power"]
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        vari = int(engines[j]) - 1
        df_ft = pd.DataFrame()
        df_0 = df[df['fuel_type_'+engines[j]] == 'HFO']
        df_1 = df[df['fuel_type_'+engines[j]] == 'MDO']
        df_0 = df_0[[self.var_power[vari], self.var_mass[vari]]].sort_values(
            [self.var_power[vari]])
        df_1 = df_1[[self.var_power[vari], self.var_mass[vari]]].sort_values(
            [self.var_power[vari]])
        df_0 = clean_data(df_0, self.var_power[vari], self.var_mass[vari])
        df_1 = clean_data(df_1, self.var_power[vari], self.var_mass[vari])
        mdf0 = df_0[self.var_mass[vari]].median()
        mdf1 = df_1[self.var_mass[vari]].median()
        if mdf0 < mdf1:
            df_00 = df_0
            df_11 = df_1
            df_1 = df_00
            df_0 = df_11
        var1 = df_0.columns
        iso = ['sfocPlc.1:temperature.air@engineroom.1',
               'sfocPlc.1:temperature.chargeaircoolant@engine.main.' +
               engines[j], 'sfocPlc.1:pressure.air@engineroom.1']
        if fuel_type is 'None':
            if len(var1[var1.isin(iso)].unique()) > 2:
                print("ISO correction available")
                df_0_iso = sfc_iso(df_0, engines[j])
                ff1_iso = df_0_iso * df_0[self.var_power[vari]]
                ff1 = self.calculate(df_0[self.var_power[vari]], ff1_iso,
                                     engines, j, fuel_type='hfo')
                model1['model.ted.rig:sfc1.iso@engine.main.' + engines[
                    j]] = ff1 / xgp
            else:
                ff1 = self.calculate(df_0[self.var_power[vari]],
                                     df_0[self.var_mass[vari]], engines, j,
                                     fuel_type='hfo')
                model1[
                    'model.ted.rig:sfc1@engine.main.' + engines[j]] = ff1 / xgp
            var2 = df_1.columns
            if len(var2[var2.isin(iso)].unique()) > 2:
                print("ISO correction available")
                df_1_iso = sfc_iso(df_1, engines[j])
                ff2_iso = df_1_iso * df_1[self.var_power[vari]]
                ff2 = self.calculate(df_1[self.var_power[vari]], ff2_iso,
                                     engines, j, fuel_type='mdo')
                model2['model.ted.rig:sfc2.iso@engine.main.' + engines[
                    j]] = ff2 / xgp
            else:
                ff2 = self.calculate(df_1[self.var_power[vari]],
                                     df_1[self.var_mass[vari]], engines, j,
                                     fuel_type='mdo')
                model2['model.ted.rig:sfc2@engine.main.' + engines[
                    j]] = ff2 / xgp
            model = pd.concat([model1, model2], axis=1)
        elif fuel_type is 'hfo':
            if len(var1[var1.isin(iso)].unique()) > 2:
                print("ISO correction available")
                df_0_iso = sfc_iso(df_0, engines[j])
                ff1_iso = df_0_iso * df_0[self.var_power[vari]]
                ff1 = self.calculate(df_0[self.var_power[vari]], ff1_iso,
                                     engines, j, fuel_type='hfo')
                model1['model.ted.rig:sfc1.iso@engine.main.' + engines[
                    j]] = ff1 / xgp
            else:
                ff1 = self.calculate(df_0[self.var_power[vari]],
                                     df_0[self.var_mass[vari]], engines, j,
                                     fuel_type='hfo')
                model1[
                    'model.ted.rig:sfc1@engine.main.' + engines[j]] = ff1 / xgp
            model = model1
        elif fuel_type is 'mdo':
            var2 = df_1.columns
            if len(var2[var2.isin(iso)].unique()) > 2:
                print("ISO correction available")
                df_1_iso = sfc_iso(df_1, engines[j])
                ff2_iso = df_1_iso * df_1[self.var_power[vari]]
                ff2 = self.calculate(df_1[self.var_power[vari]], ff2_iso,
                                     engines, j, fuel_type='mdo')
                model2['model.ted.rig:sfc2.iso@engine.main.' + engines[
                    j]] = ff2 / xgp
            else:
                ff2 = self.calculate(df_1[self.var_power[vari]],
                                     df_1[self.var_mass[vari]], engines, j,
                                     fuel_type='mdo')
                model2['model.ted.rig:sfc2@engine.main.' + engines[
                    j]] = ff2 / xgp
            model = model2
        return model, df

    def fit_nosplit(self, df, engines, j):
        """
        Fits the model on data for a single fuel type, if split is disabled.
        :param df: Data Frame with cleaned data.
        :param engines: Array of engine numbers.
        :param j: Index number of a specific engine.
        :return model: Calculated SFC values.
        """
        model1 = pd.DataFrame()
        min_power = self.hfo_model.params["min_power"]
        max_power = self.hfo_model.params["max_power"]
        vari = int(engines[j]) - 1
        df_1 = df.sort_values([self.var_power[vari]])
        df_1 = clean_data(df_1, self.var_power[vari], self.var_mass[vari])
        var3 = df.columns
        iso = ['sfocPlc.1:temperature.air@engineroom.1',
               'sfocPlc.1:temperature.chargeaircoolant@engine.main.' +
               engines[j],
               'sfocPlc.1:pressure.air@engineroom.1']
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        if len(var3[var3.isin(iso)].unique()) > 2:
            print("ISO correction available")
            df_iso = sfc_iso(df_1, engines[j])
            ff_iso = df_iso * df_1[self.var_power[vari]]
            ff = self.calculate(df_1[self.var_power[vari]], ff_iso,
                                engines, j, fuel_type='hfo')
            model1['model.ted.rig:sfc.iso@engine.main.' + engines[j]] = \
                ff / xgp
        else:
            ff = self.calculate(df_1[self.var_power[vari]],
                                df_1[self.var_mass[vari]], engines, j,
                                fuel_type='hfo')
            model1['model.ted.rig:sfc@engine.main.' + engines[j]] = ff / xgp
        model = model1
        return model, df

    def fit(self, df, engines):
        """
        Wrapper code which runs all the functions necessary to model
        all the data
        param df: The data after cleaning and rebinning.
        :param df: Data frame with cleaned time series data.
        :type df: pd.DataFrame
        :param engines: List with engine numbers.
        :return model: Data frame with modeled sfc values.
        :return out_file: Name of the output file(s).
        :return xgp: Array with power values.
        """
        ship = self.hfo_model.params['ship_id']
        max_power = self.hfo_model.params['max_power']
        split = self.hfo_model.params['split']
        min_power = self.hfo_model.params['min_power']
        fuel_type = self.hfo_model.params['fuel_type']
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        output = pd.DataFrame()
        var = df.columns
        # self.var_power = self.var_power[
        #     np.array([item in var for item in self.var_power])]
        # self.var_mass = self.var_mass[
        #     np.array([item in var for item in self.var_mass])]
        if split is True:
            for j in range(len(engines)):
                vari = int(engines[j]) - 1
                print("Modeling engine " + engines[j])
                model1, df = self.fit_split(df, engines, j,
                                                       fuel_type)
                output = pd.concat([output, model1], axis=1)
        else:
            for j in range(len(engines)):
                vari = int(engines[j]) - 1
                print("Modeling engine " + engines[j])
                model1, df = self.fit_nosplit(df, engines, j)
                output = pd.concat([output, model1], axis=1)
        return output, xgp, df

    def plot_model(self, df, i, engine, model):
        """
        Plots the data and models for one engine.
        :param df: The cleaned and rebinned data.
        :param i: Index of the engine number.
        :param engine: The engine number
        :param model: The SFOC model
        """
        ship = self.hfo_model.params["ship_id"]
        split = self.hfo_model.params['split']
        max_power = self.hfo_model.params['max_power']
        min_power = self.hfo_model.params['min_power']
        fuel_type = self.hfo_model.params['fuel_type']
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        var = df.columns
        vari = int(engine) - 1
        var_power = self.var_power
        var_mass = self.var_mass
        if split is True:
            df_0 = df[df['fuel_type_' + engine] == 'HFO']
            df_1 = df[df['fuel_type_' + engine] == 'MDO']
            #            df_0_sort = df_0.sort_values([var_power[vari]])
            #            df_1_sort = df_1.sort_values([var_power[vari]])
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plot_results(df_0, df_1, engine, model,
                         xgp, var, var_power[vari], var_mass[vari], fuel_type)
            plt.ylabel("Mass fuel flow (kg/h)")
            plt.xlabel("Power (MW)")
            plt.subplot(122)
            plot_results_sfc(df_0, df_1, engine,
                             model, xgp, var, var_power[vari],
                             var_mass[vari], fuel_type)
            rmse1, rmse2 = rmse_double(df_0, df_1,
                                       engine, model, xgp,
                                       var_power[vari],
                                       var_mass[vari], fuel_type)
            plt.ylim((180, 300))
            plt.ylabel("SFOC (MW/kg*h)")
            plt.xlabel("Power (MW)")
            plt.show()
            print('Root Mean Square Error 1 = ', rmse1)
            print('Root Mean Square Error 2 = ', rmse2)
        else:
            df_sort = df.sort_values([var_power[vari]])
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plot_results_single(df_sort, engine, model, xgp, var,
                                var_power[vari], var_mass[vari])
            plt.ylabel("Mass fuel flow (kg/h)")
            plt.xlabel("Power (MW)")
            plt.subplot(122)
            plot_results_sfc_single(df_sort, engine, model,
                                    xgp, var, var_power[vari],
                                    var_mass[vari])
            rmse = rmse_single(df_sort, engine, model, xgp,
                               var_power[vari], var_mass[vari])
            plt.xlim((0, np.max(xgp) + 1))
            plt.ylim((180, 300))
            plt.ylabel("SFOC (MW/kg*h)")
            plt.xlabel("Power (MW)")
            plt.show()
            print('Root Mean Square Error: ', rmse)

    def sfc_per_load(self, df, model, engine):
        """
        Selects values in a dataframe and in the SFC model for 75.0+-2.5% load.
        :param df: A dataframe with cleaned data.
        :param model: A dataframe with modeled SFC values.
        :param engine: Engine number.
        :return df_range: Dataframe with values corresponding to 75+-2.5% load.
        :return avg_model: Average SFC values for 75+-2.5% load.
        """
        max_power = self.hfo_model.params['max_power']
        min_power = self.hfo_model.params['min_power']
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        vari = int(engine) - 1
        power_max = np.max(df[self.var_power[vari]])
        df_range = df[(df[self.var_power[vari]] > 0.725 * power_max) &
                      (df[self.var_power[vari]] < 0.775 * power_max)]
        avg_model = np.median(model[(xgp > 0.725 * power_max) &
                                    (xgp < 0.775 * power_max)])
        return df_range, avg_model

    def plot_pred_split(self, df, model, engine, i, xgp, begin_test, end_test):
        """
        Plots current and predicted SFC and mass fuel flow.
        :param df: The cleaned and rebinned data.
        :param model: The modeled SFC values for two fuel types
        :param engine: The engine number.
        :param i: The engine number index.
        :param xgp: Array with power values corresponding to the modeled SFC.
        :param begin_test: Beginning time of test data
        :param end_test: End time of the test data.
        :return sd_sort: Data frame with mass fuel flow and SFC time series.
        """
        model0 = model['model.ted.rig:sfc1@engine.main.' + engine].values
        model1 = model['model.ted.rig:sfc2@engine.main.' + engine].values
        v = 'noncanonical:count.sfc_notification.sfc_fuel_type.engine.main.'
        var = df.columns
        vari = int(engine) - 1
        if len(var[var.isin([v])]) > 0:
            df_0 = df[df[v + engine] == 1]
            df_1 = df[df[v + engine] == 2]
            print(
                "Fuel type split from density and temperature thresholds")
        else:
            df = self.split_model.run_split(df, engine,
                                                    cluster_type=None)
            df_0 = df[df['fuel_type_' + engine] == 'HFO']
            df_1 = df[df['fuel_type_' + engine] == 'MDO']
            print("Fuel type split from clustering methods")

        df_range_0, avg_model_0 = self.sfc_per_load(df_0, model0, engine)
        df_range_1, avg_model_1 = self.sfc_per_load(df_1, model1, engine)
        df_0['avg_model'] = avg_model_0
        df_1['avg_model'] = avg_model_1
        sfc_range1 = df_range_0[self.var_mass[vari]] / df_range_0[
            self.var_power[vari]]
        df_range_0['sfc'] = sfc_range1.values
        sfc_range2 = df_range_1[self.var_mass[vari]] / df_range_1[
            self.var_power[vari]]
        df_range_1['sfc'] = sfc_range2.values
        sfc1 = df_0[self.var_mass[vari]] / df_0[self.var_power[vari]]
        df_0['sfc'] = sfc1.values
        sfc2 = df_1[self.var_mass[vari]] / df_1[self.var_power[vari]]
        df_1['sfc'] = sfc2.values
        if np.median(sfc1.values[~np.isnan(sfc1.values)]) \
                < np.median(sfc2.values[~np.isnan(sfc2.values)]):
            df_00 = df_1
            df_11 = df_0
            df_0 = df_00
            df_1 = df_11
        if np.median(sfc_range1.values[~np.isnan(sfc_range1.values)]) \
                < np.median(sfc_range2.values[~np.isnan(sfc_range2.values)]):
            df_00 = df_range_1
            df_11 = df_range_0
            df_range_0 = df_00
            df_range_1 = df_11

        sfc_0 = interp(df_0[self.var_power[vari]], xgp,
                       model0)
        sfc_1 = interp(df_1[self.var_power[vari]], xgp,
                       model1)

        ff_0 = sfc_0 * df_0[self.var_power[vari]]
        ff_1 = sfc_1 * df_1[self.var_power[vari]]
        sd_sort_0 = df_0
        sd_sort_1 = df_1
        sd_sort_0['ff_0'] = ff_0
        sd_sort_1['ff_1'] = ff_1
        sd_sort_0['sfc_0'] = sfc_0
        sd_sort_1['sfc_1'] = sfc_1
        sd_sort_0 = sd_sort_0.sort_index()
        sd_sort_1 = sd_sort_1.sort_index()
        sd_sort = pd.concat([sd_sort_0[['ff_0', 'sfc_0']],
                             sd_sort_1[['ff_1', 'sfc_1']]], axis=1)
        fig = plt.figure(figsize=(12, 13))
        plt.subplot(321)
        plt.legend(loc='upper right')
        plt.plot(pd.to_datetime(df_0.index, format='%d-%m-%y'),
                 df_0['avg_model'], color='red', linewidth=2)
        plt.plot(pd.to_datetime(df_range_0.index, format='%d-%m-%Y'),
                 df_range_0['sfc'], 'bo')
        fig.autofmt_xdate()
        plt.ylabel("SFOC (kg/MW*h) @ 75% load")
        plt.ylim((180, 350))
        plt.subplot(322)
        plt.plot(pd.to_datetime(df_1.index, format='%d-%m-%y'),
                 df_1['avg_model'], color='red', linewidth=2)
        plt.plot(pd.to_datetime(df_range_1.index, format='%d-%m-%Y'),
                 df_range_1['sfc'], 'bo')
        fig.autofmt_xdate()
        plt.ylabel("SFOC (kg/MW*h) @ 75% load")
        plt.ylim((180, 350))
        plt.subplot(323)
        plt.xlim([pd.to_datetime(begin_test), pd.to_datetime(end_test)])
        plt.ylabel('SFOC (kg/MW*h)')
        plt.legend(loc='upper right')
        plt.plot(pd.to_datetime(df_0.index, format='%d-%m-%y'),
                 df_0['sfc'].values, 'bo', alpha=0.1,
                 label='Measured SFC')
        plt.plot(pd.to_datetime(sd_sort_0.index, format='%d-%m-%y'),
                 sd_sort_0['sfc_0'].values, 'ro', alpha=0.1,
                 label='Predicted SFC')
        fig.autofmt_xdate()
        plt.ylim((180, 350))
        plt.subplot(324)
        plt.plot(pd.to_datetime(df_1.index, format='%d-%m-%y'),
                 df_1['sfc'].values, 'bo', alpha=0.1,
                 label='Measured SFC')
        plt.plot(pd.to_datetime(sd_sort_1.index, format='%d-%m-%y'),
                 sd_sort_1['sfc_1'].values, 'ro', alpha=0.1,
                 label='Predicted SFC')
        plt.xlim([pd.to_datetime(begin_test), pd.to_datetime(end_test)])
        fig.autofmt_xdate()
        plt.ylabel("SFOC (kg/MW*h)")
        plt.ylim((180, 350))
        plt.legend(loc='upper right')
        plt.subplot(325)
        ax = sd_sort_0[self.var_mass[vari]].plot()
        ax = sd_sort_0['ff_0'].plot()
        ax.set_xlim(
            [pd.to_datetime(begin_test), pd.to_datetime(end_test)])
        plt.ylabel("Mass Fuel Flow (kg/h)")
        plt.subplot(326)
        ax = sd_sort_1[self.var_mass[vari]].plot()
        ax = sd_sort_1['ff_1'].plot()
        ax.set_xlim(
            [pd.to_datetime(begin_test), pd.to_datetime(end_test)])
        plt.ylabel("Mass Fuel Flow (kg/h)")
        plt.show()
        return sd_sort

    def plot_pred_single(self, df, model, engine, xgp, begin_test, end_test):
        """
        Plots current and predicted SFC and mass fuel flow.
        :param df: The cleaned and rebinned data.
        :param model: The modeled SFC values.
        :param engine: The engine number.
        :param xgp: Array with power values corresponding to the modeled SFC.
        :param begin_test: Beginning time of test data
        :param end_test: End time of the test data.
        :return:
        """
        model0 = model['model.ted.rig:sfc@engine.main.' + engine].values
        vari = int(engine) - 1
        df_range, avg_model = self.sfc_per_load(df, model0, engine)
        df['avg_model'] = avg_model
        sfc_range = df_range[self.var_mass[vari]] / df_range[
            self.var_power[vari]]
        df_range['sfc'] = sfc_range.values
        sfc = df[self.var_mass[vari]] / df[self.var_power[vari]]
        df['sfc'] = sfc.values
        sfc_0 = np.interp(df[self.var_power[vari]], xgp, model0)
        ff_0 = sfc_0 * df[self.var_power[vari]]
        df['ff_0'] = ff_0
        df['sfc_0'] = sfc_0
        fig = plt.figure(figsize=(8, 15))
        plt.subplot(311)
        plt.plot(pd.to_datetime(df.index, format='%d-%m-%y'),
                 df['sfc'].values, 'ro', alpha=0.1,
                 label='Measured SFC')
        plt.plot(pd.to_datetime(df.index, format='%d-%m-%y'),
                 df['sfc_0'], 'bo', alpha=0.1,
                 label='Predicted SFC')
        plt.xlim([pd.to_datetime(begin_test), pd.to_datetime(end_test)])
        fig.autofmt_xdate()
        plt.ylabel("SFOC (kg/MW*h)")
        plt.ylim((180, 250))
        plt.legend(loc='upper right')
        plt.subplot(312)
        plt.legend(loc='upper right')
        plt.plot(pd.to_datetime(df.index, format='%d-%m-%y'), df['avg_model'],
                 color='blue', linewidth=2)
        plt.plot(pd.to_datetime(df_range.index, format='%d-%m-%Y'),
                 df_range['sfc'], 'ro')
        fig.autofmt_xdate()
        plt.ylabel("SFOC (kg/MW*h) @ 75% load")
        plt.subplot(313)
        ax = df[self.var_mass[vari]].plot()
        ax = df['ff_0'].plot()
        ax.set_xlim(
            [pd.to_datetime(begin_test), pd.to_datetime(end_test)])
        plt.ylabel("Mass Fuel Flow (kg/h)")
        plt.show()

    def predict(self, model, i, engine, begin_test, end_test):
        """
        Calculates mass fuel flow and SFC for a test data based on model, and
        plots a comparison between modeled SFC and
        fuel flow and measured SFC and fuel flow.
        :param i: Index corresponding to the engine number.
        :param engine: The engine number.
        :param begin_test: Beginning of time frame for test data.
        :param end_test: End of time frame for test data.
        :return sd_sort: Data frame with mass fuel flow and SFC time series.
        """
        self.hfo_model.params['begin'] = begin_test
        self.hfo_model.params['end'] = end_test
        ship = self.hfo_model.params['ship_id']
        split = self.hfo_model.params['split']
        max_power = self.hfo_model.params['max_power']
        min_power = self.hfo_model.params['min_power']
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        df = self.load_data(engine)
        sd_sort = pd.DataFrame()
        if len(df) == 0:
            print("No data for this period")
        else:
            if split is True:
                sd_sort = self.plot_pred_split(df, model, engine, i, xgp,
                                               begin_test, end_test)
            else:
                sd_sort = self.plot_pred_single(df, model, engine, xgp,
                                                begin_test, end_test)
        return sd_sort

    def save_model(self, model, engines, file=None):
        """
        Make a JSON file from the models
        :param model: Pandas Dataframe with the model.
        :param engines: List with the engine numbers.
        """
        split = self.hfo_model.params['split']
        max_power = self.hfo_model.params['max_power']
        min_power = self.hfo_model.params['min_power']
        xgp = np.arange(min_power, max_power + 0.1, 0.1).tolist()
        self.hfo_model.params["gp_input_grid"] = [xgp]
        self.hfo_model.params["gp_input_vals"] = xgp
        self.hfo_model.params["t_grid"] = xgp
        if split is True:
            model1d = []
            model2d = []
            d1 = {"model.name": "SFC_HFO", "date": time.strftime("%c"),
                  "model_params": self.hfo_model.params,
                  'model.ted.rig:power.scale@engine.main': xgp
                  }
            d2 = {"model.name": "SFC_MDO", "date": time.strftime("%c"),
                  "model_params": self.hfo_model.params,
                  'model.ted.rig:power.scale@engine.main': xgp
                  }
            for j in range(len(engines)):
                model1 = model[[j]].to_dict(orient='list')
                model1d.append(model1)
                d1["engine.main." + engines[j]] = model1d[j]

            for j in range(len(engines)):
                model2 = model[[j + len(engines)]].to_dict(orient='list')
                model2d.append(model2)
                d2["engine.main." + engines[j]] = model2d[j]

            f1 = open(os.getcwd() + '/' + file + '_hfo_model.json', 'w')
            json.dump(d1, f1)
            f2 = open(os.getcwd() + '/' + file + '_mdo_model.json', 'w')
            json.dump(d2, f2)
        else:
            model1d = []
            d1 = {"model.name": "SFC_HFO", "date": time.strftime("%c"),
                  "model_params": self.hfo_model.params,
                  'model.ted.rig:power.scale@engine.main': xgp
                  }
            for j in range(len(engines)):
                model1 = model[[j]].to_dict(orient='list')
                model1d.append(model1)
                d1["engine.main." + engines[j]] = model1

            f1 = open(os.getcwd() + '/' + file + '_model.json', 'w')
            json.dump(d1, f1)

    def save_table(self, model, engines, file=None):
        """
        Saves the SFC modeled values.
        :param model: The modeled SFC values.
        :param engines: Array with engine numbers.
        :param file: Name of the output file
        :return:
        """
        split = self.hfo_model.params['split']
        max_power = self.hfo_model.params['max_power']
        min_power = self.hfo_model.params['min_power']
        xgp = np.arange(min_power, max_power + 0.1, 0.1).tolist()
        self.hfo_model.params["gp_input_grid"] = [xgp]
        self.hfo_model.params["gp_input_vals"] = xgp
        self.hfo_model.params["t_grid"] = xgp
        xgp_df = pd.DataFrame()
        xgp_df['model.ted.rig:power.scale.sfc@engine.main'] = xgp
        xgp_dict = xgp_df.to_dict(orient='list')
        d1 = {"model.name": "SFC", "date": time.strftime("%c"),
              "model_params": self.params}
        z1 = {**d1, **xgp_dict}
        z2 = {**d1, **xgp_dict}
        if split is True:
            model1d = []
            model2d = []
            for j in range(len(engines)):
                model1 = model[[j]].to_dict(orient='list')
                model1d.append(model1)
                z1["engine.main." + engines[j]] = model1d[j]
            for j in range(len(engines)):
                model2 = model[[j + len(engines)]].to_dict(orient='list')
                model2d.append(model2)
                z2["engine.main." + engines[j]] = model2d[j]
            f1 = open(os.getcwd() + '/' + file + '_hfo_table.json', 'w')
            json.dump(z1, f1)
            f2 = open(os.getcwd() + '/' + file + '_mdo_table.json', 'w')
            json.dump(z2, f2)
        else:
            model1d = []
            for j in range(len(engines)):
                model1 = model[[j]].to_dict(orient='list')
                model1d.append(model1)
                z1["engine.main." + engines[j]] = model1
            f1 = open(os.getcwd() + '/' + file + '_table.json', 'w')
            json.dump(z1, f1)

    def load_model(self, file):
        """
        Reads JSON file with models and model parameters.
        :param file: Name of the file containing the model parameters.
        :return loaded_data: Data Frame with modeled SFC and parameters
        """
        ship = self.hfo_model.params['ship_id']
        split = self.hfo_model.params['split']
        if split is True:
            with open(file + '_hfo_model.json') as data_file:
                data1 = json.load(data_file)
            with open(file + '_mdo_model.json') as data_file:
                data2 = json.load(data_file)

            model_params = data1['model_params']
        else:
            with open(file + '_hfo_model.json') as data_file:
                data1 = json.load(data_file)

            model_params = data1['model_params']
        return model_params
