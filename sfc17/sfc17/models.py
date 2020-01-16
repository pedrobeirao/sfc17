import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sfc17 import tools
from sfc17.validation import plot_results, plot_results_sfc, rmse_double, \
    plot_results_single, plot_results_sfc_single, rmse_single
from sfc17.version import __version__
from model_helpers.helper_features import ParamsFeature
from model_helpers.base_models import AdditiveModelWithGP
from datetime import date
from model_helpers.io import get_ship_wrapper
import copy


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
        't_grid': np.arange(-0.1, 10.1, 0.1)
        # 'gp_name': 'f(x)',
        # 'gp_ind': 0,
        # 'gp_input_grid': [np.arange(-0.1, 10.1, 0.1)],
        # 'gp_input_vals': np.arange(-0.1, 10.1, 0.1),
        # 'gp_input_names': ['x'],
        # 'gp_kernel_parameters': [[4]],
        # 'gp_sigma': 20.0,
        # 'gp_mean': 'lambda x: np.zeros((len(x), 1))'
    }

    MODEL_SPEC = {'quadratic':
                      {'get_column': lambda d: d[['a']].values ** 2,
                       'ind_column': 1,
                       'input_names': ['a'],
                       'gp': None},
                  'non-parametric':
                      {'get_column': lambda d: np.ones((len(d), 1)),
                       'ind_column': 0,
                       'input_names': ['b'],
                       'gp': {'input_grid': [np.linspace(0., 2 * np.pi, 100)],
                              'kernels': [SE],
                              'kernel_parameters': [[.5, 1.e-04]],
                              'mean': lambda d: np.zeros((len(d), 1)),
                              'sigma': .1}}}

    PARAMS = {'obs': 'y',
              'obs_noise': .01}

    def __init__(self, json_path=None, **kwargs):

#        self.inherit_defaults(SfcCalc)
        super(SfcCalc, self).__init__(model_spec=copy.deepcopy(self.MODEL_SPEC), params=self.PARAMS)
        #        self.track_attributes('variables', ('variables',))
        #        self.track_attributes('metadata', ('metadata',))
        ship = self.DEFAULTS['ship_id']
        t = self.DEFAULTS["out_threshold"]
        ed, gd, epd = tools.engine_info(ship, verbose=False)
        var_power, var_mass = tools.variable_select(ship, ed, gd)
        self.var_mass = var_mass
        self.var_power = var_power
        self.regressor_model = RANSACRegressor(residual_threshold=t)
        self.track_attributes('regressor_model',
                              ('regressor_model', 'residual_threshold'))

        if json_path is None:
            self.__build_spec_init()
        else:
            self.load(json_path)
            self.__build_spec_init()
            self.user_model = tools.decode_user_model(self.params['user_model'])

    def __build_spec_init(self):
        """Build and set the 'gp_spec' attribute as required by the parent class
        LinearModelsWithGP. Used in initialization of the object.
        :return:
        """
        self.MODEL_SPEC['var_power'] = self.var_power
        self.MODEL_SPEC['var_mass'] = self.var_mass

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
        return rlr, inlier_mask

    def gp_regression(self, df, xgp, l, engine):
        """
        Performs non-parametric, non-linear regression using Gaussian Processes.
        :param df: Data Frame containing power, fuel flow, and residuals from
        linear regression.
        :param xgp: Array with power values.
        :param l: Scale length of the covariance functions.
        :return yp_grid: Array of modeled residuals.
        :return yp_grid_var: Arrray with variance of modeled residuals.
        """
        vari = int(engine) - 1
        df_sort = df.sort_values([self.var_power[vari]])
        power = df_sort[self.var_power[vari]].values
        dy = df_sort['delta_y'].values
        data_train = pd.DataFrame()
        data_train['x'] = power
        data_train['y'] = dy
#        x_ones = np.ones(dy.shape)
        params = {'obs': 'y',
                  'obs_noise': np.std(dy)/dy}
        print(data_train)
        # gp_specs = tools.build_spec(self.params)
        self.MODEL_SPEC['quadratic']['get_column'] = \
            lambda data_train: data_train[['x']].values ** 2
        self.MODEL_SPEC['quadratic']['input_names'] = ['x']
        # self.MODEL_SPEC['non-parametric']['input_grid'] = [xgp]
        self.MODEL_SPEC['non-parametric']['input_names'] = ['y']
        # self.MODEL_SPEC['non-parametric']['kernel_parameters'] = [[l, 1.e-04]]
        # gp_specs[0]['kernels'] = [SE]
        # self.gp_model = LinearModelWithGP()
        # self.gp_model.fit(x_ones, dy, gp_spec=gp_specs, ysig=10.0)
        # gp_specs[0]['input_vals'] = xgp
#        x_pred = np.ones((len(xgp), 1))
        # yp_grid = self.gp_model.predict(x_pred, gp_spec=gp_specs)
        # yp_grid_var = self.gp_model.predict_variance(x_pred, gp_spec=gp_specs)
        self.select_training = lambda x: power
        self.select_test = lambda x: xgp
        self.train(data_train)
        data_test = pd.DataFrame()
        data_test['x'] = xgp
        data_test['y'] = np.interp(xgp, power, dy)
        yp_grid = self.simulate_terms(data_test)
        yp_grid_std = np.std(yp_grid)
        return yp_grid, yp_grid_std


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
        't_grid': np.arange(-0.1, 10.1, 0.1)
        # 'gp_name': 'f(x)',
        # 'gp_ind': 0,
        # 'gp_input_grid': [np.arange(-0.1, 10.1, 0.1)],
        # 'gp_input_vals': np.arange(-0.1, 10.1, 0.1),
        # 'gp_input_names': ['x'],
        # 'gp_kernel_parameters': [[4]],
        # 'gp_sigma': 20.0,
        # 'gp_mean': 'lambda x: np.zeros((len(x), 1))'
    }

    def __init__(self, json_path=None, **kwargs):

        self.inherit_defaults(SplitFuelData)
        self.hfo = merge_dict({'test': 3}, kwargs)
        self.mdo = merge_dict({'test': 3}, kwargs)
        super(SplitFuelData, self).__init__(**kwargs)
        #        self.track_attributes('variables', ('variables',))
        #        self.track_attributes('metadata', ('metadata',))
        self.track_attributes('state_hfo', ('hfo',))
        self.track_attributes('state_mdo', ('mdo',))
        t = self.params["out_threshold"]
        calc_name = 'calc_data'
        # setattr(self, calc_name, SfcCalc())
        # self.track_interface(calc_name,
        #                      (calc_name, 'get_state'),
        #                      (calc_name, 'set_state'))

        self.agglo_model = AgglomerativeClustering(linkage="ward")
        self.track_attributes('agglo_model',
                              ('agglo_model', 'linkage'))
        self.regressor_model = RANSACRegressor(residual_threshold=t)
        self.track_attributes('regressor_model',
                              ('regressor_model', 'residual_threshold'))
        self.kmeans_model = KMeans(n_clusters=2)
        self.track_attributes('kmeans_model', ('kmeans_model', 'n_clusters'))
        ship = self.params['ship_id']
        ed, gd, epd = tools.engine_info(ship, verbose=False)
        var_power, var_mass = tools.variable_select(ship, ed, gd)
        self.variables = {'var_mass': var_mass, 'var_power': var_power}
        self.var_mass = var_mass
        self.var_power = var_power

        if json_path is None:
            self.__build_spec_init()
        else:
            self.load(json_path)
            self.__build_spec_init()

    def __build_spec_init(self):
        """Build and set the 'gp_spec' attribute as required by the parent class
        LinearModelsWithGP. Used in initialization of the object.
        :return:
        """
        spec = tools.build_spec(self.params)
        spec[0]['var_power'] = self.var_power
        spec[0]['var_mass'] = self.var_mass
        self.spec = spec

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

    def residual_class(self, df, engine, xgp):
        """
        Labels fuel type (0 or 1) without density or temperature.
        :param df: The data after cleaning and rebinning.
        :param engine_no: Engine number.
        :param xgp: Array with power values.
        :return y_pred: Array with predictions from clustering.
        :return df_gp: Data frame with sorted data.
        """
        l = self.params['scale_length']
        vari = int(engine) - 1
        t = self.params['out_threshold']
        df_res = df[[self.var_power[vari], self.var_mass[vari]]].dropna()
        x = df_res[self.var_power[vari]].values
        y = df_res[self.var_mass[vari]].values
        self.regressor_model = RANSACRegressor(residual_threshold=t)
        self.regressor_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        i_mask = self.regressor_model.inlier_mask_
        pred_y = self.regressor_model.predict(x.reshape(-1, 1))
        delta_y = y[i_mask].reshape(-1, 1) - pred_y[i_mask]
        df_gp = tools.make_df_gp(x[i_mask], y[i_mask], delta_y,
                                 self.var_power[vari],
                                 self.var_mass[vari])
        df_gp.index = pd.DatetimeIndex(df_res[i_mask].index)
        y_pred_gp, y_std_gp = self.calc_data.gp_regression(df_gp, xgp, l,
                                                           engine)
        y_gp = interp(np.sort(x[i_mask], axis=0), xgp, y_pred_gp[:, 0])
        residual_gp = df_gp['delta_y'].values - y_gp
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

    def run_split(self, df, engine_no, i, xgp, cluster_type=None):
        """
        Splits data based on fuel types.
        :param df: The data after cleaning and rebinning.
        :param engine_no: Engine number.
        :param cluster_type: String with cluster type.
        :return df_0: Data frame with data for one fuel type.
        :return df_1: Data frame with data for one fuel type.
        """
        if cluster_type is None:
            cluster_type = self.detect_cluster_type(df, engine_no)
        if cluster_type == 'density_based':
            fuel_type, df_split = self.dens_class(df, engine_no)
        elif cluster_type == 'residual_based':
            fuel_type, df_split = self.residual_class(df, engine_no, xgp)
        else:
            raise ValueError(
                'Invalid option: cluster_type={}'.format(cluster_type))
        vari = int(engine_no) - 1
        sfc = df_split[self.var_mass[vari]] / df_split[self.var_power[vari]]
        print(np.median(sfc[fuel_type == 0]))
        print(np.median(sfc[fuel_type == 1]))
        if np.median(df_split[fuel_type == 0]) > np.median(
                df_split[fuel_type == 1]):
            df_0 = df_split[fuel_type == 0]
            df_1 = df_split[fuel_type == 1]
        else:
            df_1 = df_split[fuel_type == 0]
            df_0 = df_split[fuel_type == 1]
        return df_0, df_1


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
        'split': 'False'
    }

    def __init__(self, json_path=None, hfo_model=None, mdo_model=None,
                 split_model=None, **kwargs):
        if json_path is None and hfo_model is None:
            raise ValueError('Must give either json_path to load, or submodels')

        self.hfo_model = hfo_model or SfcCalc()
        self.mdo_model = mdo_model or SfcCalc()
        self.split_model = split_model or SplitFuelData()
        super(SfcModel, self).__init__(**kwargs)
        for name in ['hfo_model', 'mdo_model', 'split_model']:
            self.track_interface(name,
                                 (name, 'get_state'),
                                 (name, 'set_state'))

        ship = self.hfo_model.params["ship_id"]
        ed, gd, epd = tools.engine_info(ship, verbose=False)
        var_power, var_mass = tools.variable_select(ship, ed, gd)
        self.var_power = var_power
        self.var_mass = var_mass
        if json_path is None:
            self.variables = {'var_mass': var_mass, 'var_power': var_power}
            self.metadata = {'package.version': __version__}
            self.__build_spec_init()

        else:
            with open(json_path, 'r') as f:
                jsondata = json.load(f)
                self.params = jsondata

            self.__build_spec_init()

    def __build_spec_init(self):
        """Build and set the 'gp_spec' attribute as required by the parent class
        LinearModelsWithGP. Used in initialization of the object.
        :return:
        """
        spec = tools.build_spec(self.params)
        spec[0]['var_power'] = self.var_power
        spec[0]['var_mass'] = self.var_mass
        self.spec = spec

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
        ed, gd, epd = tools.engine_info(ship, verbose=False)
        df_all = pd.DataFrame()
        var_power, var_mass = tools.variable_select(ship, ed, gd)
        val = 'validator.aggregate:mode.ok{massFlowRate.fuel@engine.main.'
        val_sb = 'validator.aggregate:mode.ok{' \
                 'sfocPlc.1:massFlowRate.fuel@engine.main.'
        for j in range(len(engines)):
            vari = int(engines[j]) - 1
            variables = tools.load_variables(j, engines, var_power[vari],
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
                    df_n = tools.clean_data(df, 'sfocPlc.1:' + var_power[vari],
                                            'sfocPlc.1:' + var_mass[vari])
                else:
                    df_n = tools.clean_data(df, var_power[vari], var_mass[vari])
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
            model_ransac, inlier_mask = self.hfo_model.rlinear_regression(
                x.reshape(-1, 1), y.reshape(-1, 1), t=t)
            pred_y = model_ransac.predict(x[inlier_mask].reshape(-1, 1))
            delta_y = y[inlier_mask].reshape(-1, 1) - pred_y
            df = tools.make_df_gp(x[inlier_mask], y[inlier_mask], delta_y,
                                  self.var_power[vari], self.var_mass[vari])
            xgp = np.arange(min_power, max_power + 0.1, 0.1)
            y_pred_gp, y_std_gp = self.hfo_model.gp_regression(df, xgp, l,
                                                               engines[i])
            ff = (model_ransac.estimator_.coef_[0]) * np.transpose(xgp) + \
                 model_ransac.estimator_.intercept_[0] + y_pred_gp[:, 0]
        elif fuel_type is 'mdo':
            min_power = self.mdo_model.params["min_power"]
            max_power = self.mdo_model.params["max_power"]
            l = self.mdo_model.params['scale_length']
            t = self.mdo_model.params['out_threshold']
            model_ransac, inlier_mask = self.mdo_model.rlinear_regression(
                x.reshape(-1, 1), y.reshape(-1, 1), t=t)
            pred_y = model_ransac.predict(x[inlier_mask].reshape(-1, 1))
            delta_y = y[inlier_mask].reshape(-1, 1) - pred_y
            df = tools.make_df_gp(x[inlier_mask], y[inlier_mask], delta_y,
                                  self.var_power[vari], self.var_mass[vari])
            xgp = np.arange(min_power, max_power + 0.1, 0.1)
            y_pred_gp, y_std_gp = self.mdo_model.gp_regression(df, xgp, l,
                                                               engines[i])
            ff = (model_ransac.estimator_.coef_[0]) * np.transpose(xgp) + \
                 model_ransac.estimator_.intercept_[0] + y_pred_gp[:, 0]
        return ff, y_std_gp

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

    def fit_split(self, df, engines, j, split_type=None):
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
        df_0 = pd.DataFrame()
        df_1 = pd.DataFrame()
        min_power = self.hfo_model.params["min_power"]
        max_power = self.hfo_model.params["max_power"]
        ship = self.hfo_model.params["ship_id"]
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        vari = int(engines[j]) - 1
        var = 'noncanonical:count.sfc_notification.sfc_fuel_type.engine.main.'
        if split_type is None:
            split_type = self.detect_split_type(df)
        if split_type == 'clustering_based':
            df = df[[self.var_power[vari], self.var_mass[vari]]].dropna()
            df_0, df_1 = self.split_model.run_split(df, engines[j], j, xgp,
                                                    cluster_type=None)
            print("Using clustering algorithms for fuel type split")
        elif split_type == 'threshold_based':
            print(
                "Using temperature and density thresholds for fuel type split")
            df_0 = df[df[var + engines[j]] == 1]
            df_1 = df[df[var + engines[j]] == 2]
        df_0 = df_0[[self.var_power[vari], self.var_mass[vari]]].sort_values(
            [self.var_power[vari]])
        df_1 = df_1[[self.var_power[vari], self.var_mass[vari]]].sort_values(
            [self.var_power[vari]])
        df_0 = tools.clean_data(df_0, self.var_power[vari], self.var_mass[vari])
        df_1 = tools.clean_data(df_1, self.var_power[vari], self.var_mass[vari])
        var1 = df_0.columns
        iso = ['sfocPlc.1:temperature.air@engineroom.1',
               'sfocPlc.1:temperature.chargeaircoolant@engine.main.' +
               engines[j], 'sfocPlc.1:pressure.air@engineroom.1']
        if len(var1[var1.isin(iso)].unique()) > 2:
            print("ISO correction available")
            df_0_iso = tools.sfc_iso(df_0, engines[j])
            ff1_iso = df_0_iso * df_0[self.var_power[vari]]
            ff1, std_1 = self.calculate(df_0[self.var_power[vari]], ff1_iso,
                                        engines, j, fuel_type='hfo')
            model1['model.ted.rig:sfc1.iso@engine.main.' + engines[
                j]] = ff1 / xgp
        else:
            ff1, std_1 = self.calculate(df_0[self.var_power[vari]],
                                        df_0[self.var_mass[vari]], engines, j,
                                        fuel_type='hfo')
            model1['model.ted.rig:sfc1@engine.main.' + engines[j]] = ff1 / xgp
        var2 = df_1.columns
        if len(var2[var2.isin(iso)].unique()) > 2:
            print("ISO correction available")
            df_1_iso = tools.sfc_iso(df_1, engines[j])
            ff2_iso = df_1_iso * df_1[self.var_power[vari]]
            ff2, std_2 = self.calculate(df_1[self.var_power[vari]], ff2_iso,
                                        engines, j, fuel_type='mdo')
            model2['model.ted.rig:sfc2.iso@engine.main.' + engines[
                j]] = ff2 / xgp
        else:
            ff2, std_2 = self.calculate(df_1[self.var_power[vari]],
                                        df_1[self.var_mass[vari]], engines, j,
                                        fuel_type='mdo')
            model2['model.ted.rig:sfc2@engine.main.' + engines[
                j]] = ff2 / xgp
        model = pd.concat([model1, model2], axis=1)
        return model

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
        ship = self.hfo_model.params["ship_id"]
        vari = int(engines[j]) - 1
        df_1 = df.sort_values([self.var_power[vari]])
        df_1 = tools.clean_data(df_1, self.var_power[vari], self.var_mass[vari])
        var3 = df.columns
        iso = ['sfocPlc.1:temperature.air@engineroom.1',
               'sfocPlc.1:temperature.chargeaircoolant@engine.main.' +
               engines[j],
               'sfocPlc.1:pressure.air@engineroom.1']
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        if len(var3[var3.isin(iso)].unique()) > 2:
            print("ISO correction available")
            df_iso = tools.sfc_iso(df_1, engines[j])
            ff_iso = df_iso * df_1[self.var_power[vari]]
            ff, std = self.calculate(df_1[self.var_power[vari]], ff_iso,
                                     engines, j, fuel_type='hfo')
            model1['model.ted.rig:sfc.iso@engine.main.' + engines[j]] = ff / xgp
        else:
            ff, std = self.calculate(df_1[self.var_power[vari]],
                                     df_1[self.var_mass[vari]], engines, j,
                                     fuel_type='hfo')
            model1['model.ted.rig:sfc@engine.main.' + engines[j]] = ff / xgp
        model = model1
        return model

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
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        model = pd.DataFrame()
        var = df.columns
        self.var_power = self.var_power[
            np.array([item in var for item in self.var_power])]
        self.var_mass = self.var_mass[
            np.array([item in var for item in self.var_mass])]
        out_files = []
        if split is True:
            for j in range(len(engines)):
                vari = int(engines[j]) - 1
                df_nan = df[
                    [self.var_power[vari], self.var_mass[vari]]].dropna()
                print("Modeling engine " + engines[j])
                df_nan = df_nan[(df_nan[self.var_power[vari]] > min_power) & (
                    df_nan[self.var_power[vari]] < max_power)]
                model1 = self.fit_split(df_nan, engines, j)
                model = pd.concat([model, model1], axis=1)
        else:
            for j in range(len(engines)):
                vari = int(engines[j]) - 1
                df_nan = df[
                    [self.var_power[vari], self.var_mass[vari]]].dropna()
                print("Modeling engine " + engines[j])
                df_nan = df_nan[(df_nan[self.var_power[vari]] > min_power) & (
                    df_nan[self.var_power[vari]] < max_power)]
                model1 = self.fit_nosplit(df_nan, engines, j)
                model = pd.concat([model, model1], axis=1)
        return model, xgp

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
        xgp = np.arange(min_power, max_power + 0.1, 0.1)
        var = df.columns
        vari = int(engine) - 1
        var_power = self.var_power
        var_mass = self.var_mass
        if split is True:
            df = df[[var_power[vari], var_mass[vari]]].dropna()
            df_0, df_1 = self.split_model.run_split(df, engine, i, xgp,
                                                    cluster_type=None)
            df_0_sort = df_0.sort_values([var_power[vari]])
            df_1_sort = df_1.sort_values([var_power[vari]])
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plot_results(df_0_sort, df_1_sort, engine, model,
                         xgp, var, var_power[vari], var_mass[vari])
            plt.ylabel("Mass fuel flow (kg/h)")
            plt.xlabel("Power (MW)")
            plt.subplot(122)
            plot_results_sfc(df_0_sort, df_1_sort, engine,
                             model, xgp, var, var_power[vari],
                             var_mass[vari])
            rmse1, rmse2 = rmse_double(df_0_sort, df_1_sort,
                                       engine, model, xgp,
                                       var_power[vari],
                                       var_mass[vari])
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
        avg_model = np.mean(model[(xgp > 0.725 * power_max) &
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
        :return:
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
            df_0, df_1 = self.split_model.run_split(df, engine, i, xgp,
                                                    cluster_type=None)
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

        sfc_0 = interp(np.sort(df_0[self.var_power[vari]], axis=0), xgp,
                       model0)
        sfc_1 = interp(np.sort(df_1[self.var_power[vari]], axis=0), xgp,
                       model1)
        ff_0 = sfc_0 * np.sort(df_0[self.var_power[vari]], axis=0)
        ff_1 = sfc_1 * np.sort(df_1[self.var_power[vari]], axis=0)
        sd_sort_0 = df_0.sort_values([self.var_power[vari]])
        sd_sort_1 = df_1.sort_values([self.var_power[vari]])
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
                 df_0['avg_model'], color='blue', linewidth=2)
        plt.plot(pd.to_datetime(df_range_0.index, format='%d-%m-%Y'),
                 df_range_0['sfc'], 'ro')
        fig.autofmt_xdate()
        plt.ylabel("SFOC (kg/MW*h) @ 75% load")
        plt.ylim((180, 350))
        plt.subplot(322)
        plt.plot(pd.to_datetime(df_1.index, format='%d-%m-%y'),
                 df_1['avg_model'], color='blue', linewidth=2)
        plt.plot(pd.to_datetime(df_range_1.index, format='%d-%m-%Y'),
                 df_range_1['sfc'], 'ro')
        fig.autofmt_xdate()
        plt.ylabel("SFOC (kg/MW*h) @ 75% load")
        plt.ylim((180, 350))
        plt.subplot(323)
        plt.xlim([pd.to_datetime(begin_test), pd.to_datetime(end_test)])
        plt.ylabel('SFOC (kg/MW*h)')
        plt.legend(loc='upper right')
        plt.plot(pd.to_datetime(df_0.index, format='%d-%m-%y'),
                 df_0['sfc'].values, 'ro', alpha=0.1,
                 label='Measured SFC')
        plt.plot(pd.to_datetime(sd_sort_0.index, format='%d-%m-%y'),
                 sd_sort_0['sfc_0'].values, 'bo', alpha=0.1,
                 label='Predicted SFC')
        fig.autofmt_xdate()
        plt.ylim((180, 240))
        plt.subplot(324)
        plt.plot(pd.to_datetime(df_1.index, format='%d-%m-%y'),
                 df_1['sfc'].values, 'ro', alpha=0.1,
                 label='Measured SFC')
        plt.plot(pd.to_datetime(sd_sort_1.index, format='%d-%m-%y'),
                 sd_sort_1['sfc_1'].values, 'bo', alpha=0.1,
                 label='Predicted SFC')
        plt.xlim([pd.to_datetime(begin_test), pd.to_datetime(end_test)])
        fig.autofmt_xdate()
        plt.ylabel("SFOC (kg/MW*h)")
        plt.ylim((180, 240))
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
        sfc_0 = interp(np.sort(df[self.var_power[vari]], axis=0), xgp,
                       model0)
        ff_0 = sfc_0 * np.sort(df[self.var_power[vari]], axis=0)
        sd_sort = df.sort_values([self.var_power[vari]])
        sd_sort['ff'] = ff_0
        sd_sort['sfc'] = sfc_0
        sd_sort = sd_sort.sort_index()
        sd_sort = pd.concat([sd_sort[['ff', 'sfc']]], axis=1)
        fig = plt.figure(figsize=(8, 15))
        plt.subplot(311)
        plt.plot(pd.to_datetime(df.index, format='%d-%m-%y'),
                 df['sfc'].values, 'ro', alpha=0.1,
                 label='Measured SFC')
        plt.plot(pd.to_datetime(sd_sort.index, format='%d-%m-%y'),
                 sd_sort['sfc'], 'bo', alpha=0.1,
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
        ax = sd_sort[self.var_mass[vari]].plot()
        ax = sd_sort['ff'].plot()
        ax.set_xlim(
            [pd.to_datetime(begin_test), pd.to_datetime(end_test)])
        plt.ylabel("Mass Fuel Flow (kg/h)")
        plt.show()
        return sd_sort

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

    def save(self, model, engines, file=None):
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

    def read(self, file):
        """
        Reads JSON file with models and model parameters.
        :param file: Name of the file containing the model parameters.
        :return loaded_data: Data Frame with modeled SFC and parameters
        """
        ship = self.hfo_model.params['ship_id']
        split = self.hfo_model.params['split']
        if split is True:
            with open(file + '_hfo.json') as data_file:
                data1 = json.load(data_file)
            with open(file + '_mdo.json') as data_file:
                data2 = json.load(data_file)

            model_params = data1['model_params']
        else:
            with open(file + '_hfo.json') as data_file:
                data1 = json.load(data_file)

            model_params = data1['model_params']
        return model_params
