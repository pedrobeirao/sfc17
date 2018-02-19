"""
This demo contains the model I used at Wartsila to predict Specific Fuel Consumption.
Author:
    Pedro Beirao
Date:
    05/02/2018
"""


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