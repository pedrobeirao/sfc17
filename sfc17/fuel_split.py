import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy import interp, arange, cluster
from eniflow.actions.base import Action

class FuelSplit(Action):

    name = "fuel_split"

    def __init__(self, **kwargs):
        """
        :param kwargs:
        """
        super(FuelSplit, self).__init__(**kwargs)
        self.params['scale_length'] = scale_length
	self.params['residual_threshold'] = residual_threshold
	self.params['linkage'] = linkage
	self.params['n_clusters'] = n_clusters

    def execute(self, df):

	def density_class(df):
		"""
		Labels fuel type (0 or 1) based on density. Not used for Carnival '1010-xxx' vessels.
		:param df: The data after cleaning and rebinning.
		"""
		kmeans = KMeans(n_clusters=2, random_state=0).fit(df[[2]].values)
		y_pred = kmeans.predict(df[[2]].values)
		#    y_pred = bgm.predict(sfoc_data[[2]].values)
		return y_pred

	def residual_class(df):
		"""
		Labels fuel type (0 or 1) without density or temperature.
		:param df: The data after cleaning and rebinning.
		"""
		X = df[[0]]
		y = df[[1]]
		rr = RANSACRegressor(residual_threshold = residual_threshold)
		rr.fit(X, y)
		i_mask = rr.inlier_mask_
	#	model_ransac, inlier_mask = models.rlinear_regression(X[:,0].reshape(-1, 1),y[:,0].reshape(-1, 1))
		#    print(model_ransac.estimator_.coef_[0],model_ransac.estimator_.intercept_[0])
		pred_y = rr.predict(X)
		delta_y = y[i_mask]-pred_y[i_mask]
		df_gp = make_df_gp(X[i_mask].values,y[i_mask].values,delta_y.values)
		y_pred_gp, y_std_gp = models.gp_regression(df,scale_length)
		residual_gp = df['delta_y'].values-y_pred_gp
		data = np.vstack((residual_gp/df['X'].values,df['X'].values)).T
		wdata = cluster.vq.whiten(data)	
		connectivity = kneighbors_graph(residual_gp.reshape(-1, 1)/df['X'].values.reshape(-1, 1))
		# make connectivity symmetric
		connectivity = 0.5 * (connectivity + connectivity.T)
		y_pred = AgglomerativeClustering(linkage=linkage, connectivity=connectivity,
			                   n_clusters=n_clusters).fit_predict(wdata)
		return y_pred, df_gp

#	sfoc_data = sfoc_data[variables].dropna()
	if np.mean(df[[2]].values) > 0:
#		sfoc_data = sfoc_data[sfoc_data[[2]].values > 500]
		fuel_type = density_class(df)
		df_split = df
	#    dens = sfoc_data[[2]]
	#    fuel_type = sfoc_data['fuel']
	else:
		fuel_type, df_split = residual_class(df)
	if np.median(df_split[fuel_type == 0]) > np.median(df_split[fuel_type == 1]):
		df_0 = df_split[fuel_type == 0]
		df_1 = df_split[fuel_type == 1]
	else:
		df_1 = df_split[fuel_type == 0]
		df_0 = df_split[fuel_type == 1]
	return df_0,df_1
