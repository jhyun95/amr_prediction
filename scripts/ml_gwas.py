#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 02 10:05:12 2019

@author: jhyun95
"""

import numpy as np
import scipy.stats
import pandas as pd
import sklearn.metrics, sklearn.svm, sklearn.base, sklearn.ensemble
import joblib

def train_ensemble(allele_table_path, amr_path, antibiotic, gene_path=None,
    core_cutoff=10, bootstrap_instances=0.8, bootstrap_features=0.5, num_models=500, 
    log_iter=50):
    ''' 
    Trains a random subspace ensemble of SVMs to predict AMR from the allele 
    and/or gene content of a strain for a given antibiotic.

    Parameters
    ----------
    allele_table_path : str
        Path to binary table containing allelic content of strains as CSV, where
        rows = alleles, columns = strains. Will interpret blanks as 0s.
    amr_path : str
        Path to table containing AMR profiles of strains as CSV, where rows = strains, 
        columns = drugs. Will convert "Resistant" and "Intermediate" to 1s, 
        "Susceptible" to 0s, and ignore blanks.
    antibiotic : str
        Drug to train model for, must be in columns of amr_path
    gene_path : str
        Similar to allele_table_path, but for gene content. If provided, will
        select model non-core alleles as defined by core_cutoff at the gene level,
        rather than each allele individually (default None)
    bootstrap_instances : float
        Fraction of total strains sampled for training, must be <1.0 (default 0.8)
    bootstrap_features : float
        Fraction of total genetic features sampled for training, uses all if 1.0 (default 0.5)
    num_models : int
        Number of individual models to train for ensemble (default 500)
    log_iter : int
        Print a message after this many models have been trained (default 50)

    Returns
    -------
    ensemble : RSE
        Trained RSE object
    df_features : DataFrame
        DataFrame of binary matrix encoding genetic features of each strain
    df_amr : DataFrame
        DataFrame of binary vector encoding AMR phenotypes of each strain
    '''

    ''' Reduce data to strains with AMR data for selected antibiotic '''
    df_features, df_amr = __prepare_amr_data__(allele_table_path, amr_path, 
        antibiotic, gene_path=gene_path, core_cutoff=core_cutoff)

    ''' Train and return ensemble '''
    ensemble = RSE(num_models=num_models, bootstrap_instances=bootstrap_instances,
        bootstrap_features=bootstrap_features)
    ensemble.fit(df_features.values, df_amr.values)
    return ensemble, df_features, df_amr


class RSE:
    ''' Generic Random Subspace Ensemble supporting most sklearn classifiers.
        Has basic functions such as fitting, predicting, extracting coefficients,
        tracking sampled features/samples, saving, and loading ''' 

    LOG_ITER = 50

    def __init__(self, base_model=None, weight_function=None, num_models=100, 
        bootstrap_instances=0.8, bootstrap_features=0.5):
        '''     
        Parameters
        ----------
        base_model : classifier
            An sklearn classifier to form the base unit of of the ensemble.
            If None, uses an sklearn.svm.LinearSVC with L1 penalty, squared hinge 
            loss, and balanced class weight (default None)
        weight_function : function
            Function to extract a vector of weights for each feature from a
            trained base_model. If using the default base_model, uses the coef_ 
            attribute of the trained model (default None)
        num_models : int
            Number of individual models to train for ensemble (default 100)
        bootstrap_instances : float
            Fraction of total data points sampled for training per model (default 0.8)
        bootstrap_features : float
            Fraction of total features sampled for training per model (default 0.5)
        '''   
        self.base_model = base_model
        if base_model is None:
            self.base_model = sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', 
                dual=False, class_weight='balanced')

        self.weight_function = weight_function
        if weight_function is None:
            def __weight_fxn__(clf):
                coefs = None
                if hasattr(clf, 'coef_'):
                    coefs = clf.coef_
                elif hasattr(clf, 'feature_importances_'):
                    coefs = clf.feature_importances_
                if not coefs is None:
                    return np.reshape(coefs, newshape=np.max(coefs.shape))
            self.weight_function = __weight_fxn__

        self.bootstrap_instances = bootstrap_instances
        self.bootstrap_features = bootstrap_features
        self.models = [None] * num_models
        self.selected_instances = None
        self.selected_features = None


    def fit(self, X, y):
        ''' 
        Fits the models in the ensemble 

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-list, shape = (n_samples)
            Target vector
        '''
        num_models = len(self.models)
        n_instances, n_features = X.shape
        instance_limit = int(float(n_instances) * self.bootstrap_instances)
        feature_limit = int(float(n_features) * self.bootstrap_features)
        self.selected_instances =  np.zeros((instance_limit, num_models), dtype='int32') # integer
        self.selected_features = np.zeros((n_features, num_models), dtype='int8') # binary
        
        for i in range(num_models):
            model_name = 'Model_' + str(i)
            if (i+1) % self.LOG_ITER == 0:
                print 'Iteration', i+1

            ''' Instance sampling: Re-sampling with replacement, aware of imbalanced data '''    
            y_train = [0] # dummy initialization
            while np.sum(y_train) == 0 or np.sum(y_train) == len(y_train): # shuffle until both label types in group
                all_instances = np.arange(n_instances)
                insample = np.random.choice(all_instances, instance_limit)
                X_train = X[insample,:]
                y_train = y[insample] 
                #outsample = np.delete(all_instances, insample)
                #X_test = X[outsample,:]
                #y_test = X[outsample] 

            ''' Feature sampling: Subsampling w/o replacement, to simplify feature weight averaging '''
            if self.bootstrap_features < 1.0: # random subspace, sub-select features each time
                features = range(n_features)
                np.random.shuffle(features)
                infeatures = sorted(list(features[:feature_limit]))
                X_train = X_train[:,infeatures]
                #if bootstrap_instances < 1.0: # alter the out-of-box test set
                #    X_test = X_test[:,selected_features]

            ''' Train and store classifier '''
            clf = sklearn.base.clone(self.base_model)
            clf.fit(X_train, y_train)
            self.models[i] = clf
            self.selected_instances[:,i] = insample
            self.selected_features[infeatures,i] = 1


    def get_coefficient_matrix(self, feature_names=None, reduced=True, order=0):
        ''' 
        Extract feature weights averaged across the ensemble. For features a 
        model did not have access to, the coefficient returned is NaN. 

        Parameters
        ----------
        feature_names : list, shape = (n_features)
            List of feature names, corresponding to those in X when fitting the ensemble. 
            If None, will name features [Feature_1, Feature_2, ...] (default None)
        reduced : boolean
            If True, discards rows/features that contain only 0 or NaN (default True)
        order : int
            If positive, sorts features s.t. most positive average weights are first.
            If negative, sorts features s.t. most negative average weights are first.
            If 0, does not sort features (default 0)

        Returns
        -------
        df_coefs : DataFrame, shape = (n_features, n_models)
            DataFrame of trained ensemble coefficients. For featuers a model did not
            have access to (i.e. if bootstrap_features < 1.0), the coefficient is NaN
        '''

        num_features, num_models = self.selected_features.shape
        coefs = np.zeros((num_features, num_models))
        for i,model in enumerate(self.models):
            weights = self.weight_function(model)
            if self.bootstrap_features < 1.0: # if sampled features, remap to original indicies
                infeatures = np.nonzero(self.selected_features[:,i])[0]
                true_weights = np.zeros(num_features) # remap to original list
                true_weights[:] = np.nan
                for j,a in enumerate(infeatures):
                    true_weights[a] = weights[j]
                weights = true_weights 
            coefs[:,i] = weights

        ''' Export to DataFrame '''
        model_names = map(lambda x: 'Model_' + str(x), range(1,num_models+1))
        if feature_names is None:
            feature_names = map(lambda x: 'Feature_' + str(x), range(1, num_features+1))
        df_coefs = pd.DataFrame(index=feature_names, columns=model_names, data=coefs)

        ''' Optionally remove unused features and/or sort by average weight '''
        if reduced: # remove rows that are only 0s and NaNs
            df_coefs_nulls = df_coefs.replace(0,np.nan)
            df_coefs_nulls = df_coefs_nulls.dropna(axis=0, how='all')
            non_null_features = df_coefs_nulls.index
            df_coefs = df_coefs.loc[non_null_features,:]
        if order != 0: # sorting by average weight
            averages = np.nanmean(df_coefs.values, axis=1)
            df_coefs.loc[:,'average'] = averages
            df_coefs.sort_values(by='average', ascending=(order < 0), inplace=True)
            df_coefs.drop(labels='average', axis=1, inplace=True)
        return df_coefs


def __prepare_amr_data__(allele_table_path, amr_path, antibiotic, gene_path=None, core_cutoff=10):
    ''' Processes allele, AMR, and optionally, gene data into DataFrames containing
        only strains with relevant AMR data. See train_ensemble() for parameters. '''

    ''' Load genetic data '''
    df_alleles = pd.read_csv(allele_table_path, index_col=0).T
    if not gene_path is None: # split into core gene alleles / non-core genes
        df_genes = pd.read_csv(gene_path, index_col=0).T
        df_genes.columns = df_genes.columns.map(lambda x: x.replace(' ','_'))
        n_strains = df_genes.shape[0]
        df_counts = df_genes.sum(axis=0)
        df_core = df_counts[df_counts >= n_strains - core_cutoff] 
        df_noncore = df_counts[df_counts <= n_strains - core_cutoff]
        df_noncore_genes = df_genes.reindex(df_noncore.index, axis=1)
        print 'Non-core genes:', df_noncore_genes.shape
    
        alleles = df_alleles.columns
        allele_clusters = alleles.map(lambda x: x.split('_Allele')[0])
        df_allele_to_cluster = pd.DataFrame(index=alleles, data=allele_clusters, columns=['allele'])
        df_core_alleles = df_allele_to_cluster[df_allele_to_cluster['allele'].isin(df_core.index)]
        df_core_alleles = df_alleles.reindex(df_core_alleles.index, axis=1)
        print 'Core-gene alleles:', df_core_alleles.shape
    
        df_features = pd.concat([df_core_alleles, df_noncore_genes], axis=1).fillna(0)
        print 'Feature table:', df_features.shape
    else: # model all alleles
        df_features = df_alleles

    ''' Load AMR data and consolidate genetic data  '''
    df_amr = pd.read_csv(amr_path, index_col=0).loc[:,antibiotic]
    df_amr = __binarize_amr_table__(df_amr)
    df_amr = df_amr.dropna()
    strain_order = [] 
    for strain in df_features.index:
        if strain in df_amr.index:
            strain_order.append(strain)
    df_features = df_features.reindex(strain_order).fillna(0)
    df_amr = df_amr.reindex(strain_order)
    
    return df_features, df_amr


def __binarize_amr_table__(df_amr):
    ''' Replaces AMR annotations with 0 (susceptible) or 1 (resistant), or NaN (no data). '''
    df_amr = df_amr.replace('Not Defined', np.nan)
    df_amr = df_amr.replace('Not defined', np.nan)
    df_amr = df_amr.replace('Nonsusceptible', 1.0)
    df_amr = df_amr.replace('Susceptible', 0.0)
    df_amr = df_amr.replace('Intermediate', 1.0)
    df_amr = df_amr.replace('Resistant', 1.0)
    return df_amr