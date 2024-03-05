import sys
sys.path.append("..")
sys.path.append('Fair-PGRank/')
from fair_training_ranking_xgb import train_fair_nn, compl_svd_projector
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.linear_model import RidgeCV
from optparse import OptionParser

def get_sensitive_direction(X_queries_train, X_queries_validate, X_queries_test):
    X_train = np.delete(X_queries_train, [135-4], axis=1)
    X_valid = np.delete(X_queries_validate, [135-4], axis=1)
    X_test = np.delete(X_queries_test, [135-4], axis=1)

    ridge = RidgeCV().fit(X_train, X_queries_train[:,135-4])
    print('train R^2', ridge.score(X_train, X_queries_train[:,135-4]))
    print('validate R^2', ridge.score(X_valid, X_queries_validate[:,135-4]))
    print('test R^2', ridge.score(X_test, X_queries_test[:,135-4]))

    w = ridge.coef_

    direction_1 = np.zeros(X_queries_train.shape[1])
    direction_2 = np.eye(X_queries_train.shape[1])[:,135-4]

    direction_1[0:135-4] = w[0:135-4]
    direction_1[135-4 + 1:] = w[135-4:]

    sens_directions = np.zeros((2,X_queries_train.shape[1]))
    sens_directions[0,:] = direction_1
    sens_directions[1,:] = direction_2
    return sens_directions

def parse_args():

    parser = OptionParser()

    # GBDT-Rank parameters
    parser.add_option("--n_units", type="int", dest="n_units")
    parser.add_option("--l2_reg", type="float", dest="l2_reg")

    (options, args) = parser.parse_args()

    return options

def main():

    options = parse_args()
    print(options)

    n_units = options.n_units
    l2_reg = options.l2_reg

    if n_units == 0:
        n_units = []
    else:
        n_units = [n_units]

    X_train = np.load('data/X_train.npy', allow_pickle = True)
    rel_train = np.load('data/y_train.npy', allow_pickle = True)
    group_train =  np.load('data/group_train.npy', allow_pickle = True)

    X_validate = np.load('data/X_valid.npy', allow_pickle = True)
    rel_validate = np.load('data/y_valid.npy', allow_pickle = True)
    group_validate =  np.load('data/group_valid.npy', allow_pickle = True)

    X_test = np.load('data/X_test.npy', allow_pickle = True)
    rel_test = np.load('data/y_test.npy', allow_pickle = True)
    group_test =  np.load('data/group_test.npy', allow_pickle = True)

    sens_directions = get_sensitive_direction(np.copy(X_train), np.copy(X_validate), np.copy(X_test))

    num_docs_per_query = 20
    print('there are', X_train.shape[0]/num_docs_per_query, 'queries in training')
    print('there are', X_validate.shape[0]/num_docs_per_query, 'queries in validation')
    print('there are', X_test.shape[0]/num_docs_per_query, 'queries in testing')

    print('Every query has at least one relevant item.')
    print('there are', np.mean([np.sum(rel_train[i*num_docs_per_query:(i+1)*num_docs_per_query]) for i in range(int(X_train.shape[0]/num_docs_per_query))]), 'relevant docs on avg in each training query')
    print('there are', np.mean([np.sum(rel_validate[i*num_docs_per_query:(i+1)*num_docs_per_query]) for i in range(int(X_validate.shape[0]/num_docs_per_query))]), 'relevant docs on avg in each validation query')
    print('there are', np.mean([np.sum(rel_test[i*num_docs_per_query:(i+1)*num_docs_per_query]) for i in range(int(X_test.shape[0]/num_docs_per_query))]), 'relevant docs on avg in each test query')
    print('there are', np.mean(group_train), 'train minorities')
    print('there are', np.mean(group_validate), 'validate minorities')
    print('there are', np.mean(group_test), 'test minorities')

    # sens_directions = get_gender_direction(X_queries, X_test, male_sex, male_sex_test)
    group_membership_train = group_train
    group_membership_validate = group_validate
    group_membership_test = group_test

    lr = .001
    epoch = 40*1700
    batch_size = 10
    num_monte_carlo_samples = 32
    entropy_regularizer = 0.

    _  = train_fair_nn(X_train, rel_train, group_membership_train, X_test = X_validate, relevance_test = rel_validate, group_membership_test = group_membership_validate,
                        num_items_per_query = num_docs_per_query, sens_directions = sens_directions, tf_prefix='baseline',
                        n_units = n_units,
                        lr=lr,
                        batch_size=batch_size,
                        epoch=epoch,
                        verbose=True,
                        activ_f = tf.nn.relu,
                        l2_reg=l2_reg,
                        plot=True,
                        fair_reg=0.,
                        fair_start=1.0,
                        seed=None,
                        simul=False,
                        num_monte_carlo_samples = num_monte_carlo_samples,
                        bias = True,
                        init_range = .0001,
                        entropy_regularizer = entropy_regularizer,
                        baseline_ndcg = True)

    if len(n_units) == 0:
        tf.reset_default_graph()
        print('RANDOM WEIGHTS')
        #weights = np.array([np.random.randn(X_queries.shape[1]).reshape(-1,1)/2, np.array([np.random.randn(1).reshape(-1,1)/2])])
        weights = np.array([np.random.randn(X_train.shape[1]).reshape(-1,1), np.array([np.random.randn(1).reshape(-1,1)])])

        _  = train_fair_nn(X_train, rel_train, group_membership_train, X_test = X_validate,
                            relevance_test = rel_validate, group_membership_test = group_membership_validate,
                            num_items_per_query = num_docs_per_query, sens_directions = sens_directions, tf_prefix='random',
                            n_units = n_units,
                            lr=lr,
                            weights = weights,
                            batch_size=batch_size,
                            epoch=0,
                            verbose=True,
                            activ_f = tf.nn.relu,
                            l2_reg=l2_reg,
                            plot=True,
                            fair_reg=0.,
                            fair_start=1.0,
                            seed=None, simul=False,
                            num_monte_carlo_samples = num_monte_carlo_samples,
                            bias = True,
                            init_range = .0001,
                          entropy_regularizer = 1., baseline_ndcg = True)
    #


    # project
    tf.reset_default_graph()
    proj_compl = compl_svd_projector(sens_directions, svd=-1)

    X_train_project = X_train@proj_compl.T
    X_validate_project = X_validate@proj_compl.T

    _  = train_fair_nn(X_train_project, rel_train, group_membership_train, X_test = X_validate_project, relevance_test = rel_validate, group_membership_test = group_membership_validate,
                        num_items_per_query = num_docs_per_query, sens_directions = sens_directions, tf_prefix='project',
                        n_units = n_units,
                        lr=lr,
                        batch_size=batch_size,
                        epoch=epoch,
                        verbose=True,
                        activ_f = tf.nn.relu,
                        l2_reg=l2_reg,
                        plot=True,
                        fair_reg=0.,
                        fair_start=1.0,
                        simul=False,
                        num_monte_carlo_samples = num_monte_carlo_samples,
                        bias = True,
                        init_range = .0001,
                      entropy_regularizer = entropy_regularizer, baseline_ndcg = True)

if __name__ == '__main__':
    main()
