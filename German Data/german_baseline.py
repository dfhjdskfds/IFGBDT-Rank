import sys
sys.path.append("..")
sys.path.append("../..")
from fair_training_ranking_xgb import train_fair_nn, compl_svd_projector
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from get_german import get_data
from sklearn.metrics import f1_score
from sklearn.linear_model import RidgeCV
from optparse import OptionParser

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def get_age_direction(X_queries, X_queries_test):

    X = np.delete(X_queries, [0], axis=1)
    X_test = np.delete(X_queries_test, [0], axis=1)

    ridge = RidgeCV().fit(X, X_queries[:,0])
    print('R^2 train', ridge.score(X,X_queries[:,0]))
    print('R^2 test', ridge.score(X_test,X_queries_test[:,0]))

    # removing duplicates
    X = unique_rows(X_queries)
    X_test = unique_rows(X_queries_test)

    ridge = RidgeCV().fit(X[:, 1:], X[:,0])

    print('R^2 train', ridge.score(X[:, 1:], X[:,0]))
    print('R^2 test', ridge.score(X_test[:, 1:], X_test[:,0]))

    w = ridge.coef_
    print(w)
    direction_1 = np.zeros(X_queries.shape[1])
    direction_1[1:] = w
    direction_2 = np.eye(X_queries.shape[1])[:,0]

    sens_directions = np.zeros((2,X_queries.shape[1]))
    sens_directions[0,:] = direction_1
    sens_directions[1,:] = direction_2

    return sens_directions

def parse_args():

    parser = OptionParser()

    # GBDT_Rank parameters
    parser.add_option("--seed", type="int", dest="seed")
    parser.add_option("--n_units", type="int", dest="n_units")
    parser.add_option("--l2_reg", type="float", dest="l2_reg")

    (options, args) = parser.parse_args()

    return options

def main():

    options = parse_args()
    print(options)

    seeds = options.seed
    n_units = options.n_units
    l2_reg = options.l2_reg

    if n_units == 0:
        n_units = []
    else:
        n_units = [n_units]

    for seed in range(seeds):
        print('on seed', seed)
        sens_directions = get_data(seed = seed, gender = False)
        # num queries 500
        X_queries, relevances, male_sex, age = pd.read_pickle(r'german_train_rank_{}.pkl'.format(str(seed)))
        X_queries_test, relevances_test, male_sex_test, age_test = pd.read_pickle(r'german_test_rank_{}.pkl'.format(str(seed)))

        num_docs_per_query = 10
        print('there are', X_queries.shape[0]/num_docs_per_query, 'queries in training')
        print('there are', X_queries_test.shape[0]/num_docs_per_query, 'queries in training')

        print('Every query has at least one relevant item.')
        print('there are', np.mean([relevances[i*num_docs_per_query:(i+1)*num_docs_per_query] for i in range(int(X_queries.shape[0]/num_docs_per_query))]), 'relevant docs on avg in each training query')
        print('there are', np.mean([relevances_test[i*num_docs_per_query:(i+1)*num_docs_per_query] for i in range(int(X_queries_test.shape[0]/num_docs_per_query))]), 'relevant docs on avg in each test query')
        print('there are', np.mean(age), 'train old')
        print('there are', np.mean(age_test), 'test old')

        # sens_directions = get_gender_direction(X_queries, X_queries_test, male_sex, male_sex_test)
        group_membership_train = age
        group_membership_test = age_test

        # group_membership_train = age
        # group_membership_test = age_test

        print('sens_directions', sens_directions)

        lr = .001
        epoch = 40*500
        batch_size = 10
        # l2_reg = .0001

        CF_X_train = np.copy(X_queries)
        CF_X_train[np.where(X_queries[:,3] == 1), 3] = 0
        CF_X_train[np.where(X_queries[:,3] == 1), 4] = 1
        CF_X_train[np.where(X_queries[:,4] == 1), 4] = 0
        CF_X_train[np.where(X_queries[:,4] == 1), 3] = 1

        CF_X_test = np.copy(X_queries_test)
        CF_X_test[np.where(X_queries_test[:,3] == 1), 3] = 0
        CF_X_test[np.where(X_queries_test[:,3] == 1), 4] = 1
        CF_X_test[np.where(X_queries_test[:,4] == 1), 4] = 0
        CF_X_test[np.where(X_queries_test[:,4] == 1), 3] = 1
        
        tf.reset_default_graph()
        print('baseline without fairness')
        weights = np.array([np.random.randn(X_queries.shape[1]).reshape(-1,1), np.array([np.random.randn(1).reshape(-1,1)])])
        _  = train_fair_nn(X_queries, relevances, group_membership_train, CF_X_train = CF_X_train, CF_X_test = CF_X_test, X_test = X_queries_test, relevance_test = relevances_test, group_membership_test = group_membership_test,
                            num_items_per_query = num_docs_per_query, tf_prefix='german_baseline',
                            n_units = n_units,
                            lr=lr,
                            batch_size=batch_size,
                            sens_directions = sens_directions,
                            epoch=epoch,
                            verbose=True,
                            activ_f = tf.nn.relu,
                            l2_reg=l2_reg,
                            plot=True,
                            fair_reg=0.,
                            fair_start=1.0,
                            seed=seed, simul=False,
                            num_monte_carlo_samples = 25,
                            bias = True,
                            init_range = .0001,
                            entropy_regularizer = 0., baseline_ndcg = True)


        tf.reset_default_graph()
        if len(n_units) == 0:
            print('RANDOM WEIGHTS')
            weights = np.array([np.random.randn(X_queries.shape[1]).reshape(-1,1), np.array([np.random.randn(1).reshape(-1,1)])])
            _  = train_fair_nn(X_queries, relevances, group_membership_train, CF_X_train = CF_X_train, CF_X_test = CF_X_test, X_test = X_queries_test, relevance_test = relevances_test, group_membership_test = group_membership_test,
                                num_items_per_query = num_docs_per_query, tf_prefix='german_random',
                                n_units = n_units,
                                sens_directions = sens_directions,
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
                                num_monte_carlo_samples = 25,
                                bias = True,
                                init_range = .0001,
                              entropy_regularizer = 0., baseline_ndcg = True)

        # project
        tf.reset_default_graph()
        print('projection baseline')
        proj_compl = compl_svd_projector(sens_directions, svd=-1)

        X_queries_project = X_queries@proj_compl.T
        X_queries_test_project = X_queries_test@proj_compl.T

        _  = train_fair_nn(X_queries_project, relevances, group_membership_train, CF_X_train = CF_X_train@proj_compl.T, CF_X_test = CF_X_test@proj_compl.T, X_test = X_queries_test_project, relevance_test = relevances_test, group_membership_test = group_membership_test,
                            num_items_per_query = num_docs_per_query, tf_prefix='project',
                            n_units = n_units,
                            lr=lr,
                            batch_size=batch_size,
                            sens_directions = sens_directions,
                            epoch=epoch,
                            verbose=True,
                            activ_f = tf.nn.relu,
                            l2_reg=l2_reg,
                            plot=True,
                            fair_reg=0.,
                            fair_start=1.0,
                            seed=seed, simul=False,
                            num_monte_carlo_samples = 25,
                            bias = True,
                            init_range = .0001,
                          entropy_regularizer = 0., baseline_ndcg = True)

if __name__ == '__main__':
    main()
