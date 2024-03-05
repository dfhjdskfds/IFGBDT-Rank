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

def parse_args():

    parser = OptionParser()

    # IFGBDT-Rank parameters
    parser.add_option("--epsilon", type="float", dest="epsilon")
    parser.add_option("--fair_reg", type="float", dest="fair_reg")
    parser.add_option("--seed", type="int", dest="seed")
    parser.add_option("--n_units", type="int", dest="n_units")
    parser.add_option("--l2_reg", type="float", dest="l2_reg")

    (options, args) = parser.parse_args()

    return options

def main():

    options = parse_args()
    print(options)

    epsilon = options.epsilon
    fair_reg = options.fair_reg
    seed = options.seed
    n_units = options.n_units
    l2_reg = options.l2_reg

    if n_units == 0:
        n_units = []
    else:
        n_units = [n_units]

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
    print('there are', np.mean(male_sex), 'train males')
    print('there are', np.mean(male_sex_test), 'test males')

    group_membership_train = age
    group_membership_test = age_test

    print('sens_directions', sens_directions)

    lr = .001
    epoch = 40*500
    batch_size = 10


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
    _ = train_fair_nn(X_queries, relevances, group_membership_train,  CF_X_train = CF_X_train, CF_X_test = CF_X_test, num_items_per_query = num_docs_per_query, tf_prefix='german_sensei', X_test=X_queries_test, relevance_test=relevances_test, group_membership_test = group_membership_test, n_units = n_units, lr=lr,
                      batch_size=batch_size, epoch=epoch, verbose=True, activ_f = tf.nn.relu, l2_reg=l2_reg, plot=True,
                      lamb_init=2., adv_epoch=20, adv_step=.01, epsilon=epsilon, sens_directions=sens_directions, l2_attack=.001, adv_epoch_full=20,
                      fair_reg=fair_reg, fair_start=0.1, seed=seed, simul=False, num_monte_carlo_samples = 25, bias = True, init_range = .0001,
                      entropy_regularizer = .0, baseline_ndcg = True, COUNTER_INIT = .005)

if __name__ == '__main__':
    main()
