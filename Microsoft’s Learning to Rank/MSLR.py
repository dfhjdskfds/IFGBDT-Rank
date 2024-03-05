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

    # IFGBDT-Rank parameters
    parser.add_option("--l2_reg", type="float", dest="l2_reg")
    parser.add_option("--adv_epoch", type="int", dest="adv_epoch")
    parser.add_option("--adv_step", type="float", dest="adv_step")
    parser.add_option("--adv_epoch_full", type="int", dest="adv_epoch_full")
    parser.add_option("--adv_step_full", type="float", dest="adv_step_full")
    parser.add_option("--epsilon", type="float", dest="epsilon")
    parser.add_option("--fair_reg", type="float", dest="fair_reg")
    parser.add_option("--n_units", type="int", dest="n_units")

    (options, args) = parser.parse_args()

    return options

def main():

    options = parse_args()
    print(options)

    l2_reg = options.l2_reg
    adv_epoch = options.adv_epoch
    adv_step = options.adv_step
    adv_epoch_full = options.adv_epoch_full
    adv_step_full = options.adv_step_full
    epsilon = options.epsilon
    fair_reg = options.fair_reg
    n_units = options.n_units

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

    _ = train_fair_nn(X_train, rel_train, group_membership_train,
                    num_items_per_query = num_docs_per_query,
                    tf_prefix='mslr_sensei',
                    X_test=X_validate,
                    relevance_test=rel_validate,
                    group_membership_test = group_membership_validate,
                    n_units = n_units,
                    lr=lr,
                    batch_size=batch_size,
                    epoch=epoch,
                    verbose=True, activ_f = tf.nn.relu, l2_reg=l2_reg, plot=True,
                      lamb_init=2., adv_epoch=adv_epoch, adv_step=adv_step, epsilon=epsilon, sens_directions=sens_directions, l2_attack=adv_step_full, adv_epoch_full=adv_epoch_full,
                      fair_reg=fair_reg, fair_start=0.1, simul=False, num_monte_carlo_samples = num_monte_carlo_samples, bias = True, init_range = .0001,
                      entropy_regularizer = .0, baseline_ndcg = True, COUNTER_INIT = .005)
if __name__ == '__main__':
    main()
