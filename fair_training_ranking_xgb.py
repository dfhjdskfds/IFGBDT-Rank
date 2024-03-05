from scipy import stats
import math
import numpy as np
from sklearn.decomposition import TruncatedSVD
from collections import OrderedDict
import sys
import os
import ot
import time
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def DCG(position_bias, relevances):
    # there is a +2 because the min value of i in DCG definition is 1
    return tf.math.reduce_sum(tf.multiply(position_bias, 2.**relevances-1), axis=1) 

def best_DCG(position_bias, relevances):
    return tf.math.reduce_sum(tf.multiply(position_bias, 2.**tf.sort(relevances, axis=1, direction='DESCENDING') - 1), axis=1)

def query_distance_or_plan(proj_compl, query_1, query_2, distance = True):
    num_items, d = query_1.shape

    a = np.ones(num_items) / num_items
    b = np.copy(a)
    query_1 = np.copy(query_1)@proj_compl.T  # 矩阵乘法
    query_2 = np.copy(query_2)@proj_compl.T

    r_1 = np.tile(np.linalg.norm(query_1, axis = 1)**2, (num_items, 1)).T  # np.tile: 把数组沿各个方向复制
    r_2 = np.tile(np.linalg.norm(query_2, axis = 1)**2, (num_items, 1))    # np.linalg.norm(), 求范数，axis=1表示按行向量处理，求多个行向量的范数
    C = r_1 - 2*query_1@query_2.T + r_2

    if distance:
        return ot.emd2(a, b, C)
    else:
        return ot.emd(a, b, C)

def get_binary_group_exposure(tf_position_bias, tf_group_membership, tf_relevances):
    majority_count = tf.reduce_sum(tf_group_membership, axis = 1)
    minority_count = tf.reduce_sum(1-tf_group_membership, axis = 1)
    majority_relevance_count = tf.reduce_sum(tf_group_membership*tf_relevances, axis = 1)
    minority_relevance_count = tf.reduce_sum((1-tf_group_membership)*tf_relevances, axis = 1)
    
    beta = minority_count*majority_count*majority_relevance_count*minority_relevance_count

    group_1_merit = tf.math.divide_no_nan(beta*tf.math.divide_no_nan(tf.reduce_sum(tf.multiply(tf_group_membership,tf_relevances), axis =1), majority_count), beta)

    group_0_merit = tf.math.divide_no_nan(beta*tf.math.divide_no_nan(tf.reduce_sum(tf.multiply((1-tf_group_membership),tf_relevances), axis = 1), minority_count), beta)

    group_1_exposure =tf.math.divide_no_nan(beta*tf.math.divide_no_nan(tf.reduce_sum(tf.multiply(tf_position_bias,tf_group_membership), axis = 1), majority_count), beta)

    group_0_exposure = tf.math.divide_no_nan(beta*tf.math.divide_no_nan(tf.reduce_sum(tf.multiply(tf_position_bias,(1-tf_group_membership)), axis = 1), minority_count), beta)

    return group_1_merit, group_0_merit, group_1_exposure, group_0_exposure

def compute_individual_exposure(position_bias, all_rankings, relevances):
    individual_exposure = []
    # print('relevances', relevances)
    # per query look at exposure over all monte carlo samples
    for i in range(len(all_rankings)):
        # print(i,'------------------')
        rankings = all_rankings[i]
        # print('rankings', rankings)
        temp_exposure = []
        num_monte_carlo_samples, num_items = rankings.shape
        merit = relevances[num_items*i:(i+1)*num_items]
        non_zero_indices = merit != 0
        if sum(non_zero_indices) == 0:
            continue
        exposure = np.zeros(num_items)
        for j in range(num_monte_carlo_samples):
            exposure[rankings[j,:]] += position_bias
        exposure = exposure / num_monte_carlo_samples

        for j in range(num_items):
            for k in range(j+1, num_items):
                if merit[j] >= merit[k]:
                    if merit[k] > 0:
                        temp_exposure.append(np.max([0, exposure[j] / merit[j] - exposure[k] / merit[k]]))
                        #print(np.max([0, exposure[j] / merit[j] - exposure[k] / merit[k]]))
                elif merit[j] < merit[k]:
                    if merit[j] > 0:
                        temp_exposure.append(np.max([0, exposure[k] / merit[k] - exposure[j] / merit[j]]))
                        #print(np.max([0, exposure[k] / merit[k] - exposure[j] / merit[j]]))
        if len(temp_exposure) >0:
            individual_exposure.append(np.mean(temp_exposure))
            #print('individual exposure empty!')
    return np.mean(individual_exposure)

def get_closest_queries(X_1, X_2, proj_compl, num_items_per_query):
    '''
    Is there anyway to do this faster other than looping?
    closest_query is the same shape as X_1 and matches element by element
    '''
    n_1, _ = X_1.shape
    n_1 = int(n_1 / num_items_per_query)
    n_2, _ = X_2.shape
    n_2 = int(n_2 / num_items_per_query)

    closest_query = np.zeros(X_1.shape)
    D = np.zeros((n_1, n_2))
    plans = []
    for i in range(n_1):
        for j in range(n_2):
            A = X_1[i*num_items_per_query:(i+1)*num_items_per_query]
            B = X_2[j*num_items_per_query:(j+1)*num_items_per_query]
            # you do not want to measure the distance of a query to itself, set it to Infinity since we are taking the argmin later
            if (A==B).all():
                D[i,j] = np.Infinity
            else:
                D[i,j] = query_distance_or_plan(proj_compl, A, B, distance = True)

    for i in range(n_1):
        idx = np.argmin(D[i,:])
        plans.append(query_distance_or_plan(proj_compl, X_1[i*num_items_per_query:(i+1)*num_items_per_query], X_2[idx*num_items_per_query:(idx+1)*num_items_per_query], distance = False))
        if (plans[-1] > 1/num_items_per_query).any():
            print('Uh oh there is an issue with', i,'in compute_pairwise_query_distances_and_plans')
        closest_query[i*num_items_per_query:(i+1)*num_items_per_query,:] = X_2[idx*num_items_per_query+np.where(plans[-1] == 1/num_items_per_query)[1], :]
    return D, closest_query

def weighted_kendall_tau(ranking_1, ranking_2, p, score_1, score_2):
    weights, displacement_vec = get_weights(ranking_1, ranking_2, p)
    num_pairs = (len(ranking_1))*(len(ranking_2)-1)/2
    agree = 0
    disagree = 0
    num_agree = 0
    num_disagree = 0
    for i in range(len(ranking_1)):
        for j in range(i+1, len(ranking_1)):
            #print(i,j, score_1[i], score_1[j], score_2[i], score_2[j], (score_1[i] - score_1[j])*(score_2[i] - score_2[j]))
            if (score_1[i] - score_1[j])*(score_2[i] - score_2[j]) >=0:
                #print('agree')
                agree += weights[i]*weights[j]
                num_agree +=1
            else:
                #print('disgaree')
                #print(i,j, weights[i]*weights[j])
                disagree += weights[i]*weights[j]
                num_disagree +=1
    return (num_agree - num_disagree) / num_pairs, disagree, displacement_vec

def displacement(ranking_1, ranking_2):
    displacement = np.zeros(len(ranking_1))
    for i in range(len(ranking_1)):
        displacement[i] = int(np.argwhere(ranking_2 == ranking_1[i])[0])
    return displacement

def get_weights(ranking_1, ranking_2, p):
    p_id = np.zeros(len(ranking_1))
    p_moved = np.zeros(len(ranking_1))
    denom = np.zeros(len(ranking_1))
    displacement_vec = displacement(ranking_1,ranking_2)
    p_id[ranking_1] = p  
    p_moved[ranking_1] = p[displacement_vec.astype(int)]  
    denom[ranking_1] = np.arange(len(ranking_1)) - displacement_vec
    p_bar = [(p_id[i] - p_moved[i] )/ denom[i] if np.abs(denom[i]) > 0 else 1 for i in range(len(p_id))]
    return p_bar, displacement_vec

def get_ranking_prob(ranking, scores):
    prob = 1
    scores = scores[ranking]

    for i in range(len(ranking)):
        prob *= scores[i] / np.sum(scores[i:])
    return prob

def compl_svd_projector(names, svd=-1):
    if svd > 0:
        tSVD = TruncatedSVD(n_components=svd)
        tSVD.fit(names)
        basis = tSVD.components_.T
        print('Singular values:')
        print(tSVD.singular_values_)
    else:
        basis = names.T

    proj = np.linalg.inv(np.matmul(basis.T, basis))  
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj_compl = np.eye(proj.shape[0]) - proj  
    # print('proj_compl', proj_compl)
    return proj_compl

def fair_dist(proj, w=0.1):
    tf_proj = tf.constant(proj, dtype=tf.float32)
    if w > 0:
        return lambda x, y: tf.math.reduce_sum(tf.square(tf.matmul(x-y, tf_proj)) + w*tf.square(tf.matmul(x-y,tf.eye(proj.shape[0]) - tf_proj)), axis=1)
    else:
        return lambda x, y: tf.math.reduce_sum(tf.square(tf.matmul(x-y, tf_proj)), axis=1)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def phi_variable(shape, name):
    init_range = np.sqrt(.5 / (shape[-1] + shape[-2]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def weight_variable(shape, name, init_range=-1):
    if init_range == -1:
        if len(shape) > 1:
            init_range = np.sqrt(.5/(shape[-1]+shape[-2]))

        else:
            init_range = np.sqrt(.5/(shape[0]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def fc_network(variables, t, layer_in, n_layers, l=0, activ_f = tf.nn.relu, units = [], bias = True):
    #this is setting up all the matrix multiplication to do a forward pass
    if l==n_layers-1:
        if bias == True:
            layer_out = tf.matmul(layer_in, variables['weight_' + str(t) + str(l)]) + variables['bias_' + str(t) + str(l)]
            A = tf.math.sigmoid(layer_out)
        else:
            layer_out = tf.matmul(layer_in, variables['weight_' + str(t) + str(l)])
            A = tf.math.sigmoid(layer_out)
        units.append(layer_out)
        return A
    else:
        if bias == True:
            layer_out = activ_f(tf.matmul(layer_in, variables['weight_' + str(t) + str(l)]) + variables['bias_' + str(t) + str(l)])
        else:
            layer_out = activ_f(tf.matmul(layer_in, variables['weight_' + str(t) + str(l)])) #+ variables['bias_' + str(l)])
        l += 1
        units.append(layer_out)
        return fc_network(variables, t, layer_out, n_layers, l=l, activ_f=activ_f, units=units, bias = bias)

def forward_fair_reg(tf_X, tf_X_utility, tf_fair_X, tf_counter_X, weights=None, n_units = None, activ_f = tf.nn.relu, bias = True, init_range = -1):
    T = 40
    tree_levels = 4
    leafs_numb = 2 ** (tree_levels - 1)
    inner_numb = leafs_numb - 1
    k = 1
    lmbd = 0.1
    alpha = 0.01
    # weights are not none when you just want to evaluate
    if weights is not None:
        n_layers = int(len(weights)/2)
        n_units = [weights[i].shape[0] for i in range(0,len(weights),2)]  # 步长为2
    else:
        n_features = int(tf_X.shape[1])
        # n_class = int(tf_y.shape[1])
        n_class = inner_numb
        n_layers = len(n_units) + 1
        n_units = [n_features] + n_units + [n_class]

    variables = OrderedDict()
    if weights is None:
        for t in range(T):
            for l in range(n_layers):
                variables['weight_' + str(t) + str(l)] = weight_variable([n_units[l],n_units[l+1]], name='weight_' + str(t) + str(l), init_range = init_range)
                if bias:
                    variables['bias_' + str(t) + str(l)] = bias_variable([n_units[l+1]], name='bias_' + str(t) + str(l))
            variables['phi_' + str(t)] = phi_variable([leafs_numb, k], name='phi' + str(t))
    else:
        weight_ind = 0
        for t in range(T):
            for l in range(n_layers):
                variables['weight_' + str(t) + str()] = tf.constant(weights[weight_ind], dtype=tf.float32)
                weight_ind += 1
                if bias:
                    variables['bias_' + str(t) + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
                    weight_ind += 1

    def node_probability(index_node, A):
        a = tf.ones_like(A)
        p = a[:,0]
#         p = tf.ones(A.shape[0], dtype=tf.float64)
        while index_node - 1 >= 0:
            father_index = int((index_node - 1) / 2)
            if (index_node - 1) % 2 == 0:
                p = p * (1.0 - A[:, father_index])
            else:
                p = p * (A[:, father_index])
            index_node = father_index
        return p

    def compute_leafs_prob_matrix(A):
        ta = list()
        i = 0
        while i < leafs_numb:
            ta.append(node_probability(leafs_numb - 1 + i, A))
            i = i + 1
        leafs_prob_matrix = tf.stack(ta, axis=0)
        return leafs_prob_matrix

    def compute_inner_prob_matrix(A):
        ta = list()
        i = 0
        while i < inner_numb:
            ta.append(node_probability(i, A))
            i = i + 1
        inner_prob_matrix = tf.stack(ta, axis=0)
        return inner_prob_matrix

    def compute_boosting_output(X, variables):
        a = tf.zeros_like(X)
        output_sum = a[:,0]
        output = []
        for t in range(T):
            A = fc_network(variables, t, X, n_layers, activ_f = activ_f, bias = bias)
            leafs_prob_matrix = compute_leafs_prob_matrix(A)
            inner_prob_matrix = compute_inner_prob_matrix(A)
            output.append(tf.matmul(tf.transpose(leafs_prob_matrix, perm=[1, 0]), variables['phi_' + str(t)]))
            output_sum = output_sum + 0.1 * tf.squeeze(output[t])
        return output_sum

    l_pred = compute_boosting_output(tf_X, variables)
    l_preds_fair = compute_boosting_output(tf_fair_X, variables)
    l_pred_utility = compute_boosting_output(tf_X_utility, variables)
    l_preds_counter = compute_boosting_output(tf_counter_X, variables)

    return variables, l_pred, l_pred_utility, l_preds_fair, l_preds_counter

    
def get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, feed_dict, sess, deterministic = False):
    if deterministic:
        scores = sess.run(l_pred, feed_dict = feed_dict).reshape(-1, num_items_per_query)
        return [np.array([np.argsort(scores[i,:])[::-1]]) for i in range(scores.shape[0])]
    else:
        scores = sess.run(l_pred, feed_dict = feed_dict).reshape(-1, num_items_per_query)
        # reparameterize to avoid overflow errors
        scores = scores - np.max(scores, axis = 1).reshape(-1, 1)
        scores = np.exp(scores)
        rankings = sess.run(tf_sample_ranking, feed_dict = {tf_scores: scores})
        #return rankings
        return  [rankings[:,i,:] for i in range(rankings.shape[1])]

def prepare_data_for_utility_objective(X, relevance, rankings_list, group_membership):
    # checked
    # number of monte carlo samples = rankings_list[0].shape[0]
    num_monte_carlo_samples = rankings_list[0].shape[0]
    num_items_per_query = rankings_list[0].shape[1]
    d = X.shape[1]
    X_reordered = np.zeros((len(rankings_list)*num_monte_carlo_samples*num_items_per_query, d))
    relevance_reordered = np.zeros((len(rankings_list)*num_monte_carlo_samples*num_items_per_query))
    group_membership_reordered = np.zeros((len(rankings_list)*num_monte_carlo_samples*num_items_per_query))
    for idx,rankings in enumerate(rankings_list):
        X_reordered[idx*num_monte_carlo_samples*num_items_per_query:(idx+1)*num_monte_carlo_samples*num_items_per_query, :]= X[rankings + idx*num_items_per_query, :].reshape(num_monte_carlo_samples*num_items_per_query,d)
        relevance_reordered[idx*num_monte_carlo_samples*num_items_per_query:(idx+1)*num_monte_carlo_samples*num_items_per_query]= relevance[rankings + idx*num_items_per_query].reshape(num_monte_carlo_samples*num_items_per_query)

        group_membership_reordered[idx*num_monte_carlo_samples*num_items_per_query:(idx+1)*num_monte_carlo_samples*num_items_per_query]= group_membership[rankings + idx*num_items_per_query].reshape(num_monte_carlo_samples*num_items_per_query)

    return X_reordered, relevance_reordered, group_membership_reordered

def get_tf_ones_mean_matrix(num_monte_carlo_samples, batch_size):
    matrix = np.zeros((num_monte_carlo_samples*batch_size, batch_size))

    for i in range(batch_size):
        for j in range(num_monte_carlo_samples):
            matrix[i*num_monte_carlo_samples+j,i] = 1
    return tf.constant(matrix, dtype = tf.float32)

def get_utility_objective(tf_group_membership, num_items_per_query, batch_size, baseline_ndcg, num_monte_carlo_samples, tf_ones_mean_matrix, tf_scores, tf_lower_ones_matrix, position_bias, tf_relevances, ranking_metric = DCG, PG = False, PG_reg = 0):
    # scores are already exponentiated
    # maybe you can vectorize this
    # we are negating it because we want to maximize the utililty, but use a minimizer
    # relevances needs to be num_monte_carlo_samples*num_queries by num_items_per_query

    tf_reversed_scores = tf.reverse(tf.reshape(tf_scores, shape=(num_monte_carlo_samples*batch_size, num_items_per_query)), axis =[1])  # tf.reverse反转张量，axis为轴，axis = [1] 为在最外层反转

    tf_reshaped_group_membership = tf.reshape(tf_group_membership, shape=(num_monte_carlo_samples*batch_size, num_items_per_query))

    dcg = tf.reshape(ranking_metric(position_bias, tf_relevances), shape = (tf_reversed_scores.shape[0],1))
    # ndcg = tf.multiply(tf.reshape(1/best_DCG(position_bias, tf_relevances),shape = (tf_exp_scores.shape[0],1)) , dcg)
    ndcg = tf.reshape(tf.math.divide_no_nan(1.,best_DCG(position_bias, tf_relevances)),shape = (tf_reversed_scores.shape[0],1))*dcg
    group_1_merit, group_0_merit, group_1_exposure, group_0_exposure = get_binary_group_exposure(position_bias, tf_reshaped_group_membership, tf_relevances)

    group_1_merit = tf.reduce_mean(tf.reshape(group_1_merit, shape = (-1, num_monte_carlo_samples)), axis = 1)
    group_0_merit = tf.reduce_mean(tf.reshape(group_0_merit, shape = (-1, num_monte_carlo_samples)), axis = 1)
    group_1_exposure = tf.reduce_mean(tf.reshape(group_1_exposure, shape = (-1, num_monte_carlo_samples)), axis = 1)
    group_0_exposure = tf.reduce_mean(tf.reshape(group_0_exposure, shape = (-1, num_monte_carlo_samples)), axis = 1)


    alpha = tf.math.sign(group_1_merit-group_0_merit) # y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.
    alpha = tf.where(group_1_merit-group_0_merit>=0, tf.ones(batch_size), alpha )

    temp = tf.math.maximum(0., alpha*(tf.math.divide_no_nan(group_1_exposure,group_1_merit)  - tf.math.divide_no_nan(group_0_exposure, group_0_merit)))
    binary_group_exposure = tf.reduce_mean(tf.math.maximum(0., alpha*(tf.math.divide_no_nan(group_1_exposure,group_1_merit)  - tf.math.divide_no_nan(group_0_exposure, group_0_merit))))

    _binary_group_exposure = tf.math.maximum(0., alpha*(tf.math.divide_no_nan(group_1_exposure,group_1_merit)  - tf.math.divide_no_nan(group_0_exposure, group_0_merit)))

    _binary_group_exposure = tf.repeat(_binary_group_exposure, repeats = num_monte_carlo_samples)


    alpha = tf.math.sign(group_1_merit-group_0_merit)

    if baseline_ndcg:
        ndcg_mean = tf.reshape(tf.reduce_mean(tf.reshape(ndcg, shape = (batch_size,num_monte_carlo_samples)), axis =1), shape = (-1, 1))
        ndcg_mean_vec = tf.linalg.matmul(tf_ones_mean_matrix, ndcg_mean)
        # return ndcg, dcg, ((1/tf.linalg.matmul(tf_exp_scores, tf_lower_ones_matrix))*tf_exp_scores)*(ndcg-ndcg_mean_vec), binary_group_exposure, [group_1_merit, group_0_merit, group_1_exposure,  group_0_exposure, temp, tf_max, tf_exp_scores]#, [group_1_merit, group_0_merit, group_1_exposure, group_0_exposure, alpha, temp]

        obj = tf.reshape(tf.reduce_sum(tf_reversed_scores - tf.math.cumulative_logsumexp(tf_reversed_scores, axis = 1), axis = 1), shape = (-1,1))*(ndcg-ndcg_mean_vec)
        if PG:
            print('REGULARIZING WITH PG WITH', PG_reg)
            obj += PG_reg*tf.reshape(tf.reduce_sum(tf_reversed_scores - tf.math.cumulative_logsumexp(tf_reversed_scores, axis = 1), axis = 1), shape = (-1,1))*(tf.reshape(_binary_group_exposure, shape = (-1,1)))
        return ndcg, dcg, obj, binary_group_exposure, [group_1_merit, group_0_merit, group_1_exposure,  group_0_exposure, tf_reversed_scores, tf.reduce_sum(tf_reversed_scores - tf.math.cumulative_logsumexp(tf_reversed_scores, axis = 1), axis = 1), (ndcg-ndcg_mean_vec), obj, tf_reversed_scores - tf.math.cumulative_logsumexp(tf_reversed_scores, axis = 1), tf.math.cumulative_logsumexp(tf_reversed_scores, axis = 1)]#, [group_1_merit, group_0_merit, group_1_exposure, group_0_exposure, alpha, temp]
    else:
        obj = tf.reduce_sum(tf_reversed_scores - tf.math.cumulative_logsumexp(tf_reversed_scores), axis = 1)*(ndcg)
        return ndcg, dcg, obj, binary_group_exposure, []

def get_adv_objective(tf_pi_list, tf_X, tf_fair_X, num_items_per_query, tf_proj_compl):
    objective = 0
    for i in range(len(tf_pi_list)):
        start_idx = i*num_items_per_query
        end_idx = (i+1)*num_items_per_query

        rA = tf.reshape(tf.math.reduce_sum(tf.matmul(tf_X[start_idx:end_idx, :], tf_proj_compl)**2, 1), [-1,1])
        rB = tf.transpose(tf.reshape(tf.math.reduce_sum(tf.matmul(tf_fair_X[start_idx:end_idx, :], tf_proj_compl)**2, 1), [-1,1]))  # tf.transpose为转置
        D = -2*tf.linalg.matmul(tf.matmul(tf_X[start_idx:end_idx, :], tf_proj_compl),tf.transpose(tf.matmul(tf_fair_X[start_idx:end_idx, :], tf_proj_compl)))
        objective += tf.multiply(tf_pi_list[i], tf.identity(rA) + tf.identity(D) + tf.identity(rB))

    return tf.reduce_sum(objective) / float(len(tf_pi_list))

def evaluate_query_distance(X, fair_X, proj_compl, batch_size, num_items_per_query):
    distance = np.zeros(batch_size)
    for i in range(batch_size):
        # get transport plan
        start_idx = i
        end_idx = i+num_items_per_query
        distance[i] = query_distance_or_plan(proj_compl, X[start_idx:end_idx, :], fair_X[start_idx:end_idx, :], distance = True)

    return distance

def get_minibatch(X, relevances, batches_of_queries, num_items_per_query, document_batch_size, k, group_membership):
    _, d = X.shape
    batch_size = batches_of_queries.shape[1]
    batch_doc_idx = [i*num_items_per_query + j for i in batches_of_queries[k] for j in range(num_items_per_query)]
    X_batch = np.zeros((document_batch_size, d))
    X_batch[range(len(batch_doc_idx)), :] = X[batch_doc_idx, :]
    batch_relevance = np.zeros((document_batch_size))
    batch_group_membership= np.zeros((document_batch_size))
    batch_relevance[range(len(batch_doc_idx))] = relevances[batch_doc_idx]
    batch_group_membership[range(len(batch_doc_idx))] = group_membership[batch_doc_idx]

    return X_batch, batch_relevance, batch_group_membership

def train_fair_nn(X_train, relevance_train, group_membership_train, num_items_per_query, CF_X_train = None, tf_prefix='', X_test=None,
                  group_membership_test = None, relevance_test=None, CF_X_test = None, weights=None, n_units=None, lr=0.001,
                  batch_size=100, epoch=100, verbose=True, activ_f=tf.nn.relu, l2_reg=0., plot=False,
                  lamb_init=2., adv_epoch=100, adv_step=1., epsilon=None, sens_directions=[], l2_attack=0.01, adv_epoch_full=10,
                  fair_reg=0., fair_start=0.5, seed=None, simul=False, load=False, num_monte_carlo_samples = 100, bias = True, init_range = -1,
                  entropy_regularizer=.01, baseline_ndcg=False, COUNTER_INIT = .05, PG = False, PG_reg = 0, T = 40):
    '''
    This code assumes that each query has the same number of items.

    Inputs:
    - X_train: an array of size (number of queries)*(num_items_per_query) by (dimension of item representation space) such that each row corresponds to a item in a query, i.e., the first num_items_per_query rows correspond to the items in the first query, the next num_items_per_query rows correspond to the items in the second query, etc.
    - relevance_train: (number of queries)*(number of items per query) by 1 array such that each entry corresponds to the relevance of an item in a query.
    - group_membership_train: (number of queries)*(number of items per query) by 1 array such that each entry corresponds to which of two groups each item belongs to where 1 indicates the majority group.
    - num_items_per_query: number of items in each query
    - CF_X_train: an array with the same dimensions as X_train that contain counterfactual queries (for instance if you want to manually flip the gender). Leave as "None" if you want to find the closest queries in the train set with respect to the fair distance.
    - tf_prefix: prefix for tensorboard files
    - X_test, group_membership_test, relevance_test, CF_X_test: same as training counterparts except you can leave as "None"
    - weights: Leave as "None" if you want to learn weights. Otherwise, if not "None", metrics will be computed but weights will not be learned with SenSTIR
    - n_units: "None" means no hidden layers. Otherwise, a list where each element is the number of nodes in each hidden layer
    - lr: learning rate for the Adam optimizer to learn the weights of the LTR model
    - batch_size: batch size
    - epoch: number of epochs
    - verbose: True means to print out various metrics while training
    - activ_f: activation function
    - l2_reg: l2 regularzation on the weights of the LTR model
    - plot: True means to save tensorboard files
    - lamb_init: initialization for lambda in SenSTIR
    - adv_epoch: number of epochs for finding counterfactuals in the fair subspace
    - adv_step: learning rate for Adam optimizer for finding counterfactuals in the fair subspace
    - epsilon: epsilon in the paper that shows up in the definition of the fair regularizer. set to "None" if you want to try without fair regularzation
    - sens_directions: (# sensitive directions) by (dimension of item representation space) array
    - l2_attack: learning rate for Adam optimizer for finding counterfactuals in the full space
    - adv_epoch_full: number of epochs for finding counterfactuals in the full space
    - fair_reg: fair regularization strength (rho in the paper)
    - fair_start: number in [0,1] that represents the fraction of epochs to train without fair regularzation
    - seed: number representing a split of the data
    - simul: True if using simulated data in R^2 and you want to make the plots in the paper
    - load: True if you simul is True and you want to load synthetic data
    - num_monte_carlo_samples: number of monte carlo samples to estimate the gradient and also for stochastic evaluation of metrics like NDCG
    - bias: True if you want to use a bias term in the LTR model
    - init_range: LTR model weights will be initialized in [-init_range, init_range] uniformly at random
    - entropy_regularizer: entropy regularization strength (used in Policy Learning for Fairness in Ranking paper). larger values encourages the LTR model to learn more stochastic distributions over rankings
    - baseline_ndcg: True means to subtract average NDCG over the monte carlo samples of each query when estimating the gradient (used in Policy Learning for Fairness in Ranking paper)
    - COUNTER_INIT: used in initializing counterfactual queries
    - PG: True means to use the fair regularizer in the Policy Learning for Fairness in Ranking paper
    - PG_reg: regularization strength for fair regularizer in the Policy Learning for Fairness in Ranking paper

    Output:
    - returns the learned weights of the LTR model and saves various evaluation metrics
    '''

    num_training_queries = int(X_train.shape[0] / num_items_per_query)

    # batch size cannot be larger than number of queries
    batch_size = np.min([batch_size, num_training_queries])

    # get minibatches for each epoch
    if batch_size == num_training_queries:
        batches_of_queries = np.array([np.arange(num_training_queries) for _ in range(epoch)])
    if batch_size == 1:
        batches_of_queries = np.zeros(epoch)
        for i in range(int(epoch/num_training_queries)):
            temp = np.arange(0,num_training_queries)
            np.random.shuffle(temp)
            batches_of_queries[i*num_training_queries:(i+1)*num_training_queries] = np.copy(temp)
        batches_of_queries = batches_of_queries.reshape(-1,1).astype(int)
    else:
        batches_of_queries = np.random.choice(num_training_queries, size=(epoch,batch_size))

    # number of items in each minibatch
    document_batch_size = num_items_per_query*batch_size

    K_protected = sens_directions.shape[0]

    proj_compl = compl_svd_projector(sens_directions, svd=-1)
    dist_f = fair_dist(proj_compl, 0.0)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    _, D = X_train.shape

    lamb = lamb_init

    train_loss_val = 0
    # used in computing the utility of a ranking policy objective
    lower_ones_matrix = np.tril(np.ones(num_items_per_query))

    tf_X = tf.placeholder(tf.float32, shape=[None,D], name='tf_X')
    # we need this since we use monte carlo sampling to estimate the objective
    tf_X_utility = tf.placeholder(tf.float32, shape=[None,D], name='tf_X_utility')
    tf_proj_compl = tf.constant(proj_compl, dtype = tf.float32, name = 'proj_compl')
    tf_lower_ones_matrix = tf.constant(lower_ones_matrix, dtype = tf.float32, name = 'lower_ones_matrix')
    tf_relevances = tf.placeholder(tf.float32, shape=[None], name='tf_relevances')
    tf_scores = tf.placeholder(tf.float32, shape=[None,num_items_per_query], name='tf_scores')
    #PG code uses log base 2
    position_bias = np.array([1 / np.log2(i + 2) for i in range(num_items_per_query)])
    tf_position_bias = tf.constant(np.array([1 / np.log2(i + 2) for i in range(num_items_per_query)]), shape = (1, num_items_per_query), dtype = tf.float32, name = 'position_bias')
    tf_group_membership = tf.placeholder(tf.float32, shape=[None], name='tf_group_membership')
    # used in the weighted kendall tau computation
    delta = [position_bias[i] - position_bias[i+1] for i in range(len(position_bias)-1)]
    p = np.array([1+np.sum(delta[:i]) for i in range(len(position_bias))])

    # for sampling rankings
    dist = tfp.distributions.PlackettLuce(tf_scores)
    tf_sample_ranking = dist.sample(num_monte_carlo_samples)

    ## Fair variables
    tf_counter_X = tf.placeholder(tf.float32, shape=[None,D], name='tf_counter_X')
    tf_directions = tf.constant(sens_directions, dtype=tf.float32)

    #in subspace
    adv_weights = tf.Variable(tf.zeros([document_batch_size,K_protected]))

    #out of subspace
    full_adv_weights = tf.Variable(tf.zeros([document_batch_size,D]))

    if fair_reg > 0:
        tf_fair_X = tf_counter_X + tf.matmul(adv_weights, tf_directions) + full_adv_weights
    else:
        tf_fair_X = tf_X + tf.matmul(adv_weights, tf_directions) + full_adv_weights

    variables, l_pred, l_pred_utility, l_pred_fair, l_pred_counter = forward_fair_reg(tf_X, tf_X_utility, tf_fair_X, tf_counter_X, weights=weights, n_units = n_units, activ_f = activ_f, bias = bias, init_range = init_range)

    # we multiply by fair_reg because (1) it won't effect maximizing the adversarial examples since it's like changing lambda and (2) when we update the parameters of the NN we need this term
    fair_subspace_loss = fair_reg*tf.math.reduce_sum(tf.squared_difference(l_pred, l_pred_fair))/2

    ## Attack in subspace
    fair_optimizer = tf.train.AdamOptimizer(learning_rate=adv_step)

    # this is making the adversarial examples in the subspace so that's why there is no -lambda d_x term
    # the negative is because we want to maximize
    fair_step = fair_optimizer.minimize(-fair_subspace_loss, var_list=[adv_weights], global_step=global_step)
    reset_fair_optimizer = tf.variables_initializer(fair_optimizer.variables())
    reset_adv_weights = adv_weights.assign(tf.zeros([document_batch_size,K_protected]))

    ## Attack out of subspaces
    tf_lamb = tf.placeholder(tf.float32, shape=(), name='lambda')

    # place holder for transport plan for each query
    # maybe can be changed to a 3d tensor to speed things up? originally was a list since query sizes can vary
    tf_pi_list = [tf.placeholder(tf.float32, shape = [num_items_per_query, num_items_per_query], name = 'pi_'+str(i)) for i in range(batch_size)]

    dist_loss = get_adv_objective(tf_pi_list, tf_X, tf_fair_X, num_items_per_query, tf_proj_compl)
    fair_loss = fair_subspace_loss - tf_lamb*dist_loss

    tf_l2_attack = tf.placeholder(tf.float32, shape=(), name='full_attack_rate')
    if l2_attack > 0:
        full_fair_optimizer = tf.train.AdamOptimizer(learning_rate=tf_l2_attack)
        full_fair_step = full_fair_optimizer.minimize(-fair_loss, var_list=[full_adv_weights], global_step=global_step)
        reset_full_fair_optimizer = tf.variables_initializer(full_fair_optimizer.variables())
        reset_full_adv_weights = full_adv_weights.assign(tf.zeros([document_batch_size,D]))

    # in order to evaluate metrics on the mini batches
    ndcg, dcg, _train_loss, group_exposure_train_stochastic, _results = get_utility_objective(tf_group_membership, num_items_per_query, batch_size, baseline_ndcg, num_monte_carlo_samples, get_tf_ones_mean_matrix(num_monte_carlo_samples, batch_size), l_pred_utility, tf_lower_ones_matrix, tf_position_bias, tf.reshape(tf_relevances, shape=(num_monte_carlo_samples*batch_size, num_items_per_query)), PG = PG, PG_reg = PG_reg)
    # in order to evaluate metrics on the training set
    ndcg_train, dcg_train, _, group_exposure_train_stochastic_all, _ = get_utility_objective(tf_group_membership, num_items_per_query, int(X_train.shape[0]/num_items_per_query), baseline_ndcg, num_monte_carlo_samples, get_tf_ones_mean_matrix(num_monte_carlo_samples, int(X_train.shape[0] / num_items_per_query)), l_pred_utility, tf_lower_ones_matrix, tf_position_bias, tf.reshape(tf_relevances, shape=(int(num_monte_carlo_samples*X_train.shape[0]/num_items_per_query), num_items_per_query)))

    if X_test is not None:
        # in order to evaluate metrics on the test set stochastically (ie sampling from the learned distributions)
        ndcg_test, dcg_test, _, group_exposure_test_stochastic, alpha = get_utility_objective(tf_group_membership, num_items_per_query, int(X_test.shape[0]/num_items_per_query), baseline_ndcg, num_monte_carlo_samples, get_tf_ones_mean_matrix(num_monte_carlo_samples, int(X_test.shape[0]/num_items_per_query)), l_pred_utility, tf_lower_ones_matrix, tf_position_bias, tf.reshape(tf_relevances, shape=(int(num_monte_carlo_samples*X_test.shape[0]/num_items_per_query), num_items_per_query)))

        # in order to evaluate metrics on the test set deterministically (ie ranking via sorting by scores)
        ndcg_test_deterministic, _, _, group_exposure_test_deterministic, _ = get_utility_objective(tf_group_membership, num_items_per_query, int(X_test.shape[0]/num_items_per_query), baseline_ndcg, 1, get_tf_ones_mean_matrix(1, int(X_test.shape[0]/num_items_per_query)), l_pred_utility, tf_lower_ones_matrix, tf_position_bias, tf.reshape(tf_relevances, shape=(int(X_test.shape[0]/num_items_per_query), num_items_per_query)))

    # we want to maximize but using a minimizer so multiply by -1
    train_loss = -tf.reduce_sum(_train_loss) / float(batch_size*num_monte_carlo_samples)
    # add l_2 regularization of weights
    for t in range(T):
        train_loss += l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(t) + str(l)])  for l in range(len(n_units) + 1)])
    # entropy regularizer
    probs = tf.math.softmax(tf.reshape(l_pred_utility, shape=(num_monte_carlo_samples*batch_size, num_items_per_query)))
    # we do not negate entropy since we want to maximize entropy but feed it into a minimzer
    entropy = probs*tf.log(probs)
    train_loss += (entropy_regularizer / float(batch_size*num_monte_carlo_samples))*tf.reduce_sum(entropy)
    if fair_reg > 0:
        # fair subspace loss is just the distance between the scores of the original and adversarial examples
        train_loss += fair_subspace_loss / float(batch_size)
    ## Train step
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    if weights is None:
        train_step = optimizer.minimize(train_loss, var_list=list(variables.values()), global_step=global_step)
        reset_optimizer = tf.variables_initializer(optimizer.variables())
        reset_main_step = True
    ###################### CONTINUE

    failed_attack_count = 0
    failed_full_attack = 0
    failed_subspace_attack = 0

    out_freq = 1000
    save_freq = 100000
    fair_start = int(epoch*fair_start)

    baseline_saved = False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for it in range(epoch):
            do_upd = True
            batch_x, batch_relevance, batch_group_membership = get_minibatch(X_train, relevance_train, batches_of_queries, num_items_per_query, document_batch_size, it, group_membership_train)

            if it >= fair_start:
                if fair_reg>0.:
                    # initialization to stay away from 0 since gradient is 0 there
                    batch_flip_x = batch_x + np.matmul(np.random.uniform(-COUNTER_INIT, COUNTER_INIT, size=(document_batch_size,K_protected)), sens_directions)

                if reset_main_step:
                    sess.run(reset_optimizer)
                    reset_main_step = False

                if (not baseline_saved) and (fair_start>0):
                    print('Saving baseline before starting fair training')

                    try:
                        os.makedirs(RESULTS_BASE)
                    except:
                        pass

                    # saver.save(sess,
                    #          os.path.join(tb_dir, 'baseline_model'),
                    #          global_step=global_step)

                    weights = [x.eval() for x in variables.values()]
                    np.save(RESULTS_BASE + tf_prefix + '_' + tb_long + '_' + 'baseline-weights' + '_' + str(post_idx), weights)

                    print('Baseline train saved')
                    baseline_saved = True

                ## SenSeI begins
                if fair_reg > 0:
                    all_dict = {tf_X: batch_x, tf_lamb: lamb, tf_l2_attack: l2_attack, tf_counter_X: batch_flip_x, tf_relevances: batch_relevance}

                    fair_X = tf_fair_X.eval(feed_dict = all_dict)
                    for i in range(batch_size):
                        all_dict['pi_{}:0'.format(str(i))] = query_distance_or_plan(proj_compl, batch_x[i*num_items_per_query:(i+1)*num_items_per_query, :], fair_X[i*num_items_per_query:(i+1)*num_items_per_query, :], distance = False)
                    X_dict = {tf_X: batch_x, tf_counter_X: batch_flip_x}

                loss_before_subspace_attack = fair_loss.eval(feed_dict=all_dict)
                ## Do subspace attack
                for adv_it in range(adv_epoch):
                    fair_step.run(feed_dict=all_dict)

                ## Check result
                loss_after_subspace_attack = fair_loss.eval(feed_dict=all_dict)

                if loss_after_subspace_attack < loss_before_subspace_attack:
                    print(loss_after_subspace_attack, loss_before_subspace_attack)
                    print('WARNING: subspace attack failed: objective decreased from %f to %f; resetting the attack' % (loss_before_subspace_attack, loss_after_subspace_attack))
                    sess.run(reset_adv_weights)
                    failed_subspace_attack += 1

                if l2_attack > 0:
                    fair_X = tf_fair_X.eval(feed_dict = all_dict)
                    for i in range(batch_size):
                        all_dict['pi_{}:0'.format(str(i))] = query_distance_or_plan(proj_compl, batch_x[i*num_items_per_query:(i+1)*num_items_per_query, :], fair_X[i*num_items_per_query:(i+1)*num_items_per_query, :], distance = False)

                    fair_loss_before_l2_attack = fair_loss.eval(feed_dict=all_dict)
                    ## Do full attack
                    for full_adv_it in range(adv_epoch_full):
                        full_fair_step.run(feed_dict=all_dict)

                    ## Check result
                    fair_loss_after_l2_attack = fair_loss.eval(feed_dict=all_dict)
                    #print('fair_loss_before_l2_attack', fair_loss_before_l2_attack)
                    if fair_loss_after_l2_attack < fair_loss_before_l2_attack:
                        print('WARNING: full attack failed: objective decreased from %f to %f; skipping update steps' % (fair_loss_before_l2_attack, fair_loss_after_l2_attack))
                        failed_full_attack += 1
                        do_upd = False
                        l2_attack *= 0.999
                        print('Decreasing learning rate: new rate is %f' % l2_attack)

                adv_batch = tf_fair_X.eval(feed_dict=X_dict)

                if np.isnan(adv_batch.sum()):
                    print('Nans in adv_batch; making no change')
                    sess.run(reset_adv_weights)
                    if l2_attack > 0:
                        sess.run(reset_full_adv_weights)
                    failed_attack_count += 1
                    do_upd = False

                elif epsilon is not None:
                    if do_upd:
                        fair_X = tf_fair_X.eval(feed_dict = all_dict)
                        mean_dist = np.mean(evaluate_query_distance(batch_x, fair_X, proj_compl, batch_size, num_items_per_query))
                        lamb = max(0.00001,lamb + (max(mean_dist,epsilon)/min(mean_dist,epsilon))*(mean_dist - epsilon))

                ranking_list = get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, all_dict, sess)

                batch_x_utility, batch_relevance_utility, batch_group_membership_utility = prepare_data_for_utility_objective(batch_x, batch_relevance, ranking_list, batch_group_membership)
                all_dict[tf_X_utility] = batch_x_utility
                all_dict[tf_relevances] = batch_relevance_utility
                all_dict[tf_group_membership] = batch_group_membership_utility
            else:
                ## Baseline training
                adv_batch = batch_x
                if fair_reg > 0:
                    all_dict = {tf_X: batch_x, tf_lamb: lamb, tf_counter_X: batch_x, tf_relevances: batch_relevance, tf_group_membership: batch_group_membership}
                    # import pdb; pdb.set_trace() 
                    ranking_list = get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, all_dict, sess)
                    batch_x_utility, batch_relevance_utility, batch_group_membership_utility = prepare_data_for_utility_objective(batch_x, batch_relevance, ranking_list, batch_group_membership)
                    all_dict[tf_X_utility] = batch_x_utility
                    all_dict[tf_relevances] = batch_relevance_utility
                    all_dict[tf_group_membership] = batch_group_membership_utility
                    fair_X = tf_fair_X.eval(feed_dict = all_dict)
#                     print('fair_X:',fair_X)
                    for i in range(batch_size):
                        all_dict['pi_{}:0'.format(str(i))] = query_distance_or_plan(proj_compl, batch_x[i*num_items_per_query:(i+1)*num_items_per_query, :], fair_X[i*num_items_per_query:(i+1)*num_items_per_query, :], distance = False)

                    X_dict = {tf_X: batch_x, tf_counter_X: batch_x}
                else:
                    all_dict = {tf_X: batch_x, tf_counter_X: batch_x}
                    ranking_list = get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, all_dict, sess)
                    batch_x_utility, batch_relevance_utility, batch_group_membership_utility= prepare_data_for_utility_objective(batch_x, batch_relevance, ranking_list, batch_group_membership)
                    all_dict[tf_X_utility] = batch_x_utility
                    all_dict[tf_relevances] = batch_relevance_utility
                    all_dict[tf_group_membership] = batch_group_membership_utility
                    X_dict = {tf_X: batch_x}

            ## Parameter update step

            if do_upd and fair_reg >0:
                _, loss_at_update = sess.run([train_step,fair_loss], feed_dict=all_dict)
            elif do_upd and fair_reg <=0:
                sess.run(train_step, feed_dict=all_dict)
                loss_at_update = -1
            else:
                loss_at_update = fair_loss.eval(feed_dict=all_dict)

            if it % out_freq == 0 and verbose:
                fair_X = tf_fair_X.eval(feed_dict = all_dict)
                tf_dist = evaluate_query_distance(batch_x, fair_X, proj_compl, batch_size, num_items_per_query)
            if it > fair_start:
                sess.run(reset_adv_weights)
                sess.run(reset_fair_optimizer)
                if l2_attack > 0:
                    sess.run(reset_full_fair_optimizer)
                    sess.run(reset_full_adv_weights)

            if (it % out_freq == 0 or it == epoch - 1) and verbose:
                print('----iteration------', it/epoch)
                NDCG_train = np.mean(ndcg.eval(feed_dict = all_dict))
                train_loss_val = train_loss.eval(feed_dict = all_dict)
                train_exposure_stochastic_val = group_exposure_train_stochastic.eval(feed_dict = all_dict)
                print('NDCG', NDCG_train)
                print('train loss', train_loss_val)
                print('train exposure (stochastic)', train_exposure_stochastic_val)

                dd = evaluate_query_distance(batch_x, fair_X, np.eye(batch_x.shape[1]), batch_size, num_items_per_query)

                print('Real and fair distances (max/min/mean):')
                print(dd.max(), dd.min(), dd.mean())
                print(tf_dist.max(), tf_dist.min(), tf_dist.mean())

                if relevance_test is not None:
                    all_dict = {tf_X: X_test, tf_lamb: lamb, tf_l2_attack: l2_attack, tf_relevances: relevance_test, tf_group_membership: group_membership_test}
                    ranking_list = get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, all_dict, sess)
                    X_utility, relevance_utility, membership_utility = prepare_data_for_utility_objective(X_test, relevance_test, ranking_list, group_membership_test)
                    all_dict[tf_X_utility] = X_utility
                    all_dict[tf_relevances] = relevance_utility
                    all_dict[tf_group_membership] = membership_utility
                    test_logits = sess.run(l_pred, feed_dict=all_dict)
                    NDCG_test = np.mean(ndcg_test.eval(feed_dict=all_dict))

                    print('\ntest NDCG (stochastic)', NDCG_test)
                    train_all_dict = {tf_X: X_train, tf_lamb: lamb, tf_l2_attack: l2_attack, tf_relevances: relevance_train}
                    ranking_list = get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, train_all_dict, sess)
                    X_utility, relevance_utility, membership_utility = prepare_data_for_utility_objective(X_train, relevance_train, ranking_list, group_membership_train)
                    train_all_dict[tf_X_utility] = X_utility
                    train_all_dict[tf_relevances] = relevance_utility
                    train_all_dict[tf_group_membership] = membership_utility
                    ndcg_train_all_val = np.mean(ndcg_train.eval(feed_dict=train_all_dict))
                    print('\ntrain NDCG (stochastic)',ndcg_train_all_val , fair_reg)
                    group_exposure_stochastic_val = group_exposure_test_stochastic.eval(feed_dict = all_dict)
                    print('test group exposure (stochastic)', group_exposure_stochastic_val)

                    ranking_list = get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, all_dict, sess, deterministic = True)
                    X_utility, relevance_utility, membership_utility = prepare_data_for_utility_objective(X_test, relevance_test, ranking_list, group_membership_test)
                    all_dict[tf_X_utility] = X_utility
                    all_dict[tf_relevances] = relevance_utility
                    all_dict[tf_group_membership] = membership_utility
                    NDCG_test_deterministic = np.mean(ndcg_test_deterministic.eval(feed_dict = all_dict))
                    print('\ntest NDCG (deterministic)', NDCG_test_deterministic)
                    group_test_exposure_deterministic_val = group_exposure_test_deterministic.eval(feed_dict = all_dict)
                    print('test group exposure (deterministic)', group_test_exposure_deterministic_val)
                    train_logits = sess.run(l_pred, feed_dict=all_dict)

                else:
                    test_logits = None

                ## Debugging:
                if it > fair_start:
                    print('FAILED attacks: subspace %d; full %d; Nans after attack %d' % (failed_subspace_attack, failed_full_attack, failed_attack_count))
                    print('before subspace {}; after subspace {}; before l2 {}; after l2 {}'.format(loss_before_subspace_attack, loss_after_subspace_attack, fair_loss_before_l2_attack, fair_loss_after_l2_attack))

                if plot:
                    summary = tf.Summary(value=[
                    tf.Summary.Value(tag='train NDCG', simple_value = NDCG_train),
                    tf.Summary.Value(tag='test NDCG (stochastic)', simple_value = NDCG_test),
                    tf.Summary.Value(tag='test NDCG (deterministic)', simple_value = NDCG_test_deterministic),
                    tf.Summary.Value(tag='train loss', simple_value = train_loss_val),
                    tf.Summary.Value(tag='lambda', simple_value = lamb),
                    tf.Summary.Value(tag='L2 max', simple_value = dd.max()),
                    tf.Summary.Value(tag='L2 mean', simple_value = dd.mean()),
                    tf.Summary.Value(tag='Fair distance max', simple_value = tf_dist.max()),
                    tf.Summary.Value(tag='Fair distance mean', simple_value = tf_dist.mean()),
                    tf.Summary.Value(tag='Distance mean difference', simple_value = dd.mean() - tf_dist.mean()),
                    tf.Summary.Value(tag='Distance max difference', simple_value = dd.max() - tf_dist.max()),
                    tf.Summary.Value(tag='Test exposure (stochastic)', simple_value = group_exposure_stochastic_val),
                    tf.Summary.Value(tag='Test exposure (deterministic)', simple_value = group_test_exposure_deterministic_val),
                    tf.Summary.Value(tag='Train exposure (stochastic)', simple_value = train_exposure_stochastic_val),
                    tf.Summary.Value(tag ='Train NDCG all (stochastic)', simple_value = ndcg_train_all_val)]
                    )
                    summary_writer.add_summary(summary, it)
                    summary_writer.flush()

                sys.stdout.flush()

        norm = 0
        train_stochastic_metrics = [failed_full_attack, failed_subspace_attack, norm]
        train_deterministic_metrics = [failed_full_attack, failed_subspace_attack, norm]
        test_stochastic_metrics = [failed_full_attack, failed_subspace_attack, norm]
        test_deterministic_metrics = [failed_full_attack, failed_subspace_attack, norm]

        print('l_preds.eval(all_dict = {tf_X: proj_compl})', l_pred.eval(feed_dict = {tf_X: sens_directions}))
        if relevance_train is not None:
            if fair_reg <= 0:
                batch_flip_x = X_train
            all_dict = {tf_X: X_train, tf_lamb: lamb, tf_l2_attack: l2_attack, tf_counter_X: batch_flip_x, tf_relevances: relevance_train}
            ranking_list = get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, all_dict, sess)
            X_utility, relevance_utility, membership_utility = prepare_data_for_utility_objective(X_train, relevance_train, ranking_list, group_membership_train)
            all_dict[tf_X_utility] = X_utility
            all_dict[tf_relevances] = relevance_utility
            all_dict[tf_group_membership] = membership_utility
            train_logits = sess.run(l_pred, feed_dict=all_dict)


            weights = [x.eval() for x in variables.values()]

            if CF_X_train is None:
               
            print('\nFinal train DCG', np.mean(dcg_train.eval(feed_dict=all_dict)), fair_reg)
            print('\nFinal train NDCG (stochastic)', np.mean(ndcg_train.eval(feed_dict=all_dict)), fair_reg)
            train_stochastic_metrics.append(np.mean(ndcg_train.eval(feed_dict=all_dict)))
            print('\nFinal train group exposure (stochastic)', np.mean(group_exposure_train_stochastic_all.eval(feed_dict=all_dict)), fair_reg)
            train_stochastic_metrics.append(np.mean(group_exposure_train_stochastic_all.eval(feed_dict=all_dict)))
            _indv_exposure = compute_individual_exposure(position_bias, ranking_list, relevance_train)
            print('\nFinal train individual exposure (stochastic)', _indv_exposure, fair_reg)
            X_utility, relevance_utility, membership_utility = prepare_data_for_utility_objective(X_train, relevance_train, ranking_list, group_membership_train)
            all_dict[tf_X_utility] = X_utility
            all_dict[tf_relevances] = relevance_utility
            all_dict[tf_group_membership] = membership_utility

            ndcg_train, dcg_train, _, group_exposure_train_deterministic, _ = get_utility_objective(tf_group_membership, num_items_per_query, int(X_train.shape[0]/num_items_per_query), baseline_ndcg, 1, get_tf_ones_mean_matrix(1, int(X_train.shape[0] / num_items_per_query)), l_pred_utility, tf_lower_ones_matrix, tf_position_bias, tf.reshape(tf_relevances, shape=(int(1*X_train.shape[0]/num_items_per_query), num_items_per_query)))
            print('\nFinal train NDCG (deterministic)', np.mean(ndcg_train.eval(feed_dict=all_dict)), fair_reg)
            train_deterministic_metrics.append(np.mean(ndcg_train.eval(feed_dict=all_dict)))
            print('\nFinal train group exposure (deterministic)', np.mean(group_exposure_train_deterministic.eval(feed_dict=all_dict)), fair_reg)
            train_deterministic_metrics.append(np.mean(group_exposure_train_deterministic.eval(feed_dict=all_dict)))
            _indiv_exposure = compute_individual_exposure(position_bias, ranking_list, relevance_train)
            print('\nFinal train individual exposure (deterministic)', _indiv_exposure, fair_reg)
            if simul:
                if load:
                    xx = np.load('data/xx.npy')
                    yy = np.load('data/yy.npy')
                else:
                    h = .02
                    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
                    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                         np.arange(y_min, y_max, h))
                    np.save('data/xx.npy', xx)
                    np.save('data/yy.npy', yy)

                weights = [x.eval() for x in variables.values()]
                all_dict[tf_X] = np.c_[xx.ravel(), yy.ravel()]
                Z = l_pred.eval(feed_dict = all_dict)
                np.save('data/Z_{}.npy'.format(str(fair_reg)), Z)

        if relevance_test is not None:

            all_dict = {tf_X: X_test, tf_lamb: lamb, tf_l2_attack: l2_attack, tf_relevances: relevance_test}
            ranking_list = get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, all_dict, sess)
            X_utility, relevance_utility, membership_utility = prepare_data_for_utility_objective(X_test, relevance_test, ranking_list, group_membership_test)
            all_dict[tf_X_utility] = X_utility
            all_dict[tf_relevances] = relevance_utility
            all_dict[tf_group_membership] = membership_utility
            test_logits = sess.run(l_pred, feed_dict=all_dict)
            ndcg_test, dcg_test, _, group_exposure_test_stochastic, results = get_utility_objective(tf_group_membership, num_items_per_query, int(X_test.shape[0] / num_items_per_query), baseline_ndcg, num_monte_carlo_samples, get_tf_ones_mean_matrix(num_monte_carlo_samples, int(X_test.shape[0] / num_items_per_query)), l_pred_utility, tf_lower_ones_matrix, tf_position_bias, tf.reshape(tf_relevances, shape=(int(num_monte_carlo_samples*X_test.shape[0]/num_items_per_query), num_items_per_query)))

            test_NDCG = np.mean(ndcg_test.eval(feed_dict=all_dict))
            print('\nFinal test NDCG (stochastic)', test_NDCG, fair_reg)

            if CF_X_test is None:
                
            print('PREFIX', tf_prefix)
            print('\nFinal test DCG', np.mean(dcg_test.eval(feed_dict=all_dict)), fair_reg)
            test_NDCG = np.mean(ndcg_test.eval(feed_dict=all_dict))
            print('\nFinal test NDCG (stochastic)', test_NDCG, fair_reg)
            test_stochastic_metrics.append(test_NDCG)
            print('\nFinal group_exposure_test_stochastic', np.mean(group_exposure_test_stochastic.eval(feed_dict=all_dict)), fair_reg)
            test_stochastic_metrics.append(np.mean(group_exposure_test_stochastic.eval(feed_dict=all_dict)))
            _indiv_exposure = compute_individual_exposure(position_bias, ranking_list, relevance_test)
            print('\nFinal test individual exposure (stochastic)', _indiv_exposure, fair_reg)
            test_stochastic_metrics.append(_indiv_exposure)
            
            if CF_X_test is not None:
            ranking_list = get_rankings_list(tf_sample_ranking, num_monte_carlo_samples, tf_scores, num_items_per_query, l_pred, all_dict, sess, deterministic = True)
            ndcg_test, dcg_test, _, group_exposure_test_deterministic, results = get_utility_objective(tf_group_membership, num_items_per_query, int(X_test.shape[0] / num_items_per_query), baseline_ndcg, 1, get_tf_ones_mean_matrix(1, int(X_test.shape[0] / num_items_per_query)), l_pred_utility, tf_lower_ones_matrix, tf_position_bias, tf.reshape(tf_relevances, shape=(int(1*X_test.shape[0]/num_items_per_query), num_items_per_query)))


            X_utility, relevance_utility, membership_utility = prepare_data_for_utility_objective(X_test, relevance_test, ranking_list, group_membership_test)
            all_dict[tf_X_utility] = X_utility
            all_dict[tf_relevances] = relevance_utility
            all_dict[tf_group_membership] = membership_utility
            tic = time.perf_counter()
            if CF_X_test is None:
            
            print('\nFinal test NDCG (deterministic)', np.mean(ndcg_test.eval(feed_dict=all_dict)), fair_reg)
            test_deterministic_metrics.append(np.mean(ndcg_test.eval(feed_dict=all_dict)))
            print('\nFinal group_exposure_test_deterministic (deterministic)', np.mean(group_exposure_test_deterministic.eval(feed_dict=all_dict)), fair_reg)
            test_deterministic_metrics.append(np.mean(group_exposure_test_deterministic.eval(feed_dict=all_dict)))
            _indv_exposure = compute_individual_exposure(position_bias, ranking_list, relevance_test)
            print('\nFinal test individual exposure (deterministic)', _indv_exposure, fair_reg)
            test_deterministic_metrics.append(_indv_exposure)
            
        weights = [x.eval() for x in variables.values()]

        try:
            os.makedirs(RESULTS_BASE)
        except:
            pass
        np.save(RESULTS_BASE + tf_prefix + '_' + tb_long + '_' + 'fair-weights' + '_' + str(post_idx), weights)

        try:
            os.makedirs(METRICS_BASE)
        except:
            pass
        np.save(METRICS_BASE + tf_prefix + '_' + tb_long + '_' + 'train_stochastic_metrics' + '_' + str(post_idx), train_stochastic_metrics)
        np.save(METRICS_BASE + tf_prefix + '_' + tb_long + '_' + 'train_deterministic_metrics' + '_' + str(post_idx), train_deterministic_metrics)
        np.save(METRICS_BASE + tf_prefix + '_' + tb_long + '_' + 'test_stochastic_metrics' + '_' + str(post_idx), test_stochastic_metrics)
        np.save(METRICS_BASE + tf_prefix + '_' + tb_long + '_' + 'test_deterministic_metrics' + '_' + str(post_idx), test_deterministic_metrics)

        try:
            os.makedirs(HEATMAPS_BASE)
        except:
            pass

        # np.save(HEATMAPS_BASE + tf_prefix + '_' + tb_long + '_' + 'heatmap_train_stochastic' + '_' + str(post_idx), heatmap_train_stochastic)
        # np.save(HEATMAPS_BASE + tf_prefix + '_' + tb_long + '_' + 'heatmap_train_deterministic' + '_' + str(post_idx), heatmap_train_deterministic)
        np.save(HEATMAPS_BASE + tf_prefix + '_' + tb_long + '_' + 'heatmap_test_stochastic' + '_' + str(post_idx), heatmap_test_stochastic)
        np.save(HEATMAPS_BASE + tf_prefix + '_' + tb_long + '_' + 'heatmap_test_deterministic' + '_' + str(post_idx), heatmap_test_deterministic)

        try:
            os.makedirs(DISPLACEMENT_BASE)
        except:
            pass
        # np.save(DISPLACEMENT_BASE + tf_prefix + '_' + tb_long + '_' + 'displacement_train_stochastic' + '_' + str(post_idx), displacement_train_stochastic)
        # np.save(DISPLACEMENT_BASE + tf_prefix + '_' + tb_long + '_' + 'displacement_train_deterministic' + '_' + str(post_idx), displacement_train_deterministic)
        np.save(DISPLACEMENT_BASE + tf_prefix + '_' + tb_long + '_' + 'displacement_test_stochastic' + '_' + str(post_idx), displacement_test_stochastic)
        np.save(DISPLACEMENT_BASE + tf_prefix + '_' + tb_long + '_' + 'displacement_test_deterministic' + '_' + str(post_idx), displacement_test_deterministic)

        try:
            os.makedirs(NAN_FAIL)
        except:
            pass
        if math.isnan(train_loss_val):
            np.save(NAN_FAIL + tf_prefix + '_' + tb_long + '_' + 'failed' + '_' + str(post_idx), [0])

        if fair_reg <= 0:
            try:
                os.makedirs(RESULTS_BASE)
            except:
                pass

            weights = [x.eval() for x in variables.values()]
            np.save(RESULTS_BASE + tf_prefix + '_' + tb_long + '_' + 'baseline-weights' + '_' + str(post_idx), weights)

    return weights
