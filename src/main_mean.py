import math
import os
import sys
from numpy.random import seed
import numpy as np
# to deal with numpy randomness
# seed(1234)
import tensorflow.compat.v1 as tf
# added to deal with randomness
tf.set_random_seed(1234)

import tqdm
from numpy import random
from utils import *

def get_cosine_similarity(embeddings1, embeddings2, node):
    try:
        vector1 = embeddings1[node]
        vector2 = embeddings2[node]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 2+random.random()
		

def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield (np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32), np.array(neigh).astype(np.int32)) 

def train_model(network_data, log_name, store_file, all_walks):
    vocab, index2word = generate_vocab(all_walks)
    train_pairs = generate_pairs(all_walks, vocab, args.window_size)

    edge_types = list(network_data.keys())
    print('edge types: '+str(list(network_data.keys())))
    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    print('edge types '+ str(edge_types))
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions # Dimension of the embedding vector.
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples # Number of negative examples to sample.
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples 

    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        g = network_data[edge_types[r]]
        for (x, y) in g:
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(list(np.random.choice(neighbors[i][r], size=neighbor_samples-len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))

    graph = tf.Graph()

    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Parameters to learn
        node_embeddings = tf.Variable(tf.random.uniform([num_nodes, embedding_size], -1.0, 1.0))
        node_type_embeddings = tf.Variable(tf.random.uniform([num_nodes, u_num, embedding_u_size], -1.0, 1.0))
        trans_weights = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size // att_head], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s1 = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s2 = tf.Variable(tf.truncated_normal([edge_type_count, dim_a, att_head], stddev=1.0 / math.sqrt(embedding_size)))
        nce_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([num_nodes]))

        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None])
        node_neigh = tf.placeholder(tf.int32, shape=[None, edge_type_count, neighbor_samples])
        
        # Look up embeddings for nodes
        node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        
		# to get neighbors embeddings for all nodes in all types of edges: embedding_lookup(params, ids)
        node_embed_neighbors = tf.nn.embedding_lookup(node_type_embeddings, node_neigh)
        node_embed_tmp = tf.concat([tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, i, 0], [-1, 1, -1, 1, -1]), [1, -1, neighbor_samples, embedding_u_size]) for i in range(edge_type_count)], axis=0)
        node_type_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2), perm=[1,0,2])

        trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
        trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types)
        trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types)
        
        attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1)), trans_w_s2), [-1, u_num])), [-1, att_head, u_num])
        node_type_embed = tf.matmul(attention, node_type_embed)
        node_embed = node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])

        last_node_embed = tf.nn.l2_normalize(node_embed, axis=1)

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=last_node_embed,
                num_sampled=num_sampled,
                num_classes=num_nodes))
        plot_loss = tf.summary.scalar("loss", loss)

        # Optimizer.
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver(max_to_keep=20)

        merged = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
    print("Optimizing")
    
    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter("./runs/" + log_name, sess.graph) # tensorboard --logdir=./runs
        sess.run(init)

        print('Training')
        g_iter = 0
        best_score = 0
        patience = 0
        for epoch in range(epochs):
            random.shuffle(train_pairs)
            batches = get_batches(train_pairs, neighbors, batch_size)
            print(len(train_pairs))
            data_iter = tqdm.tqdm(batches,
                                desc="epoch %d" % (epoch),
                                total=(len(train_pairs) + (batch_size - 1)) // batch_size,
                                bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in enumerate(data_iter):
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2], node_neigh: data[3]}
                _, loss_value, summary_str = sess.run([optimizer, loss, merged], feed_dict)
                writer.add_summary(summary_str, g_iter)

                g_iter += 1

                avg_loss += loss_value

                if i % 5000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))
            
            final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
            for i in range(edge_type_count):
                for j in range(num_nodes):
                    final_model[edge_types[i]][index2word[j]] = np.array(sess.run(last_node_embed, {train_inputs: [j], train_types: [i], node_neigh: [neighbors[j]]})[0])
            
            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == 'all' or edge_types[i] in args.eval_type.split(','):
                    tmp_auc, tmp_f1, tmp_pr = evaluate(final_model[edge_types[i]], valid_true_data_by_edge[edge_types[i]], valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr = evaluate(final_model[edge_types[i]], testing_true_data_by_edge[edge_types[i]], testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print('valid auc:', np.mean(valid_aucs))
            print('valid pr:', np.mean(valid_prs))
            print('valid f1:', np.mean(valid_f1s))
			
            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)

            cur_score = np.mean(valid_aucs)
            if cur_score > best_score:
                best_score = cur_score
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    print('Early Stopping')
                    break
    return  average_auc, average_f1, average_pr

   
if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)

    log_name = file_name.split('/')[-1] + f'_evaltype_{args.eval_type}_b_{args.batch_size}_e_{args.epoch}'

    training_data_by_type = load_training_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')
    all_walks = generate_walks(training_data_by_type, args.num_walks, args.walk_length, file_name)
    average_auc, average_f1, average_pr = train_model(training_data_by_type, log_name + '_' + time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time())), 'embeddings1.txt', all_walks)
  
    print('Overall ROC-AUC:', average_auc)
    print('Overall PR-AUC', average_pr)
    print('Overall F1:', average_f1)
