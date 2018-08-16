import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-nf_layer_num', type=int)
parser.add_argument('-test_energy_fun', type=int)
parser.add_argument('-option', type=str)

args = parser.parse_args()

nf_layer_num = args.nf_layer_num
test_energy_fun = args.test_energy_fun
option = args.option

def normalizing_flow(input):
    w = tf.get_variable('w', [2, 1], initializer=tf.random_normal_initializer(0, 0.1))
    b = tf.get_variable('b', [1], initializer=tf.zeros_initializer())
    u = tf.get_variable('u', [2, 1], initializer=tf.random_normal_initializer(0, 0.1))

    activation = tf.tanh(tf.matmul(input, w) + b) # shape: (?, 1)
    output = input + tf.matmul(activation, tf.transpose(u)) #shape: (?, 2)

    psi = tf.matmul((1 - activation ** 2), tf.transpose(w)) #shape: (?, 2)

    abs_log_det_jacobian = tf.log(tf.abs(1 + tf.matmul(psi, u)))

    return output, abs_log_det_jacobian

def U_z(z, test_energy_fun):
    """Test energy function U(z)."""
    z1 = z[:, 0]
    z2 = z[:, 1]

    if test_energy_fun == 1:
        return 0.5*((tf.sqrt(z1**2 + z2**2) - 2)/0.4)**2 - tf.log(tf.exp(-0.5*((z1 - 2)/0.6)**2) + tf.exp(-0.5*((z1 + 2)/0.6)**2))
    elif test_energy_fun == 2:
        w1 = tf.sin((2.*np.pi*z1)/4.)
        return 0.5*((z2 - w1) / 0.4)**2
    elif test_energy_fun == 3:
        w1 = tf.sin((2.*np.pi*z1)/4.)
        w2 = 3.*tf.exp(-0.5*((z1 - 1)/0.6)**2)
        return -tf.log(tf.exp(-0.5*((z2 - w1)/0.35)**2) + tf.exp(-0.5*((z2 - w1 + w2)/0.35)**2))
    elif test_energy_fun == 4:
        w1 = tf.sin((2.*np.pi*z1)/4.)
        w3 = 3.*tf.sigmoid((z1 - 1)/0.3)**4
        return -tf.log(tf.exp(-0.5*((z2 - w1)/0.4)**2) + tf.exp(-0.5*((z2 - w1 + w3)/0.35)**2))
    else:
        raise ValueError('invalid `test_energy_fun`')


def evaluate_bivariate_pdf(p_z, range, npoints, sess):
    """Evaluate (possibly unnormalized) pdf over a meshgrid."""
    side = np.linspace(range[0], range[1], npoints)
    z1, z2 = np.meshgrid(side, side)
    z = np.hstack([z1.reshape(-1, 1), z2.reshape(-1, 1)])

    prob = sess.run(tf.exp(p_z), feed_dict={z_0:z})

    return z1, z2, prob

def evaluate_bivariate_pdf_transformed(p_z, z_k, range, npoints, sess):
    """Evaluate (possibly unnormalized) pdf over a meshgrid."""
    side = np.linspace(range[0], range[1], npoints)
    z1, z2 = np.meshgrid(side, side)
    z = np.hstack([z1.reshape(-1, 1), z2.reshape(-1, 1)])

    prob = sess.run(tf.exp(p_z), feed_dict={z_0:z})

    z_k = sess.run(z_k, feed_dict={z_0: z})
    z1 = z_k[:, 0].reshape((npoints, npoints))
    z2 = z_k[:, 1].reshape((npoints, npoints))
    return z1, z2, prob

def run_plot(epoch, sess):
    fig, ax_list = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

    plot_name_list = ['$p(z_0)$', '$q_0(z_0)$', '$q_k(z_k)$']
    plot_object = [log_p_z0, log_q0_z0, log_qk_zk]

    for i, ax, name, log_prob in zip(range(4), ax_list, plot_name_list, plot_object):
        if name == '$q_k(z_k)$':
            z1, z2, prob_val = evaluate_bivariate_pdf_transformed(log_prob, z_k, range=(-4,
                                                                                     4),
                                                      npoints=200, sess=sess)
        else:
            z1, z2, prob_val = evaluate_bivariate_pdf(log_prob, range=(-4, 4),
                                                    npoints=200, sess=sess)
        im = ax.pcolormesh(z1, z2, prob_val.reshape(200, 200))
        fig.colorbar(im, ax=ax)
        ax.set_title(name)

    plt.savefig('{}/plot_{}'.format(save_dir, epoch))
    plt.close()



output_list = list()
abs_log_det_jacobian_sum = 0

z_0 = tf.placeholder(tf.float32, [None, 2])
anneal_beta = tf.placeholder(tf.float32, [1])
output_list.append(z_0)

for i in range(nf_layer_num):
    layer_input = output_list[-1]
    with tf.variable_scope('nf_layer_{}'.format(i)):
        output, abs_log_det_jacobian = normalizing_flow(layer_input)

    output_list.append(output)
    abs_log_det_jacobian_sum += abs_log_det_jacobian

z_k = output_list[-1]
std_n = tf.distributions.Normal([0.0, 0.0], [1.0, 1.0])

log_q0_z0 = tf.reduce_sum(std_n.log_prob(z_0), axis=1)
log_det_sum = tf.squeeze(abs_log_det_jacobian_sum, axis=1)
log_p_zk = -U_z(z_k, test_energy_fun)
log_p_z0 = -U_z(z_0, test_energy_fun)
log_qk_zk = log_q0_z0 - log_det_sum

kl_loss = tf.reduce_mean(log_qk_zk - log_p_zk * anneal_beta)



train_op = tf.train.RMSPropOptimizer(1e-5, momentum=0.9).minimize(kl_loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

save_dir = 'Uz-{}_nf_num-{}_option-{}'.format(test_energy_fun, nf_layer_num, option)
save_dir = os.path.join(save_dir, str(datetime.datetime.now()))

writer = tf.summary.FileWriter(save_dir, graph=sess.graph)


max_epoch = 1000000
for epoch in range(max_epoch):
    z_0_val = np.random.rand(100, 2)
    anneal_beta_val = np.minimum([1.0], 0.01 + epoch /10000)
    sess.run(train_op, feed_dict={z_0:z_0_val,
                                  anneal_beta: anneal_beta_val})

    if epoch % 10000 == 0:
        run_plot(epoch, sess)
        kl_loss_val = sess.run([kl_loss], feed_dict={z_0:z_0_val,
                                     anneal_beta: anneal_beta_val})
        print('kl_loss', kl_loss_val)
        print(save_dir)
        saver.save(sess, '{}/model'.format(save_dir), global_step=epoch)
        writer.flush()



