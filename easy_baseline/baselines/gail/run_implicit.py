import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training_implicit as training_implicit
from baselines.ddpg.models_implicit import NoiseActor, Classifier, NoiseDropoutActor
from baselines.ddpg.models import Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.memory_implicit import FIFOMemory
from baselines.ddpg.noise import *
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier

from baselines.gail.implicit import setup_and_learn

import gym
import tensorflow as tf
from mpi4py import MPI

def make_env(env_id):
    if 'Sparse' in env_id:
        wrapper = None
        if env_id == 'SparseInvertedPendulum-v1':
            env = gym.make('InvertedPendulum-v1')
            wrapper = wrapper_InvertedPendulum
        if env_id == 'SparseInvertedDoublePendulum-v1':
            env = gym.make('InvertedDoublePendulum-v1')
            wrapper = wrapper_InvertedDoublePendulum
        if env_id == 'SparseHalfCheetah-v1':
            env = gym.make('HalfCheetah-v1')
            wrapper = wrapper_HalfCheetah
        if env_id == 'SparseReacher-v1':
            env = gym.make('Reacher-v1')
            wrapper = wrapper_Reacher
        assert wrapper is not None
        return wrapper(env)
    else:
        env = gym.make(env_id)
        return env


class CALLBACK(object):

    def __init__(self, arg):
        self.arg = arg

        dropout = arg['dropout']
        if dropout <= 0 or dropout >= 1:
            dropout = None 
        self.directory = 'ddpg_noisynet_imitate_{}/seed_{}actorlr_{}criticlr_{}classifier_{}noisetype_{}rhoW_{}rhob_{}entropy_{}dropout_{}gstep_{}timestepsperbatch_{}'.format(self.arg['env_id'],
                          self.arg['seed'], self.arg['actor_lr'], self.arg['critic_lr'], self.arg['classifier_lr'], 
                          self.arg['noise_type'], self.arg['rhoW'], self.arg['rhob'], self.arg['entropy_coeff'], self.arg['dropout'], self.arg['gstep'], self.arg['timesteps_per_batch'])
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def __call__(self, lcl, glb):
        # losses for policy
        np.save(self.directory + '/epoch_actor_losses_record', lcl['policy_losses_record']['actor_loss'])
        np.save(self.directory + '/epoch_critic_losses_record', lcl['policy_losses_record']['critic_loss'])
        np.save(self.directory + '/epoch_classifier_losses_record', lcl['policy_losses_record']['classifier_loss'])
        np.save(self.directory + '/epoch_approx_entropy_record', lcl['policy_losses_record']['entropy'])
        # losses for discriminator
        for k,v in lcl['discriminator_losses_record'].items():
            np.save(self.directory + '/' + k, v)
        # Save tensorflow models.
        saver = lcl['saver']
        sess = lcl['sess']
        saver.save(sess, self.directory + '/model/model')
        return False 

    def final_call(self, lcl, glb):
        # Save tensorflow models.
        saver = lcl['saver']
        sess = lcl['sess']
        saver.save(sess, self.directory + '/model/model')


def run(env_id, seed, noise_type, layer_norm, evaluation, actor_lr, critic_lr, classifier_lr, dropout,
    rho_W=-4, rho_b=-4, entropy_coeff=1.0, g_step=20, timesteps_per_batch=1024, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    #env = gym.make(env_id)
    env = make_env(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
    gym.logger.setLevel(logging.WARN)

    if evaluation and rank==0:
        #eval_env = gym.make(env_id)
        eval_env = make_env(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    fifomemory = FIFOMemory(limit=int(64))  # TODO: customize choosing of limit
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    if 0 < dropout and dropout < 1:
        actor = NoiseDropoutActor(nb_actions, rho_W=rho_W, rho_b=rho_b, layer_norm=layer_norm, p=dropout)
    else:
        actor = NoiseActor(nb_actions, rho_W=rho_W, rho_b=rho_b, layer_norm=layer_norm)
    classifier = Classifier(layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed_old = seed
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Build callback
    arg = {}
    arg['seed'] = seed_old
    arg['env_id'] = env_id
    arg['noise_type'] = noise_type
    arg['rhoW'] = rho_W
    arg['rhob'] = rho_b
    arg['entropy_coeff'] = entropy_coeff
    arg['actor_lr'] = actor_lr
    arg['critic_lr'] = critic_lr
    arg['classifier_lr'] = classifier_lr
    arg['dropout'] = dropout
    arg['gstep'] = g_step
    arg['timesteps_per_batch'] = timesteps_per_batch
    callback = CALLBACK(arg)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    setup_and_learn(env=env, eval_env=eval_env, action_noise=action_noise,
        actor=actor, critic=critic, classifier=classifier, memory=memory, fifomemory=fifomemory,
        actor_lr=actor_lr, critic_lr=critic_lr, classifier_lr=classifier_lr,
        callback=callback, entropy_coeff=entropy_coeff, g_step=g_step, timesteps_per_batch=timesteps_per_batch,
        **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Hopper-v1')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--classifier-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--classifier-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='none')  # choices are ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--rho-W', type=float, default=-4.0)
    parser.add_argument('--rho-b', type=float, default=-4.0)  # TODO: add factorized
    parser.add_argument('--dropout', type=float, default=0.5)  # dropout < 0 mean no dropout
    parser.add_argument('--entropy-coeff', type=float, default=1.0)

    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    parser.add_argument('--g_step', type=int, default=3)
    parser.add_argument('--d_step', type=int, default=1)
    parser.add_argument('--timesteps_per_batch', type=int, default=1024)
    parser.add_argument('--max_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_entcoeff', type=float, default=1e-3)
    parser.add_argument('--max_iters', help='max number of iterations', type=int, default=0)
    parser.add_argument('--d_stepsize', type=float, default=3e-4)  # step size for discriminator

    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    run(**args)
