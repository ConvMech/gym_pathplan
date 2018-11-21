import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train, nsteps,
                 ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        CLIPRANGE_Q = tf.placeholder(tf.float32, [])
        LR = tf.placeholder(tf.float32, [])
        LR_Q = tf.placeholder(tf.float32, [])
        ADV_BACKPROP = train_model.adv_backprop_for_policy
        ADV_PREDICT = train_model.adv

        entropy = 0  # TODO: add classifier

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, -CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        #pg_loss = -tf.reduce_mean(ADV_BACKPROP)  # TODO: clipping
        pg_loss = -tf.reduce_mean(tf.clip_by_value(ADV_BACKPROP, -CLIPRANGE_Q, CLIPRANGE_Q))
        qf_loss = .5 * tf.reduce_mean(tf.square(ADV_PREDICT - ADV))
        
        loss = pg_loss - entropy * ent_coef + vf_coef * vf_loss

        params_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/policy')
        params_value = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/vf')
        params_qvalue = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/adv')

        grads_policy = tf.gradients(loss, params_policy)
        grads_value = tf.gradients(loss, params_value)
        grads_qvalue = tf.gradients(qf_loss, params_qvalue)

        if max_grad_norm is not None:
        	grads_policy, _grad_norm = tf.clip_by_global_norm(grads_policy, max_grad_norm)
        	grads_value, _grad_norm = tf.clip_by_global_norm(grads_value, max_grad_norm)
        	grads_qvalue, _grad_norm = tf.clip_by_global_norm(grads_qvalue, max_grad_norm)

        grads_policy = list(zip(grads_policy, params_policy))
        grads_value = list(zip(grads_value, params_value))
        grads_qvalue = list(zip(grads_qvalue, params_qvalue))

        trainer_policy = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        trainer_value = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        trainer_qvalue = tf.train.AdamOptimizer(learning_rate=LR_Q, epsilon=1e-5)

        _train_p = trainer_policy.apply_gradients(grads_policy)
        _train_v = trainer_value.apply_gradients(grads_value)
        _train_q = trainer_qvalue.apply_gradients(grads_qvalue)

        def train(lr, lr_q, cliprange, cliprange_q, obs, returns, masks, actions, values):  # mask for rnn model
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, train_model.action:actions, ADV:advs, R:returns, LR:lr, LR_Q:lr_q,
                    CLIPRANGE:cliprange, CLIPRANGE_Q:cliprange_q, OLDVPRED:values}
            # train q function
            qloss, _ = sess.run([qf_loss, _train_q], td_map)
            # train the rest
            pgloss, vloss, _, _ = sess.run([pg_loss, vf_loss, _train_p, _train_v], td_map)
            return qloss, pgloss, vloss
        self.loss_names = ['qvalue_loss', 'policy_loss', 'value_loss']

        def save(save_path):
        	raise NotImplementedError

        def load(load_path):
        	raise NotImplementedError

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.save = save
        self.load = load
        self.sess = sess
        tf.global_variables_initializer().run(session=sess)
        

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.act_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        epinfos = []
        for _ in range(self.nsteps):  # TODO: each batch of action is correlated, should remove this
            actions, values = self.model.step(self.obs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.float32)
        last_values = self.model.value(self.obs)
        # discount/bootstrap value function
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values)), epinfos)

# obs, returns, masks, actions, values = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr, lr_q,
	      vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
	      log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
	      cliprangeq=100.0, save_interval=0, callback=None):
    
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(lr_q, float): lr_q = constfn(lr_q)
    else: assert callable(lr_q)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    if isinstance(cliprangeq, float): cliprangeq = constfn(cliprangeq)
    else: assert callable(cliprangeq)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    
    # save model using saver
    saver = tf.train.Saver()
    
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps // nbatch
    for update in range(1, nupdates + 1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        lrqnow = lr_q(frac)
        cliprangenow = cliprange(frac)
        cliprangeqnow = cliprangeq(frac)
        obs, returns, masks, actions, values, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        mblossvals = []
        # nonrecurrent version
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values))
                mblossvals.append(model.train(lrnow, lrqnow, cliprangenow, cliprangeqnow, *slices))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

        if callback is not None:
            eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
            eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])
            callback(locals(), globals())

    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

