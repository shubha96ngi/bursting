import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

bp.math.set_dt(0.04)
class BaseAMPASyn(bp.SynConn):
  def __init__(self, pre, post, conn, delay=0, g_max=1, E=20., alpha=12,
               beta=1, T=1, T_duration=1., tau=8.0, method='exp_auto'):
    super(BaseAMPASyn, self).__init__(pre=pre, post=post, conn=conn)

    # check whether the pre group has the needed attribute: "spike"
    self.check_pre_attrs('spike')

    # check whether the post group has the needed attribute: "input" and "V"
    self.check_post_attrs('input', 'V')
    self.conn_mat = self.conn.require('conn_mat')  # .astype(float)
    self.size = self.conn_mat.shape

    # parameters
    self.delay = delay
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration
    self.Omega = 99
    self.tau = tau

    # coupling strengths
    #if e_ij is None:
    e_ij = bm.random.normal(0.1, 0.02, size=self.size)
    self.e_ij = e_ij * self.conn_mat  # element-wise multiplication

    # use "LengthDelay" to store the spikes of the pre-synaptic neuron group
    self.delay_step = int(delay/bm.get_dt()) #5 se 6
    self.pre_spike = bm.LengthDelay(pre.spike, self.delay_step) # 6 se 6x10 where pre.spike = 10 values

    # store the arrival time of the pre-synaptic spikes
    self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)  # negative time



    # integral function
    self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)
    # integral function
    # self.integral = bp.odeint(self.derivative, method=method)

  def derivative(self, g, t, TT):
    #dg = self.alpha * TT * (5*(1 - g)/(1 + bm.exp(-(self.pre.V + 3) / 8)) )- self.beta * g
    dg = self.alpha * TT * (1 - g)  - self.beta * g
    return dg

  # for more details of how to run a simulation please see the tutorials in "Dynamics Simulation"


class HH(bp.dyn.NeuGroup):
    def __init__(self, size,ENa=120., EK=-12., EL=10.6, C=1.0, gNa=120.,
               gK=36., gL=0.3, V_th=0., method='exp_auto'):
        
        super(HH, self).__init__(size=size)

        # initialize parameters
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.C = C
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.V_th = V_th


        # initialize variables

        self.V = bm.Variable(bm.ones(size) * -65.)
        self.h = bm.Variable(bm.ones(size) * 0.07 * bm.exp(-(self.V[0]) / 20) / (
                    0.07 * bm.exp(-(self.V[0]) / 20) + (1 / (bm.exp((30 - self.V[0]) / 10) + 1))))
        self.n = bm.Variable(bm.ones(size) * (0.01 * (10 - self.V[0]) / (bm.exp((10 - self.V[0]) / 10) - 1)) / (
                    0.01 * (10 - self.V[0]) / (bm.exp((10 - self.V[0]) / 10) - 1) + 0.125 * bm.exp(-(self.V[0]) / 80)))
        self.m = bm.Variable(bm.ones(size) * (0.1 * (25 - self.V[0]) / (bm.exp((25 - self.V[0]) / 10) - 1)) / (
                    0.1 * (25 - self.V[0]) / (bm.exp((25 - self.V[0]) / 10) - 1) + 4 * bm.exp(-(self.V[0]) / 18)))
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size)) # this is synaptic current Isyn variable
       
        # integral
        self.integral = bp.odeint(bp.JointEq([self.dV, self.dh, self.dn,self.dm]), method=method)
        self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

    def dh(self, h, t, V):
        alpha = 0.07 * bm.exp(-(V) / 20)
        beta = 1 / (bm.exp((30 - V) / 10) + 1)
        dhdt = alpha * (1 - h) - beta * h
        return dhdt

    def dn(self, n, t, V):
        alpha = 0.01 * (10 - V) / (bm.exp((10 - V) / 10) - 1)
        beta = 0.125 * bm.exp(-(V) / 80)
        dndt = alpha * (1 - n) - beta * n
        return dndt

    def dm(self, m, t, V):
        alpha = 0.1 * (25 - V) / (bm.exp((25 - V) / 10) - 1)
        beta = 4 * bm.exp(-(V) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

    def dV(self, V, t,  h,n,m, Isyn):

        INa = self.gNa * m ** 3 * h * (V - self.ENa)  #fast sodium(na) current
        IK = self.gK * n ** 4 * (V - self.EK) #fast potassium(k) current
        IL = self.gL * (V - self.EL) #fast leak current
        dVdt = (- INa - IK - IL + Isyn) / self.C
        return dVdt


    def update(self): # ,input=None):
        tdi = bp.share.get_shargs()
        _t=tdi.t
        _dt=tdi.dt
        #input = 0 # we can put time varying input
        # compute V, m, h, n
        V, h, n, m = self.integral(self.V, self.h, self.n, self.m, tdi.t, self.input, tdi.dt)

        # update the spiking state and the last spiking time
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

        # update V
        self.V.value = V
        self.n.value = n
        self.h.value = h
        self.m.value = m
        self.input[:]= 0 #[:] = input # this is synaptic current

num =100
neu = HH(num)

def show_syn_model(model):
  pre =  neu# bp.neurons.HH(10, V_initializer=bp.init.Constant(-70.)) #bp.neurons.LIF(1, V_rest=-60., V_reset=-60., V_th=-40.)
  post = neu #bp.neurons.HH(10, V_initializer=bp.init.Constant(-70.))  #bp.neurons.LIF(1, V_rest=-60., V_reset=-60., V_th=-40.)

  syn = model(pre, post, conn=bp.conn.All2All(include_self=False)) #One2One())
  #syn.g_max= 0.1/100
  # print('pre_spike=', pre.spike)
  # print('********')
  # print('spike time=', syn.spike_arrival_time)
  #syn.g = bm.Variable(bm.ones((100,100)))
  net = bp.Network(pre=pre, post=post, syn=syn)


  runner = bp.DSRunner(net,
                       monitors=['pre.V', 'post.V', 'syn.g','syn.spike_arrival_time','pre.spike','post.spike'],
                       inputs=['pre.input', bm.random.uniform(low=9.0, high=10.0, size=num)])
  runner.run(500)
  fig, gs = bp.visualize.get_figure(2, 1, 3, 12)
  ax =fig.add_subplot(gs[0, 0])
  
  bp.visualize.raster_plot(runner.mon.ts,runner.mon['post.spike'])
  # bp.visualize.raster_plot(runner.mon.ts, runner.mon['pre.spike'])
  plt.legend(loc='upper right')
  plt.title('Raster plot')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
 
  ax  =fig.add_subplot(gs[1, 0])
  #bp.visualize.line_plot(runner.mon.ts, runner.mon['post.V'], legend='post.V', show=True)
  plt.plot(runner.mon.ts, runner.mon['pre.V'][:,0], label='V')
  # plt.plot(runner.mon.ts, runner.mon['post.V'][:, 3], label='V')
  plt.legend(loc='upper right')
  plt.xlabel(r'$t$ (ms)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # plt.savefig('../img/DeltaSynapse.pdf', transparent=True, dpi=500)
  plt.show()


class AMPAConnMat(BaseAMPASyn):
  def __init__(self, *args, **kwargs):
    super(AMPAConnMat, self).__init__(*args, **kwargs)
  
    # connection matrix


    # synapse gating variable
    # -------
    # NOTE: Here the synapse shape is (num_pre, num_post),
    #       in contrast to the ExpConnMat
    #self.g = bm.Variable(bm.zeros((self.pre.num, self.post.num)))
    self.g = bm.Variable(bm.zeros(self.pre.num)) #,self.post.num)))
    #self.derivative(g=0.1,t=0.1, TT=1 / (1 + bm.exp(-(pre.V - pre.V_th) / 2))

  def update(self, tdi, x=None):
    _t, _dt = tdi.t, tdi.dt
    # pull the delayed pre spikes for computation
    delayed_spike = self.pre_spike(self.delay_step)
    # push the latest pre spikes into the bottom
    self.pre_spike.update(self.pre.spike)
    # # get the time of pre spikes arrive at the post synapse
    # self.spike_arrival_time.value = bm.where(delayed_spike, _t, self.spike_arrival_time)

    # get the neurotransmitter concentration at the current time
    # TT =((_t - self.pre.t_last_spike) < self.T_duration) * self.T #
    # #integrate the synapse state
    # #TT =  1/(1 + bm.exp(-(self.pre.V + 3) / 8))  #1 / (1 + bm.exp(-(self.pre.V - neu.V_th) / 2)) # it gives some scalar value
    # #print('TT=',TT)
    # TT = TT.reshape((-1, 1)) * self.conn_mat   # NOTE: only keep the concentrations
                                                   # on the invalid connections
    self.g.value = self.integral(self.g, _t, dt=_dt)
    # update synapse states according to the pre spikes
    post_sps = bm.dot(delayed_spike.astype(float), self.conn_mat)
    self.g += post_sps

    # get the post-synaptic current
    # g_post = bm.sum(self.e_ij * self.g,axis=0)
    self.post.input += self.g_max*self.g * (self.E - self.post.V)/self.Omega
    #self.post.input -= bm.sum(self.e_ij * self.s , axis=0)* (self.V_r - self.post.V) / self.Omega 
    

show_syn_model(AMPAConnMat)

