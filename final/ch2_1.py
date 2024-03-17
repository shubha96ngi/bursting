import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

bp.math.setā_dt(0.04)
class BaseAMPASyn(bp.SynConn):
  def __init__(self, pre, post, conn, delay=0.5, g_max=0.1, E=-75., alpha=12,
               beta=0.1, T=0.5, T_duration=0.5, method='exponential_euler'):
    super(BaseAMPASyn, self).__init__(pre=pre, post=post, conn=conn)

    # check whether the pre group has the needed attribute: "spike"
    self.check_pre_attrs('spike')

    # check whether the post group has the needed attribute: "input" and "V"
    self.check_post_attrs('input', 'V')

    # parameters
    self.delay = delay
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration

    # use "LengthDelay" to store the spikes of the pre-synaptic neuron group
    self.delay_step = int(delay/bm.get_dt()) #5 se 6
    self.pre_spike = bm.LengthDelay(pre.spike, self.delay_step) # 6 se 6x10 where pre.spike = 10 values

    # store the arrival time of the pre-synaptic spikes
    self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)  # negative time

    # integral function
    self.integral = bp.odeint(self.derivative, method=method)

  def derivative(self, g, t, TT):
    dg = self.alpha * TT * (1 - g) - self.beta * g
    return dg

  # for more details of how to run a simulation please see the tutorials in "Dynamics Simulation"


class HH(bp.dyn.NeuGroup):
    def __init__(self, size,ENa=55., EK=-90., EL=-65, C=1.0, gNa=35.,
               gK=9., gL=0.1, V_th=0., method='exponential_euler',**kwargs):
        
        super(HH, self).__init__(size=size, **kwargs)

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

        self.V = bm.Variable(-70. + bm.random.normal(size=100) * 20) #bm.Variable(bm.ones(size)* -55.)
        n_alpha = -0.01 * (self.V + 34) / (bm.exp(-0.1 * (self.V + 34)) - 1)
        n_beta = 0.125 * bm.exp(-(self.V + 44) / 80)
        self.n = bm.Variable(n_alpha / (n_alpha + n_beta)) #self.n = bm.Variable(bm.zeros(size)) # * 0.8)
        h_alpha = 0.07 * bm.exp(-(self.V + 58) / 20)
        h_beta = 1 / (bm.exp(-0.1 * (self.V + 28)) + 1)
        self.h = bm.Variable(h_alpha / (h_alpha + h_beta)) #(bm.Variable(bm.zeros(size)))# * 0.05)
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size)) # this is synaptic current Isyn variable
       
        # integral
        self.integral = bp.odeint(bp.JointEq([self.dV, self.dn, self.dh]), method=method)

    def dh(self, h, t, V):
        alpha = 0.07 * bm.exp(-(V + 58) / 20) #0.07*bm.exp(-V/20.0) #0.07 * bm.exp(-V / 20.)
        beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1) #1/(1+bm.exp(3.0-0.1*V)) #1 / (1 + bm.exp(-(V - 30) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return 5.0*dhdt

    def dn(self, n, t, V):
        alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1) # 0.01*(10.0-V)/ (bm.exp(1.0-0.1*V )-1) #0.01 * (V - 10) / (1 - bm.exp(-(V -10) / 10))
        beta =  0.125 * bm.exp(-(V + 44) / 80) #0.125*bm.exp(-V/80.0) #0.125 * bm.exp(-V / 80)
        dndt = alpha * (1 - n) - beta * n
        return 5.0*dndt

    def dV(self, V, t, n, h, Isyn):
        m_alpha = -0.1 * (V + 40) / (bm.exp(-0.1 * (V + 40)) - 1)  # 0.1*(25-V) / (bm.exp(2.5 - 0.1*V)-1) #0.1 * (-V + 25) / (1 - bm.exp(-(V -25) / 10))
        m_beta = 4 * bm.exp(-(V + 65) / 18)  # 4.0*bm.exp(-V/18.0) #4.0 * bm.exp(-V / 18)
        m = m_alpha / (m_alpha + m_beta)  # alpha * (1 - m) - beta * m

        INa = self.gNa * m ** 3 * h * (V - self.ENa)  #fast sodium(na) current
        IK = self.gK * n ** 4 * (V - self.EK) #fast potassium(k) current
        IL = self.gL * (V - self.EL) #fast leak current
        dVdt = (- INa - IK - IL + Isyn) / self.C
        return dVdt


    def update(self ,input=None):
        tdi = bp.share.get_shargs()
        _t=tdi.t
        _dt=tdi.dt
        input = 0 # we can put time varying input
        # compute V, m, h, n
        V,n,h = self.integral(self.V, self.n, self.h, _t, input, _dt) # this order of variables is correct

        # update the spiking state and the last spiking time
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        #self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

        # update V
        self.V.value = V
        self.n.value = n
        self.h.value = h
        self.input[:]= 0 #[:] = input # this is synaptic current

num =100
neu = HH(num)

def show_syn_model(model):
  pre =  neu# bp.neurons.HH(10, V_initializer=bp.init.Constant(-70.)) #bp.neurons.LIF(1, V_rest=-60., V_reset=-60., V_th=-40.)
  post = neu #bp.neurons.HH(10, V_initializer=bp.init.Constant(-70.))  #bp.neurons.LIF(1, V_rest=-60., V_reset=-60., V_th=-40.)

  syn = model(pre, post, conn=bp.conn.All2All(include_self=False)) #One2One())
  syn.g_max= 0.1 #/100
  print('pre_spike=', pre.spike)
  print('********')
  print('spike time=', syn.spike_arrival_time)
  #syn.g = bm.Variable(bm.ones((100,100)))
  net = bp.Network(pre=pre, post=post, syn=syn)

  runner = bp.DSRunner(net,
                       monitors=['pre.V', 'post.V', 'syn.g','syn.spike_arrival_time','pre.spike','post.spike'],
                       inputs=['pre.input', 1.2])
  runner.run(500)
  fig, gs = bp.visualize.get_figure(2, 1, 3, 12)
  ax =fig.add_subplot(gs[0, 0])
  
  bp.visualize.raster_plot(runner.mon.ts,runner.mon['post.V'])
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['pre.V'])
  plt.legend(loc='upper right')
  plt.title('Raster plot')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
 
  ax  =fig.add_subplot(gs[1, 0])
  #bp.visualize.line_plot(runner.mon.ts, runner.mon['post.V'], legend='post.V', show=True)
  plt.plot(runner.mon.ts, runner.mon['pre.V'][:,0], label='V')
  plt.plot(runner.mon.ts, runner.mon['post.V'][:, 3], label='V')
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
    self.conn_mat = self.conn.require('conn_mat')  #.astype(float)

    # synapse gating variable
    # -------
    # NOTE: Here the synapse shape is (num_pre, num_post),
    #       in contrast to the ExpConnMat
    #self.g = bm.Variable(bm.zeros((self.pre.num, self.post.num)))
    self.g = bm.Variable(bm.zeros(self.post.num))
    #self.derivative(g=0.1,t=0.1, TT=1 / (1 + bm.exp(-(pre.V - pre.V_th) / 2))

  def update(self, tdi, x=None):
    _t, _dt = tdi.t, tdi.dt
    # pull the delayed pre spikes for computation
    delayed_spike = self.pre_spike(self.delay_step)
    # push the latest pre spikes into the bottom
    self.pre_spike.update(self.pre.spike)
    # get the time of pre spikes arrive at the post synapse
    self.spike_arrival_time.value = bm.where(delayed_spike, _t, self.spike_arrival_time)

    # get the neurotransmitter concentration at the current time
    TT = 1 / (1 + bm.exp(-(self.pre.V - neu.V_th) / 2))#
    #integrate the synapse state
    # TT = TT.reshape((-1, 1)) * self.conn_mat  # NOTE: only keep the concentrations
                                                   # on the invalid connections
    self.g.value = self.integral(self.g, _t, TT, dt=_dt)
    # get the post-synaptic current
    #g_post = self.g.sum(axis=0)
    self.post.input += self.g_max *self.g * (self.E - self.post.V)

show_syn_model(AMPAConnMat)

