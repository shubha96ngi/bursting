import brainpy as bp
import brainpy.math as bm
import numpy as np
import scipy.io as sio
bp.math.set_dt(0.01) #0.01)


# layer 1 (MSN neurons)
class MSN(bp.dyn.NeuGroup):
  # def __init__(self, size,ENa=120., EK=-12., EL=10.6, C=1.0, gNa=120.,
  #              gK=36., gL=0.3, V_th=-55., method='exp_auto',**kwargs):
    def __init__(self, size,ENa=55., EK=-90., EL=-65, C=1.0, gNa=35.,
               gK=9., gL=0.1, V_th=0., method='exp_auto',**kwargs):
        # providing the group "size" information
        # V_th was 20 earlier
        super(MSN, self).__init__(size=size, **kwargs)

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

        #self.V = bm.Variable(np.array([10.0, 20 , -60, -70, 30, -100, -65, 70, 80, 40])) #y[0:10]) #
        self.V  = bm.Variable(bm.ones(size)* -55.)
        self.m = bm.Variable(bm.zeros(size)) # * 0.004)
        self.n = bm.Variable(bm.zeros(size)) # * 0.8)
        self.h = bm.Variable(bm.zeros(size))# * 0.05)
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size)) # this is synaptic current Isyn variable
        self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

        # integral
        self.integral = bp.odeint(bp.JointEq([self.dV, self.dm, self.dn, self.dh]), method=method)



    def dm(self, m, t, V):
        alpha =  -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1) #0.1*(25-V) / (bm.exp(2.5 - 0.1*V)-1) #0.1 * (-V + 25) / (1 - bm.exp(-(V -25) / 10))
        beta = 4 * bm.exp(-(V + 60) / 18) #4.0*bm.exp(-V/18.0) #4.0 * bm.exp(-V / 18)
        dmdt = alpha / (alpha + beta)  #alpha * (1 - m) - beta * m
        return 5*dmdt

    def dh(self, h, t, V):
        alpha = 0.07 * bm.exp(-(V + 58) / 20) #0.07*bm.exp(-V/20.0) #0.07 * bm.exp(-V / 20.)
        beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1) #1/(1+bm.exp(3.0-0.1*V)) #1 / (1 + bm.exp(-(V - 30) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return 5*dhdt

    def dn(self, n, t, V):
        alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1) # 0.01*(10.0-V)/ (bm.exp(1.0-0.1*V )-1) #0.01 * (V - 10) / (1 - bm.exp(-(V -10) / 10))
        beta =  0.125 * bm.exp(-(V + 44) / 80) #0.125*bm.exp(-V/80.0) #0.125 * bm.exp(-V / 80)
        dndt = alpha * (1 - n) - beta * n
        return dndt

    def dV(self, V, t, m, n, h, Isyn):

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
        V, m,n,h = self.integral(self.V, self.m, self.n, self.h, _t, input, _dt) # this order of variables is correct

        # update the spiking state and the last spiking time
        self.spike.value = bm.logical_and(self.V < self.V_th, self.V >= self.V_th)
        self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

        # update V
        self.V.value = V
        self.m.value = m
        self.n.value = n
        self.h.value = h
        self.input[:] = input # this is synaptic current

# now add the synapse
class GABAa(bp.synapses.TwoEndConn):
    # def __init__(self, pre, post, conn, delay=8., g_max=0.25, E= 0.,
    #              alpha=0.3, beta=12, T=1.0, T_duration=1.0, method='exp_auto'):
    def __init__(self, pre, post, conn, delay=0.5, g_max=0.1, E=-75,
                 alpha=12, beta=0.1, T=1.0, T_duration=1.0, method='exp_auto'):
        super(GABAa, self).__init__(pre=pre, post=post, conn=conn)
        self.check_pre_attrs('spike')
        self.check_post_attrs('t_last_spike', 'input',
                              'V')  # check whether post synaptic neuron has these important params

        # parameters
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration
        self.delay = delay

        # connections
        self.conn_mat = self.conn.requires('conn_mat')
        self.size = self.conn_mat.shape

        # variables
        self.s = bm.Variable(bm.zeros(self.size))

        # function
        ds = lambda s, t, TT: self.alpha * TT * (1 - s) - self.beta * s
        self.integral = bp.odeint(ds, method=method)

    def update(self):
        tdi = bp.share.get_shargs()
        _t = tdi.t
        _dt = tdi.dt
        # TT = ((_t - self.pre.t_last_spike) < self.T_duration) * self.T
        TT = 1 / (1 + bm.exp(-(self.pre.V - self.V_th) / 2))
        # self.T -> transmitter concentration when synapse is triggered by a pre-synaptic spike. default 1
        # T_duration transmitter concetration duration time after being triggered #
        # means for how long it will be there
        # TT = TT.reshape((-1, 1)) * self.conn_mat # they are projecting over the pre-post neurons
        self.s.value = self.integral(self.s, _t, TT, _dt)
        self.post.input -= self.g_max * bm.sum(self.s, axis=0) * (self.post.V - self.E)
        # bm.sum(dot(self.g_max, self.s), axis=0)
        # self.post.input -= self.g_max * bm.sum(self.s, axis=0) * (self.post.V - self.E)

num = 10 # num neurons
neu = MSN(num)
neu.V[:] = -70. + bm.random.normal(size=num) * 20

# # this is connection with in first layer
syn = GABAa(pre=neu, post=neu, conn=bp.connect.All2All(include_self=False))

net = bp.Network(neu=neu, syn=syn)
# inputs = np.ones(int(50./bm.get_dt())) *5.
runner = bp.DSRunner(neu, monitors=['V'], inputs=['input', 1.2])
# #runner.run(inputs= inputs) #duration=5000.)
runner.run(duration=50.)
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=True)
