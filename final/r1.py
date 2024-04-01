import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt 
import numpy as np 

'''
class HH(bp.dyn.NeuGroup):
    def __init__(self, size,ENa=55., EK=-90., EL=-65, C=1.0, gNa=35.,
               gK=9., gL=0.1, V_th=0., method='exp_auto',**kwargs):

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



        self.V = bm.Variable(bm.ones(size) * -65.)
        self.h = bm.Variable(bm.ones(size) * 0.6)
        self.n = bm.Variable(bm.ones(size) * 0.32)
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size))  # this is synaptic current Isyn variable

        # initialize variables


        # integral
        self.integral = bp.odeint(bp.JointEq([self.dV, self.dn, self.dh]), method=method)
        self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

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
        m_alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)  # 0.1*(25-V) / (bm.exp(2.5 - 0.1*V)-1) #0.1 * (-V + 25) / (1 - bm.exp(-(V -25) / 10))
        m_beta = 4 * bm.exp(-(V + 60) / 18)  # 4.0*bm.exp(-V/18.0) #4.0 * bm.exp(-V / 18)
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
        #input = 0 # we can put time varying input
        # compute V, m, h, n
        V,n,h = self.integral(self.V, self.n, self.h, _t, self.input, _dt) # this order of variables is correct

        # update the spiking state and the last spiking time
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        #self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

        # update V
        self.V.value = V
        self.n.value = n
        self.h.value = h
        self.input[:]= self.input[:] #0 #[:] = input # this is synaptic current

num =100
neu = HH(num)

neu.V =   -70. + bm.random.random(num) * 20 

'''
class GABAa(bp.Projection):
    def __init__(self, pre, post, delay, prob, g_max, E=0.):
        super().__init__()
        self.proj = bp.dyn.ProjAlignPreMg2(
          pre=pre,
          delay=delay,
          syn=bp.dyn.GABAa.desc(pre.num, alpha=0.53, beta=0.18, T=1.0, T_dur=1.0),
          comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
          out=bp.dyn.COBA(E=E),
          post=post,
        )

class SimpleNet(bp.DynSysGroup):
    def __init__(self, E=0.):
        super().__init__()
    
        size = 100
        a = bp.dyn.HH(size)
        a._V_initializer = bm.Variable(bm.ones(size) * -65)
        # V = a._V_initializer
        a._h_initializer = bm.Variable(bm.ones(size) * 0.6)
        a._n_initializer = bm.Variable(bm.ones(size) * 0.32)
        a._m_initializer = bm.Variable(bm.ones(size) * 0.5)
        # m =a._m_initializer

        a.n_beta = lambda V: 0.125 * bm.exp(-(V + 44) / 80)
        a.n_alpha = lambda V: -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
        a.dn = lambda V: 5*(a.n_alpha(V) * (1 - n) - a.n_beta(V) * n)
        a.m_alpha = lambda V: 1. / bm.exprel(-(V + 40) / 10)
        a.m_beta = lambda V: 4.0 * bm.exp(-(V + 65) / 18)
        a.m_inf = lambda V: a.m_alpha(V) / (a.m_alpha(V) + a.m_beta(V))
        # a.dm = lambda a, V: a.m_alpha(V) * (1 - m) - a.m_beta(V) * m
        a.h_alpha = lambda V: 0.07 * bm.exp(-(V + 58) / 20)
        a.h_beta = lambda V: 1 / (bm.exp(-0.1 * (V + 28)) + 1)
        a.dh = lambda  V: 5*(a.h_alpha(V) * (1 - h) - a.h_beta(V) * h)

        a.EK = -90
        a.EL = -65
        a.ENa = 55
        a.gL = 0.1
        a.gNa = 35
        a.gK = 9
        a.V_th = 0
    
        a._V_initializer = -70. + bm.random.random(100) * 20
        # for error check this link  --https://brainpy.readthedocs.io/en/latest/_modules/brainpy/_src/dyn/neurons/hh.html#
        '''
        def dV(self, V, t, m, h, n, I):
            I = self.sum_current_inputs(V, init=I)
            I_Na = (self.gNa * m * m * m * h) * (V - self.ENa)
            n2 = n * n
            I_K = (self.gK * n2 * n2) * (V - self.EK)
            I_leak = self.gL * (V - self.EL)
            dVdt = (- I_Na - I_K - I_leak + I) / self.C
            return dVdt
        '''
        # replacing m with m_inf
        # how to write it ? a.dV.I_Na does not make sense 
        a.dV.I_Na = I want to change it from self.gNa * m **3 * h) * (V - self.ENa) to self.gNa * m_inf **3 * h) * (V - self.ENa)
        # another option is something line a.m = a.m_inf # it is also not working 
        # also is it possible to pass a.m_inf instead of a.m.value in self.integral in update function in https://brainpy.readthedocs.io/en/latest/_modules/brainpy/_src/dyn/neurons/hh.html#
        # but I dont know how to access this integral function 
    

        
        
        
        #
       # updating value of input 
        self.pre = bp.dyn.HH(10) #bp.dyn.SpikeTimeGroup(10, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.)) #bp.neurons.HH(10, V_initializer=bp.init.Constant(-70.)) #bp.dyn.SpikeTimeGroup(1, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.))
        self.pre.update(2)
        # input =  3 #np.ones(int(100. / bm.get_dt())) * 20.
        # a.update(x=input)
        self.post = neu #bp.dyn.HH(10, V_initializer=bp.init.Constant(-70.)) #bp.dyn.LifRef(1, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                  #V_initializer=bp.init.Constant(-60.))
        self.syn = GABAa(self.pre, self.post, delay=None, prob=1., g_max=0.1/100, E=0.)

    def update(self):
        self.pre()
        self.syn()
        self.post()

        # monitor the following variables
        conductance = self.syn.proj.refs['syn'].g
        current = self.post.sum_inputs(self.post.V)
        # spikes = self.pre.spike
        return conductance, current, self.post.V #,spikes

    # def return_info(self):
    #     return conductance, current,self.post.V

indices = np.arange(500)  # 100 ms, dt= 0.1 ms
conductances, currents, potentials = bm.for_loop(SimpleNet(E=0.).step_run, indices, progress_bar=True)
ts = indices * bm.get_dt()

#
# print('spikes=', spikes.shape)
# bp.visualize.raster_plot(ts, spikes)
# plt.show()

fig, gs = bp.visualize.get_figure(1, 3, 3.5, 4)
fig.add_subplot(gs[0, 0])
plt.plot(ts, conductances)
#bp.visualize.raster_plot(ts, spikes)
plt.title('Syn conductance')
fig.add_subplot(gs[0, 1])
plt.plot(ts, currents)
plt.title('Syn current')
fig.add_subplot(gs[0, 2])
plt.plot(ts, potentials)
plt.title('Post V')
plt.show()
