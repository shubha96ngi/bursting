import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
bp.math.set_dt(0.05)
# working code1 
'''
class HH(bp.NeuGroup):
  def __init__(self, size, ENa=55., EK=-90., EL=-65, C=1.0, gNa=35.,
               gK=9., gL=0.1, V_th=20., phi=5, method='exp_auto'):
    super(HH, self).__init__(size=size)

    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.C = C
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.V_th = V_th
    self.phi = phi

    # variables
    self.V = bm.Variable(bm.ones(size) * -65.)
    self.h = bm.Variable(bm.ones(size) *0.6)
    self.n = bm.Variable(bm.ones(size) * 0.32)
    self.m = bm.Variable(bm.ones(size) * 0)
    self.spike = bm.Variable(bm.zeros(size, dtype=bool))
    self.input = bm.Variable(bm.zeros(size))
    self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

    # integral
    self.integral = bp.odeint(bp.JointEq([self.dV, self.dh, self.dn]), method=method)

  def dh(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 58) / 20)
    beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1)
    dhdt = alpha * (1 - h) - beta * h
    return self.phi * dhdt

  def dn(self, n, t, V):
    alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
    beta = 0.125 * bm.exp(-(V + 44) / 80)
    dndt = alpha * (1 - n) - beta * n
    return self.phi * dndt

  # def m(self, V):
  #   alpha =  -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
  #   beta = 4 * bm.exp(-(V + 60) / 18)
  #   mt = alpha / (alpha + beta) #alpha * (1 - m) - beta * m
  #   return  mt

  def dV(self, V, t, h, n, Iext):
    m_alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
    m_beta = 4 * bm.exp(-(V + 60) / 18)
    m = m_alpha / (m_alpha + m_beta)
    INa = self.gNa * m ** 3 * h * (V - self.ENa)
    IK = self.gK * n ** 4 * (V - self.EK)
    IL = self.gL * (V - self.EL)
    dVdt = (- INa - IK - IL + Iext) / self.C

    return dVdt

  def update(self, tdi):
    V, h, n = self.integral(self.V, self.h, self.n, tdi.t, self.input, tdi.dt)
    m =  -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1) / (-0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1) +4 * bm.exp(-(V + 60) / 18) )
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.t_last_spike.value = bm.where(self.spike, tdi.t, self.t_last_spike)
    self.V.value = V
    self.h.value = h
    self.n.value = n
    self.m.value = m
    self.input[:] = 0




num = 100
neu.V[:] = -70.  bm.random.normal(size=num) * 20

syn = bp.synapses.GABAa(pre=neu, post=neu, conn=bp.connect.All2All(include_self=False),stop_spike_gradient=False)
syn.g_max = 0.1/ num
net = bp.Network(neu=neu, syn=syn)
runner = bp.DSRunner(net, monitors=['neu.spike', 'neu.V'], inputs=['neu.input',1.2]) # bm.random.normal(1,0.02,num)])
# runner = bp.DSRunner(neu, monitors=['V','spike'], inputs=['input',1,'fix','=']) # bm.random.normal(1,0.02,num)])

runner.run(duration=500.)

fig, gs = bp.visualize.get_figure(2, 1, 3, 8)

fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon['neu.V'][:,0], ylabel='Membrane potential (N0)')
bp.visualize.line_plot(runner.mon.ts, runner.mon['neu.V'][:,1])

# bp.visualize.line_plot(runner.mon.ts, runner.mon.V[:,0], ylabel='Membrane potential (N0)')
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V[:,1])

fig.add_subplot(gs[1, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['neu.spike'], show=True)
plt.show()
'''
########code2 
# I am expecting same output 
# all the params are same still I dont know whats wrong 
class A(bp.dyn.WangBuzsakiHH):
    def m_inf(self,V):
        alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
        print('m_inf*****')
        beta = 4 * bm.exp(-(V + 60) / 18)
        return alpha / (alpha + beta)
    def dn(self, n, t, V):
        alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
        print('dn_alpha*****')
        beta = 0.125 * bm.exp(-(V + 44) / 80)
        dndt = alpha * (1 - n) - beta * n
        return self.phi * dndt

    def dV(self, V, t, h, n, I):
        print('I ki value =', I )
        INa = self.gNa * self.m_inf(V) ** 3 * h * (V - self.ENa)
        print('I ki value =', INa)
        IK = self.gK * n ** 4 * (V - self.EK)
        IL = self.gL * (V - self.EL)
        dVdt = (- INa - IK - IL + I) / self.C
        return dVdt


class GABAa(bp.Projection):
    def __init__(self, pre, post, delay, prob, g_max, E=-75.):
        super().__init__()
        self.proj = bp.dyn.ProjAlignPreMg2(
          pre=pre,
          delay=delay,
          syn=bp.dyn.GABAa.desc(pre.num, alpha=0.53, beta=0.18, T=1.0, T_dur=1.0),
          comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
          out=bp.dyn.COBA(E=E),
          post=post,
        )
# a = bp.dyn.WangBuzsakiHH(10)
class SimpleNet(bp.DynSysGroup):
    def __init__(self, E=-75.):
        super().__init__()
        self.pre = A(100) #bp.neurons.WangBuzsakiModel(100) #bp.dyn.HH(100) #bp.dyn.WangBuzsakiHH(100)  # Changed to HH
        self.pre.V = -70. + bm.random.normal(size=100) * 20 #bm.ones(10)*-70
        self.post =A(100) #bp.neurons.WangBuzsakiModel(100) #bp.dyn.HH(100) #WangBuzsakiHH(100)  # Changed to HH
        self.post.V = -70. + bm.random.normal(size=100) * 20 #bm.ones(10)*-70

        self.syn = GABAa(self.pre,self.post, delay=0, prob=1., g_max=0.1/100, E=-75.)
        # super().__init__()

    def update(self, I_pre,I_post):
        self.pre(I_pre)
        self.syn()
        self.post(I_post)

        conductance = self.syn.proj.refs['syn'].g
        current = self.post.sum_inputs(self.post.V)
        return conductance, current, self.post.V,self.post.spike.value

duration = 40000
indices = np.arange(int(duration/bm.get_dt())).reshape(-1,100)
I_pre = np.ones(indices.shape) * 1.2
I_post = np.ones(indices.shape) * 1.2
conductances, currents, potentials,spks = bm.for_loop(SimpleNet(E=-75.).step_run, (indices,I_pre, I_post), progress_bar=True)

print('spikes.shape=', spks.shape)
ts = indices * bm.get_dt()
fig, gs = bp.visualize.get_figure(1, 2, 3.5, 8)
fig.add_subplot(gs[0, 0])
# plt.title('Syn conductance')
# plt.plot(ts, conductances[:,0])
# plt.plot(ts, conductances[:,5])
# fig.add_subplot(gs[0, 1])
# plt.plot(ts, currents[:,0])
# plt.title('Syn current')
# fig.add_subplot(gs[0, 2])
plt.plot(ts, potentials[:,0])
plt.plot(ts, potentials[:,10])
fig.add_subplot(gs[0, 1])
bp.visualize.raster_plot(ts, spks, show=True)
plt.show()

