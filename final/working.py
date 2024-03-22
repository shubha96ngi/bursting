# this is working example from brainpy documentation 

import brainpy as bp
bp.profile.set(jit=True,
               device='cpu',
               dt=0.04,
               numerical_method='exponential')


V_th = 0.  # the spike threshold
C = 1.0  # the membrane capacitance
gLeak = 0.1  # the conductance of leaky channel
ELeak = -65  # the reversal potential of the leaky channel
gNa = 35.  # the conductance of sodium channel
ENa = 55.  # the reversal potential of sodium
gK = 9.  # the conductance of potassium channel
EK = -90.  # the reversal potential of potassium
phi = 5.0  # the temperature depdend

@bp.integrate
def int_h(h, t, V):
    alpha = 0.07 * np.exp(-(V+58) / 20)
    beta = 1 / (np.exp(-0.1 * (V + 28)) + 1)
    dhdt = alpha * (1 - h) - beta * h
    return phi*dhdt

@bp.integrate
def int_n(n, t, V):
    alpha = -0.01 * (V + 34) / (np.exp(-0.1 * (V + 34)) - 1)
    beta = 0.125 * np.exp(-(V+44) / 80)
    dndt = alpha * (1 - n) - beta * n
    return  phi*dndt

@bp.integrate
def int_V(V, t, h, n, Isyn):
    m_alpha = -0.1 * (V + 35) / (np.exp(-0.1 * (V + 35)) - 1)
    m_beta = 4 * np.exp(-(V + 60) / 18)
    m = m_alpha / (m_alpha + m_beta)
    INa = gNa * m ** 3 * h * (V - ENa)
    IK = gK * n ** 4 * (V - EK)
    IL = gLeak * (V - ELeak)
    dvdt = (- INa - IK - IL + Isyn) / C
    return dvdt

HH_ST = bp.types.NeuState({
    'V': -55.,  # membrane potential, default initial value is -55.
    'h': 0.,  # h channel, default initial value is 0.
    'n': 0.,  # n channel, default initial value is 0.
    'spike': 0.,  # neuron spike state, default initial value is 0.,
                  # if neuron emits a spike, it will be 1.
    'input': 0.  # neuron synaptic input, default initial value is 0.
})

def update(ST, _t):
    h = int_h(ST['h'], _t, ST['V'])
    n = int_n(ST['n'], _t, ST['V'])
    V = int_V(ST['V'], _t, ST['h'], ST['n'], ST['input'])
    sp = np.logical_and(ST['V'] < V_th, V >= V_th)
    ST['spike'] = sp
    ST['V'] = V
    ST['h'] = h
    ST['n'] = n
    ST['input'] = 0.

HH = bp.NeuType(ST=HH_ST, name='HH_neuron', steps=update)

g_max = 0.1  # the maximal synaptic conductance
E = -75.  # the reversal potential
alpha = 12.  # the channel opening rate
beta = 0.1  # the channel closing rate

ST = bp.types.SynState(['s'])

@bp.integrate
def int_s(s, t, TT):
    return alpha * TT * (1 - s) - beta * s
def update(ST, _t, pre):
    T =  1 /(1 + np.exp(-(pre['V'] - V_th) / 2))
    #print('T=', T)
    s = int_s(ST['s'], _t, T)
    ST['s'] = s

@bp.delayed
def output(ST, post):
    #print('shape=', ST['s'])
    post['input'] -= g_max * ST['s'] * (post['V'] - E)

requires = dict(
    pre=bp.types.NeuState(['V']),
    post=bp.types.NeuState(['V', 'input']),
)

GABAa = bp.SynType(ST=ST,
                   name='GABAa',
                   steps=(update, output),
                   requires=requires,
                   mode='scalar')

num = 100
neu = bp.NeuGroup(HH, geometry=num, monitors=['spike', 'V'])

syn = bp.SynConn(model=GABAa,
                 pre_group=neu,
                 post_group=neu,
                 conn=bp.connect.All2All(include_self=True),
                 delay=0.5,
                 monitors=['s'])

#print('s =', syn.mon.s.shape)

import numpy as np

v_init = -70. + np.random.random(num) * 20
h_alpha = 0.07 * np.exp(-(v_init + 58) / 20)
h_beta = 1 / (np.exp(-0.1 * (v_init + 28)) + 1)
h_init = h_alpha / (h_alpha + h_beta)
n_alpha = -0.01 * (v_init + 34) / (np.exp(-0.1 * (v_init + 34)) - 1)
n_beta = 0.125 * np.exp(-(v_init + 44) / 80)
n_init = n_alpha / (n_alpha + n_beta)
neu.ST['V'] = v_init
neu.ST['h'] = h_init
neu.ST['n'] = n_init

syn.pars['g_max'] = 0.1 / num
# print('alpha=', syn.pars['alpha'], syn.pars['beta']) #, syn.beta)
net = bp.Network(neu, syn)
net.run(duration=500., inputs=[neu, 'ST.input', 1.2], report=False)

import matplotlib.pyplot as plt
ts = net.ts
fig, gs = bp.visualize.get_figure(2, 1, 3, 12)
fig.add_subplot(gs[0, 0])
plt.plot(ts, neu.mon.V[:, 0])
plt.ylabel('Membrane potential (N0)')
plt.xlim(net.t_start - 0.1, net.t_end + 0.1)
fig.add_subplot(gs[1, 0])
index, time = bp.measure.raster_plot(neu.mon.spike, net.ts)
plt.plot(time, index, '.')
plt.xlim(net.t_start - 0.1, net.t_end + 0.1)
plt.xlabel('Time (ms)')
plt.ylabel('Raster plot')
plt.show()
