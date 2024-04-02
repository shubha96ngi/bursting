import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt



bp.math.set_dt(0.01)
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
neu = HH(num)
neu.V[:] = -70. + bm.random.normal(size=num) * 20

syn = bp.synapses.GABAa(pre=neu, post=neu, conn=bp.connect.All2All(include_self=False),stop_spike_gradient=False)
syn.g_max = 0.2/ num
net = bp.Network(neu=neu, syn=syn)
runner = bp.DSRunner(net, monitors=['neu.spike', 'neu.V'], inputs=['syn.pre.input',1.2]) # bm.random.normal(1,0.02,num)])
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

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# instead of using bp.synapses.GABAa in line 74  I copied this GABAa class from their documentation 
from typing import Union, Dict, Callable, Optional
from brainpy._src.connect import TwoEndConnector
from brainpy._src.dyn import synapses
from brainpy._src.dynold.synapses import _SynSTP, _SynOut, _TwoEndConnAlignPre
from brainpy._src.dynold.synouts import COBA, MgBlock
from brainpy._src.dyn.base import NeuDyn
from brainpy.types import ArrayType


class AMPA(_TwoEndConnAlignPre):

  def __init__(
      self,
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      output: _SynOut = COBA(E=0.),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Callable] = 0.42,
      delay_step: Union[int, ArrayType, Callable] = None,
      alpha: float = 0.98,
      beta: float = 0.18,
      T: float = 0.5,
      T_duration: float = 0.5,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
      stop_spike_gradient: bool = False,
  ):
    # parameters
    self.stop_spike_gradient = stop_spike_gradient
    self.comp_method = comp_method
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration
    if bm.size(alpha) != 1:
      raise ValueError(f'"alpha" must be a scalar or a tensor with size of 1. But we got {alpha}')
    if bm.size(beta) != 1:
      raise ValueError(f'"beta" must be a scalar or a tensor with size of 1. But we got {beta}')
    if bm.size(T) != 1:
      raise ValueError(f'"T" must be a scalar or a tensor with size of 1. But we got {T}')
    if bm.size(T_duration) != 1:
      raise ValueError(f'"T_duration" must be a scalar or a tensor with size of 1. But we got {T_duration}')

    # AMPA
    syn = synapses.AMPA(pre.size, pre.keep_size, mode=mode, alpha=alpha, beta=beta,
                        T=T, T_dur=T_duration, method=method)

    super().__init__(pre=pre,
                     post=post,
                     syn=syn,
                     conn=conn,
                     output=output,
                     stp=stp,
                     comp_method=comp_method,
                     g_max=g_max,
                     delay_step=delay_step,
                     name=name,
                     mode=mode)

    # copy the references
    self.g = syn.g
    self.spike_arrival_time = syn.spike_arrival_time



  def update(self, pre_spike=None):
    return super().update(pre_spike, stop_spike_gradient=self.stop_spike_gradient)





class GABAa(AMPA):

  def __init__(
      self,
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      output: _SynOut = COBA(E=-80.),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'dense',
      g_max: Union[float, ArrayType, Callable] = 0.04,
      delay_step: Union[int, ArrayType, Callable] = None,
      alpha: Union[float, ArrayType] = 0.53,
      beta: Union[float, ArrayType] = 0.18,
      T: Union[float, ArrayType] = 1.,
      T_duration: Union[float, ArrayType] = 1.,
      method: str = 'exp_auto',

    

      # other parameters
      name: str = None,
      mode: bm.Mode = None,
      stop_spike_gradient: bool = False,
  ):
    super().__init__(pre=pre,
                     post=post,
                     conn=conn,
                     output=output,
                     stp=stp,
                     comp_method=comp_method,
                     delay_step=delay_step,
                     g_max=g_max,
                     alpha=alpha,
                     beta=beta,
                     T=T,
                     T_duration=T_duration,
                     method=method,
                     name=name,
                     mode=mode,
                     stop_spike_gradient=stop_spike_gradient, )


#I can use directly syn = GABAa 
# but I am trying to get same result for different HH model 
# some replacement like neu = bp.neurons.HH(num) in line 71  but its not giving similar plot as from neu = HH(num)\
# so I want to access some variables from class GABAa.pre or post GABAa.E or any variable listed 
# but I am not able to as it is not listed 

# can you please suggest how to access variables from GABAa
