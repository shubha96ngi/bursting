class A(bp.dyn.WangBuzsakiHH):
    def __init__(self, input, size):
        super().__init__(size=size)
        self.input = input

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

    def update(self, x=None):

        # x = 0. if x is None else self.input
        x = self.input
        print('x='*100, x)
        x = self.sum_current_inputs(self.V.value, init=x)
        print('x after=' * 100, x)
        return super().update(x)  # which one is updating WanBizsakiHH or  WanBizsakiHHLTC?


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
        # self.out = bp.dyn.COBA(E=E)  #TypeError: COBA.update() missing 1 required positional argument: 'potential'
    
     # generally they dont use self.out but is it assumed that output od self.proj will be returned output of GABAa
# what if i want to include some other function like self.out or is it possible to access value of out in self.proj in 
#GABAa 

inputs = 1.2 #np.arange(int(150/bm.get_dt()))
print('inputs=', inputs)
neu1 = A(inputs, 10)#bp.neurons.HH(1)
neu2 = A(inputs,10) #bp.neurons.HH(1)
syn1 = GABAa(neu1, neu2, None, 1.,0.1)
net = bp.DynSysGroup(pre=neu1, syn=syn1, post=neu2)

runner = bp.DSRunner(net,
                     # inputs=(neu1.input, 6.),
                     monitors={'pre.V': neu1.V, 'post.V': neu2.V, 'syn.g': syn1.proj.refs['syn'].g})
runner.run(150) #inputs=inputs)

import matplotlib.pyplot as plt

fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
plt.plot(runner.mon.ts, runner.mon['pre.V'][:,0], label='pre-V')
plt.plot(runner.mon.ts, runner.mon['post.V'][:,0], label='post-V')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(runner.mon.ts, runner.mon['syn.g'][:,0], label='g')
plt.legend()
plt.show()
