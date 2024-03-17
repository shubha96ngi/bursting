import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np


bp.math.set_dt(0.04)

class HH(bp.dyn.NeuGroup):
    def __init__(self, size,ENa=55., EK=-90., EL=-65, C=1.0, gNa=35.,
               gK=9., gL=0.1, V_th=0., method= 'exponential_euler',**kwargs ): # 'exp_auto',**kwargs):
        # providing the group "size" information
        # V_th was 20 earlier
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
        self.V = bm.Variable(-70 + bm.random.normal(size=100) * 20)
        n_alpha = -0.01 * (self.V + 34) / (bm.exp(-0.1 * (self.V + 34)) - 1)
        n_beta = 0.125 * bm.exp(-(self.V + 44) / 80)
        self.n = bm.Variable(n_alpha / (n_alpha + n_beta))
        h_alpha = 0.07 * bm.exp(-(self.V + 58) / 20)
        h_beta = 1 / (bm.exp(-0.1 * (self.V + 28)) + 1)
        self.h = bm.Variable(h_alpha / (h_alpha + h_beta))

        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size)) # this is synaptic current Isyn variable
        # self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

        # integral
        self.integral = bp.odeint(bp.JointEq([self.dV, self.dn, self.dh]), method=method)

    def dh(self, h, t, V):
        alpha = 0.07 * bm.exp(-(V + 58) / 20) #0.07*bm.exp(-V/20.0) #0.07 * bm.exp(-V / 20.)
        beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1) #1/(1+bm.exp(3.0-0.1*V)) #1 / (1 + bm.exp(-(V - 30) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return 5*dhdt

    def dn(self, n, t, V):
        alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1) # 0.01*(10.0-V)/ (bm.exp(1.0-0.1*V )-1) #0.01 * (V - 10) / (1 - bm.exp(-(V -10) / 10))
        beta =  0.125 * bm.exp(-(V + 44) / 80) #0.125*bm.exp(-V/80.0) #0.125 * bm.exp(-V / 80)
        dndt = alpha * (1 - n) - beta * n
        return 5*dndt

    def dV(self, V, t, n, h, Isyn):
        alpha = -0.1 * (V + 40) / (bm.exp(-0.1 * (V + 40)) - 1)  # 0.1*(25-V) / (bm.exp(2.5 - 0.1*V)-1) #0.1 * (-V + 25) / (1 - bm.exp(-(V -25) / 10))
        beta = 4 * bm.exp(-(V + 65) / 18)  # 4.0*bm.exp(-V/18.0) #4.0 * bm.exp(-V / 18)
        m = alpha / (alpha + beta)  # alpha * (1 - m) - beta * m

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
        print('V=', self.V,)
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        # self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

        # update V
        self.V.value = V
        self.n.value = n
        self.h.value = h
        self.input[:]= 0 #[:] = input # this is synaptic current


num = 100 # num neurons
neu = HH(num)

def run_syn(syn_model, title, run_duration=500., sp_times=(10, 20, 30), **kwargs):
    pre = neu#bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.68)) #bp.neurons.SpikeTimeGroup(1, times=sp_times, indices=[0] * len(sp_times))
    post = neu#bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.68))
   
    syn = syn_model(pre, post, conn=bp.connect.All2All(include_self=False), **kwargs)
    net = bp.Network(pre=pre, syn=syn, post=post)

    runner = bp.DSRunner(net, monitors=['pre.spike', 'post.spike', 'syn.g', 'post.input','syn.spike_arrival_time','pre.V','post.V'],inputs=['pre.input', 1.2])
    runner.run(run_duration)

    fig, gs = bp.visualize.get_figure(3, 1, 3, 12.)

    ax = fig.add_subplot(gs[0, 0])
    #plt.plot(runner.mon.ts, runner.mon['post.spike'], label='post.spike')
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['pre.V'])  # [:,0], label='V')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
  
    ax = fig.add_subplot(gs[1, 0])
    plt.plot(runner.mon.ts, runner.mon['pre.V'][:,0], label='V')
    plt.legend(loc='upper right')
    plt.xlabel(r'$t$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.savefig('../img/DeltaSynapse.pdf', transparent=True, dpi=500)
    plt.show()

    ax = fig.add_subplot(gs[2, 0])

    bp.visualize.raster_plot(runner.mon.ts, runner.mon['post.V'])  # [:,0], label='V')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


class AMPA(bp.synapses.TwoEndConn):
    def __init__(self, pre, post, conn, g_max=0.1, E=-75., alpha=12, beta=0.1,
                 T_0=0.5, T_dur=0.5, delay_step=5, method='exponential_euler', **kwargs):
        super(AMPA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
        self.check_pre_attrs('spike')
        self.check_post_attrs('input', 'V')

        # 初始化参数
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T_0 = T_0
        self.T_dur = T_dur
        self.delay_step = delay_step

        # 获取关于连接的信息
        #self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息
        self.conn_mat = self.conn.require('conn_mat')
        # 初始化变量
        #self.s = bm.Variable(bm.zeros(self.post.num))
        #self.g = bm.Variable(bm.zeros((self.pre.num,self.post.num)))
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)  # 脉冲到达的时间
        self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

        # 定义积分函数
        self.integral = bp.odeint(self.derivative, method=method)

    def derivative(self, g, t, T):
        dgdt = self.alpha * T * (1 - g) - self.beta * g
        return dgdt

    def update(self):
        tdi = bp.share.get_shargs()
        # 将突触前神经元传来的信号延迟delay_step的时间步长
        delayed_pre_spike = self.delay(self.delay_step)
        self.delay.update(self.pre.spike)

        # 更新脉冲到达的时间，并以此计算T
        self.spike_arrival_time.value = bm.where(delayed_pre_spike, tdi.t, self.spike_arrival_time)

        T = 1 / (1 + bm.exp(-(self.pre.V - neu.V_th) / 2))

        # 更新s和g
        self.g.value = self.integral(self.g, tdi.t, T, tdi.dt)

        #self.g.value = self.g_max * self.s

        # 电导模式下计算突触后电流大小
        self.post.input += self.g_max *self.g * (self.E - self.post.V) #self.g_max *self.g.sum(axis=0) * (self.E - self.post.V)

if __name__ == '__main__':
    run_syn(AMPA, title='AMPA Synapse Model', g_max=0.1/100)
