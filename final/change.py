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
        a = bp.dyn.HHLTC(10)
        a._V_initializer = bm.Variable(bm.ones(size) * -65)
        V = a._V_initializer
        a._h_initializer = bm.Variable(bm.ones(size) * 0.6)
        a._n_initializer = bm.Variable(bm.ones(size) * 0.32)
        a._m_initializer = bm.Variable(bm.ones(size) * 0.5)
        m =a._m_initializer
        a.n_beta = 0.125 * bm.exp(-(V + 44) / 80)
        a.n_alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
        a.m_alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
        a.m_beta = 4 * bm.exp(-(V + 60) / 18)
        a.h_alpha = 0.07 * bm.exp(-(V + 58) / 20)
        a.h_beta =  1 / (bm.exp(-0.1 * (V + 28)) + 1)
       
       #  a.n_beta = lambda a, V:0.125 * bm.exp(-(V + 44) / 80)
       #  a.n_alpha = lambda a, V:-0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
       #  a.m_alpha = lambda a, V: -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
       # # a.m  =
       #  a.m_beta = lambda a, V:4 * bm.exp(-(V + 60) / 18)
        a.dm = a.m_alpha * (1 - m) - a.m_beta * m
        # It is giving error in dm still accessing previous definition of dm in HH 
        # dm = lambda a, m, t, V: self.m_alpha(V) * (1 - m) - self.m_beta(V) * m
        # a.h_alpha = lambda a, V:0.07 * bm.exp(-(V + 58) / 20)
        # a.h_beta =  lambda a, V:1 / (bm.exp(-0.1 * (V + 28)) + 1)


      # I dont know how to integrate HH with a  in self.pre 
      # is it correct method?
      
        #
        self.pre = bp.dyn.HH(10) #neu#bp.dyn.SpikeTimeGroup(10, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.)) #bp.neurons.HH(10, V_initializer=bp.init.Constant(-70.)) #bp.dyn.SpikeTimeGroup(1, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.))
        
        bm.set_dt(0.01)
        bp.share.save(t=bm.get_dt())
        self.pre.update(20)

        
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

# indices = np.arange(500)  # 100 ms, dt= 0.1 ms
indices = np.arange(5000).reshape(-1, 10)  # 100 ms, dt= 0.1 ms
inputs = np.ones(indices.shape) * 1.2
#conductances, currents, potentials = bm.for_loop(SimpleNet(E=0.).step_run, indices, progress_bar=True)
conductances, currents, potentials = bm.for_loop(SimpleNet(E=0.).step_run, (indices,inputs), progress_bar=True)
# still giving some error 
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


