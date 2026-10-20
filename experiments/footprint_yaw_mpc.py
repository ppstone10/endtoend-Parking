"""车辆四角姿态残差的航向尺度试验；不读取环境，不改变默认MPC。"""
from copy import deepcopy

import numpy as np

from controller import MPCController


class FootprintYawMPC(MPCController):
    def __init__(self,*,vehicle_length,vehicle_width,gain=1.,**kwargs):
        values=np.array([vehicle_length,vehicle_width,gain],dtype=float)
        if not np.isfinite(values).all() or vehicle_length<=0 or vehicle_width<=0 or gain<0:
            raise ValueError('车身尺寸须为正、gain非负且有限')
        super().__init__(**kwargs)
        self.vehicle_length=float(vehicle_length)
        self.vehicle_width=float(vehicle_width)
        self.gain=float(gain)
        self.radius_squared=(self.vehicle_length**2+self.vehicle_width**2)/4

    @classmethod
    def from_controller(cls,controller,*,vehicle_length,vehicle_width,gain=1.):
        if type(controller) is not MPCController:raise ValueError('只能从原MPC独立复制')
        keys=('dt','horizon','max_v','max_omega','w_xy','w_yaw','w_terminal','w_v',
              'w_omega','population','elite_frac','iterations')
        kwargs={key:getattr(controller,key) for key in keys}
        clone=cls(vehicle_length=vehicle_length,vehicle_width=vehicle_width,gain=gain,
                  catchup_window=controller._catchup_window,**kwargs)
        # 整体复制保持参考/对齐缓存的内部别名关系，且不消费原随机流。
        clone.__dict__.update(deepcopy(controller.__dict__))
        return clone

    def orientation_residual(self,delta):
        return 4*self.radius_squared*np.sin(np.asarray(delta)/2)**2

    def _batch_cost(self,samples,ref,state):
        original=super()._batch_cost(samples,ref,state)
        if self.gain==0:return original
        yaw=np.full(len(samples),state.yaw,dtype=float)
        extra=np.zeros(len(samples))
        for step in range(self.horizon):
            yaw=yaw+samples[:,2*step+1]*self.dt
            residual=self.orientation_residual(yaw-ref[step,2])
            extra+=self.w_xy*residual
        extra+=self.w_terminal*residual
        return original+self.gain*extra
