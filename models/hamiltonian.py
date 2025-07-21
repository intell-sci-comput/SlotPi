import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import misc
from .attention import GeneralizedDotProductAttention, TransformerBlock
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union
from einops import rearrange

DType = Any
Array = torch.Tensor  # np.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[
    str, "ArrayTree"]]  # pytype: disable=not-supported-yet  # TODO: what is this ?
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]

class Integrator:
    '''
    用于哈密尔顿的积分求解器，提供了Euler、RK4、Leapfrog和Yoshida四种积分方法
    '''
    METHODS = ["Euler", "RK4", "Leapfrog", "Yoshida"]

    def __init__(self, delta_t=0.125, method="Euler"):
        """

        Args:
            delta_t (float): 不同积分器的积分时间步长
            method (str, optional):  积分方法. "Euler", "RK4", "Leapfrog" or "Yoshida". Defaults to "Euler".

        Raises:
            KeyError: If the integration method passed is invalid.
        """
        if method not in self.METHODS:
            msg = "%s is not a supported method. " % (method)
            msg += "Available methods are: " + "".join("%s " % m
                                                       for m in self.METHODS)
            raise KeyError(msg)

        self.delta_t = delta_t
        self.method = method

    def _get_grads(self, q, p, hnn, remember_energy=False):
        """Apply the Hamiltonian equations to the Hamiltonian network to get dq_dt, dp_dt.

        Args:
            q (torch.Tensor): Latent-space 位置张量.
            p (torch.Tensor): Latent-space 动量张量.
            energy: 利用哈密尔顿网络计算得到的能量.
            remember_energy (bool): Whether to store the computed energy in self.energy.

        Returns:
            tuple(torch.Tensor, torch.Tensor): q位置和p动量时间导数: dq_dt, dp_dt.
        """
        # Compute energy of the system

        energy = hnn(q=q, p=p)

        # dq_dt = dH/dp
        
        dq_dt = torch.autograd.grad(energy,
                                    p,
                                    create_graph=True,
                                    retain_graph=True,
                                    grad_outputs=torch.ones_like(energy))[0]

        # dp_dt = -dH/dq
        dp_dt = -torch.autograd.grad(energy,
                                     q,
                                     create_graph=True,
                                     retain_graph=True,
                                     grad_outputs=torch.ones_like(energy))[0]

        if remember_energy:
            self.energy = energy.detach().cpu().numpy()

        return dq_dt, dp_dt

    def _euler_step(self, q, p, hnn):
        """Compute next latent-space position and momentum using Euler integration method.

        Args:
            q (torch.Tensor): Latent-space 位置张量.
            p (torch.Tensor): Latent-space 动量张量.
            energy: 利用哈密尔顿网络计算得到的能量.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position, momentum and energy: q_next, p_next.
        """
        dq_dt, dp_dt = self._get_grads(q, p, hnn, remember_energy=True)

        # Euler integration
        q_next = q + self.delta_t * dq_dt
        p_next = p + self.delta_t * dp_dt
        return q_next, p_next

    def _rk_step(self, q, p, hnn):
        """Compute next latent-space position and momentum using Runge-Kutta 4 integration method.

        Args:
            q (torch.Tensor): Latent-space 位置张量.
            p (torch.Tensor): Latent-space 动量张量.
            energy: 利用哈密尔顿网络计算得到的能量.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position and momentum: q_next, p_next.
        """
        # k1
        k1_q, k1_p = self._get_grads(q, p, hnn, remember_energy=True)

        # k2
        q_2 = q + self.delta_t * k1_q / 2  # x = x_t + dt * k1 / 2
        p_2 = p + self.delta_t * k1_p / 2  # x = x_t + dt * k1 / 2
        k2_q, k2_p = self._get_grads(q_2, p_2, hnn)

        # k3
        q_3 = q + self.delta_t * k2_q / 2  # x = x_t + dt * k2 / 2
        p_3 = p + self.delta_t * k2_p / 2  # x = x_t + dt * k2 / 2
        k3_q, k3_p = self._get_grads(q_3, p_3, hnn)

        # k4
        q_3 = q + self.delta_t * k3_q / 2  # x = x_t + dt * k3
        p_3 = p + self.delta_t * k3_p / 2  # x = x_t + dt * k3
        k4_q, k4_p = self._get_grads(q_3, p_3, hnn)

        # Runge-Kutta 4 integration
        q_next = q + self.delta_t * ((k1_q / 6) + (k2_q / 3) + (k3_q / 3) +
                                     (k4_q / 6))
        p_next = p + self.delta_t * ((k1_p / 6) + (k2_p / 3) + (k3_p / 3) +
                                     (k4_p / 6))
        return q_next, p_next

    def _lf_step(self, q, p, hnn):
        """Compute next latent-space position and momentum using LeapFrog integration method.

        Args:
            q (torch.Tensor): Latent-space 位置张量.
            p (torch.Tensor): Latent-space 动量张量.
            energy: 利用哈密尔顿网络计算得到的能量.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position and momentum: q_next, p_next.
        """
        # get acceleration
        _, dp_dt = self._get_grads(q, p, hnn, remember_energy=True)
        # leapfrog step
        p_next_half = p + dp_dt * (self.delta_t) / 2
        q_next = q + p_next_half * self.delta_t
        # momentum synchronization
        _, dp_next_dt = self._get_grads(q_next, p_next_half, hnn)
        p_next = p_next_half + dp_next_dt * (self.delta_t) / 2
        return q_next, p_next

    def _ys_step(self, q, p, hnn):
        """Compute next latent-space position and momentum using 4th order Yoshida integration method.

        Args:
            q (torch.Tensor): Latent-space 位置张量.
            p (torch.Tensor): Latent-space 动量张量.
            energy: 利用哈密尔顿网络计算得到的能量.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position and momentum: q_next, p_next.
        """
        # yoshida coeficients c_n and d_m
        w_1 = 1./(2 - 2**(1./3))
        w_0 = -(2**(1./3))*w_1
        c_1 = c_4 = w_1/2.
        c_2 = c_3 = (w_0 + w_1)/2.
        d_1 = d_3 = w_1
        d_2 = w_0

        # first order
        q_1 = q + c_1*p*self.delta_t
        _, a_1 = self._get_grads(q_1, p, hnn, remember_energy=True)
        p_1 = p + d_1*a_1*self.delta_t
        # second order
        q_2 = q_1 + c_2*p_1*self.delta_t
        _, a_2 = self._get_grads(q_2, p, hnn, remember_energy=False)
        p_2 = p_1 + d_2*a_2*self.delta_t
        # third order
        q_3 = q_2 + c_3*p_2*self.delta_t
        _, a_3 = self._get_grads(q_3, p, hnn, remember_energy=False)
        p_3 = p_2 + d_3*a_3*self.delta_t
        # fourth order
        q_4 = q_3 + c_4*p_3*self.delta_t

        return q_4, p_3

    def step(self, q, p, hnn):
        """Compute next latent-space position and momentum.

        Args:
            q (torch.Tensor): Latent-space 位置张量.
            p (torch.Tensor): Latent-space 动量张量.
            energy: 利用哈密尔顿网络计算得到的能量.

        Raises:
            NotImplementedError: If the integration method requested is not implemented.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position and momentum: q_next, p_next.
        """
        if self.method == "Euler":
            return self._euler_step(q, p, hnn)
        if self.method == "RK4":
            return self._rk_step(q, p, hnn)
        if self.method == "Leapfrog":
            return self._lf_step(q, p, hnn)
        if self.method == "Yoshida":
            return self._ys_step(q, p, hnn)
        raise NotImplementedError



class SpaceAttentionBlock(nn.Module):
    '''
    q_AttentionBlock,用于单时间步内的空间位置求解.
    '''
    def __init__(self,
                 embed_dim: int,  # slot size
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 pre_norm: bool = False,
                 weight_init= None,
                 dropout_rate: float = 0.1
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.dropout_rate = dropout_rate    

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "embed dim must be divisible by num_heads"
        self.head_dim = qkv_size // num_heads

        self.attn = GeneralizedDotProductAttention()
        # mlp for q,k,v
        self.dense_q = nn.Linear(embed_dim, qkv_size)
        self.dense_k = nn.Linear(embed_dim, qkv_size)
        self.dense_v = nn.Linear(embed_dim, qkv_size)
        # output mlp
        self.mlp = misc.MLP(
            input_size=embed_dim, hidden_size=mlp_size,
            output_size=embed_dim, weight_init=weight_init)


        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.num_heads > 1:
            # sapce
            self.dense_o = nn.Linear(qkv_size, embed_dim)

        if self.dropout_rate > 0:
            self.att_drop = nn.Dropout(self.dropout_rate)
            self.ff_dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs: Array, buffer: Array,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        
        assert inputs.ndim == 3
        del buffer ,padding_mask, train # unused
        B, N, D = inputs.shape
        head_dim = self.qkv_size // self.num_heads

        if self.pre_norm:
            # print("TimeSpaceTransformerBlock1")
            x = self.layernorm_query(inputs)  # B,N,D
            # x_buffer = self.layernorm_tquery(buffer)  # B,T,N,D

            # Space attention -> xs(位置信息 q)
            q = self.dense_q(x).view(B, N, self.num_heads, head_dim)
            k = self.dense_k(x).view(B, N, self.num_heads, head_dim)
            v = self.dense_v(x).view(B, N, self.num_heads, head_dim)
            xs, _ = self.attn(query=q, key=k, value=v)
            if self.num_heads > 1:
                xs = self.dense_o(xs.reshape(B, N, self.qkv_size)).view(B, N, self.embed_dim)
            else:
                xs = xs.squeeze(-2)
            if self.dropout_rate > 0:
                xs = self.att_drop(xs)
            xs = xs + inputs
            # MLP
            y = xs
            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            if self.dropout_rate > 0:
                z = self.ff_dropout(z)
            z = z + y
        else:
            x = inputs
            # x_buffer = buffer
            # Space attention
            q = self.dense_q(x).view(B, N, self.num_heads, head_dim)
            k = self.dense_k(x).view(B, N, self.num_heads, head_dim)
            v = self.dense_v(x).view(B, N, self.num_heads, head_dim)
            xs, _ = self.attn(query=q, key=k, value=v)
            if self.num_heads > 1:
                xs = self.dense_o(xs.reshape(B, N, self.qkv_size)).view(B, N, self.embed_dim)
            else:
                xs = xs.squeeze(-2)
            if self.dropout_rate > 0:
                xs = self.att_drop(xs)
            xs = xs + inputs
            xs = self.layernorm_query(xs)
            # MLP
            y = xs
            z = self.mlp(y)
            if self.dropout_rate > 0:
                z = self.ff_dropout(z)
            z = z + y
            z = self.layernorm_mlp(z)
        return z


class TimeAttentionBlock(nn.Module):
    '''
    p_AttentionBlock,用于单时间步内的空间位置求解.
    '''
    def __init__(self,
                 embed_dim: int,  # slot size
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 pre_norm: bool = False,
                 weight_init= None,
                 dropout_rate: float = 0.1 
                 ):
        super().__init__()
        # del weight_init 
        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.dropout_rate = dropout_rate

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "embed dim must be divisible by num_heads"
        self.head_dim = qkv_size // num_heads

        
        self.attn = GeneralizedDotProductAttention()
        # mlp for q,k,v
        self.dense_tq = nn.Linear(embed_dim, qkv_size)
        self.dense_tk = nn.Linear(embed_dim, qkv_size)
        self.dense_tv = nn.Linear(embed_dim, qkv_size)
        # output mlp
        self.mlp = misc.MLP(
            input_size=embed_dim, hidden_size=mlp_size,
            output_size=embed_dim, weight_init=weight_init)

        self.layernorm_tquery = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.num_heads > 1:  
            self.dense_to = nn.Linear(qkv_size, embed_dim)

        if self.dropout_rate > 0:
            self.att_drop = nn.Dropout(self.dropout_rate)
            self.ff_dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs: Array, buffer: Array,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        
        assert inputs.ndim == 3
        del padding_mask, train # unused
        B, T, N, _ = buffer.shape
        # head_dim = self.qkv_size // self.num_heads

        if self.pre_norm:
            # print("TimeSpaceTransformerBlock1")
            x = self.layernorm_tquery(inputs)  # B,N,D
            x_buffer = self.layernorm_tquery(buffer)  # B,T,N,D

            # Time attention -> xt(时间信息 p)
            xt = torch.unsqueeze(x, dim=1)  # B, 1, N, D
            xt = rearrange(xt, 'b t o d -> (b o) t d')  # B*N, 1, D
            xt_buffer = rearrange(x_buffer, 'b t o d -> (b o) t d')  # B*N, T, D
            qt = self.dense_tq(xt).view(xt.shape[0], xt.shape[1], self.num_heads, self.head_dim)
            kt = self.dense_tk(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            vt = self.dense_tv(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            xt, _ = self.attn(query=qt, key=kt, value=vt)  # B*N, 1, h, d
            if self.num_heads > 1:
                xt = self.dense_to(xt.reshape(B, N, self.qkv_size)).view(B, N, self.embed_dim)
            else:
                xt = xt.squeeze(-2)
            if self.dropout_rate > 0:
                xt = self.att_drop(xt)
            xt = xt + inputs
            # MLP
            y = xt
            # MLP
            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            if self.dropout_rate > 0:               
                z = self.ff_dropout(z)
            z = z + y
        else:
            x = inputs
            x_buffer = buffer
            # Time attention
            xt = torch.unsqueeze(x, dim=1)  # B, 1, N, D
            xt = rearrange(xt, 'b t o d -> (b o) t d')  # B*N, 1, D
            xt_buffer = rearrange(x_buffer, 'b t o d -> (b o) t d')  # B*N, T, D
            qt = self.dense_tq(xt).view(xt.shape[0], xt.shape[1], self.num_heads, self.head_dim)
            kt = self.dense_tk(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            vt = self.dense_tv(xt_buffer).view(xt_buffer.shape[0], xt_buffer.shape[1], self.num_heads, self.head_dim)
            xt, _ = self.attn(query=qt, key=kt, value=vt)  # B*N, 1, h, d
            if self.num_heads > 1:
                xt = self.dense_to(xt.reshape(B, N, self.qkv_size)).view(B, N, self.embed_dim)
            else:
                xt = xt.squeeze(-2)
            if self.dropout_rate > 0:
                xt = self.att_drop(xt)
            xt = xt + inputs
            xt = self.layernorm_tquery(xt)
            # time
            y = xt
            # MLP
            z = self.mlp(y)
            if self.dropout_rate > 0:
                z = self.ff_dropout(z)
            z = z + y
            z = self.layernorm_mlp(z)
        return z



class qp_Attentions(nn.Module):
    def __init__(self,
                 embed_dim: int, 
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 pre_norm: bool = False,
                 weight_init= None,
                 num_layers: int = 1,
                 dropout_rate: float = 0.1,
                 # out mlp config
                 out_mlp = False,
                 out_hidden_layers = 1
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.out_mlp = out_mlp
        self.out_hidden_layers = out_hidden_layers

        assert num_heads >= 1
        assert qkv_size % num_heads == 0, "embed dim must be divisible by num_heads"
        self.head_dim = qkv_size // num_heads

        # built the time and space attention blocks
        
        self.space_model = nn.ModuleList()
        self.time_model = nn.ModuleList()
        for i in range(self.num_layers):
            self.space_model.append(
                                    
                                    module=SpaceAttentionBlock(embed_dim=embed_dim,
                                                        num_heads=num_heads,
                                                        qkv_size=qkv_size,
                                                        mlp_size=mlp_size,
                                                        pre_norm=pre_norm,
                                                        weight_init=weight_init,
                                                        dropout_rate = dropout_rate))
            self.time_model.append(
                                    
                                    module=TimeAttentionBlock(embed_dim=embed_dim,
                                                        num_heads=num_heads,
                                                        qkv_size=qkv_size,
                                                        mlp_size=mlp_size,
                                                        pre_norm=pre_norm,
                                                        weight_init=weight_init,
                                                       dropout_rate = dropout_rate))
        # 最终输出mlp
        if  self.out_mlp:
            self.q_mlp = misc.MLP(
                                input_size= embed_dim, 
                                hidden_size=2*embed_dim, 
                                output_size= embed_dim,
                                num_hidden_layers=self.out_hidden_layers,
                                activate_output=True,
                                weight_init=weight_init,
                                )
            self.p_mlp = misc.MLP(
                                input_size= embed_dim, 
                                hidden_size=2*embed_dim, 
                                output_size= embed_dim,
                                num_hidden_layers=self.out_hidden_layers,
                                activate_output=True,
                                weight_init=weight_init,
                                )
        
    def forward(self, inputs: Array, buffer: Array,
                padding_mask: Optional[Array] = None,
                train: bool = False) -> Array:
        xs = inputs  # (B,N,D)
        xt = inputs  # (B,N,D)  
        x_buffer = buffer  # (B,T,N,D)
        for i in range(self.num_layers):
            # 利用空间注意力求解坐标q
            # 如果使用cross结构，xs的更新也可以分为保持空间信息不变（k,v），只更新q，
            # 保证信息不丢失。即：
            # xs = space_model[i](xs, inputs, padding_mask, train)
            xs = self.space_model[i](xs, xs, padding_mask, train)
            # 利用时间注意力求解动量p
            xt = self.time_model[i](xt, x_buffer, padding_mask, train)
            # 1.时间上，更新buffer最后一个时间步的信息;
            # 2.只更新xt，buffer中的信息保持不变，保证信息不丢失。
            # x_buffer = torch.cat([x_buffer[:, :-1], xt.unsqueeze(1)], dim=1)
        q = xs
        p = xt
        if self.out_mlp:
            q = self.q_mlp(q)
            p = self.p_mlp(p)
        return q, p

class HamiltonianNet(nn.Module):
    def __init__(self,
                 
                 embed_dim: int,  # 
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int,
                 num_layers: int =1,
                 pre_norm: bool = False,
                 weight_init= None,
                 dropout_rate = 0.,
                 ):
        super().__init__()
        
        assert num_layers >= 1
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        self.weight_init = weight_init
        self.dropout_rate = dropout_rate

        self.energy_net = nn.ModuleList()
        for i in range(self.num_layers):
            self.energy_net.append(
                           
                            module=TransformerBlock(embed_dim=self.embed_dim,num_heads=self.num_heads,
                                             qkv_size=self.qkv_size,mlp_size=self.mlp_size,
                                             pre_norm=self.pre_norm,weight_init=self.weight_init,
                                             #dropout_rate=self.dropout_rate
                                             ))
        # output mlp -> H (B,1) 
        self.mlp = misc.MLP(
            input_size=embed_dim, hidden_size=embed_dim,
            output_size=1, weight_init=weight_init,
            activation_fn=nn.Softplus,activate_output=True)

    def forward(self, q: Array, p: Array) -> Array:
        x = torch.cat([q, p], dim=-1)
        for i in range(self.num_layers):
            x = self.energy_net[i](x)
        H = self.mlp(x)
        # 这里sum是否合理
        H = H.sum(dim=1) 
        return H

class Slot_HamiltonianNet(nn.Module):
    '''
    Slot Hamiltonian Network,用于单时间步内的求解.
    包含一个求当前时间slot的坐标q和动量p的全连网络，输入为当前时间的slot.
    一个求解当前时间slot的哈密顿量的全连网络,输入为q，p.
    一个积分器，用于求解下一个时间slot的q和p.
    '''
    def __init__(self,
                 num_slots: int,
                 embed_dim: int,  # slot size
                 num_heads: int,
                 qkv_size: int,
                 mlp_size: int, # attblock中最终输出mlp的隐藏层大小
                 pre_norm: bool = False,
                 weight_init= None,
                 dropout_rate: float = 0.1,

                 # q、p net config
                 num_layers: int = 1, # 多少个q、p_Attentions   
                 # h net config
                 h_layers = 1, # 多少个H_Attentions

                 # final mlp config
                 out_mlp = False, # 用于 att之后最终输出q、p
                 out_hidden_layers = 1 # att之后最终输出mlp的隐藏层数
                 ):
        super().__init__()
        
        # 通用参数
        self.dropout_rate = dropout_rate
        self.weight_init = weight_init
        self.pre_norm = pre_norm
        self.num_heads = num_heads
        self.mlp_size = mlp_size   

        # q、p_Attentions 参数
        self.embed_dim = embed_dim
        self.qkv_size = qkv_size
        self.num_layers = num_layers
        
        # H_Attentions 参数
        self.H_layers = h_layers

        self.out_mlp = out_mlp
        self.out_hidden_layers = out_hidden_layers
        # 求解当前时间slot的坐标q和动量p的全连网络
        self.qp_net = qp_Attentions(
                                    embed_dim = self.embed_dim,  # 
                                    num_heads = self.num_heads,
                                    qkv_size = self.qkv_size,
                                    mlp_size = self.mlp_size,
                                    pre_norm =self.pre_norm,
                                    weight_init= self.weight_init,
                                    num_layers = self.num_layers,
                                    dropout_rate = self.dropout_rate,

                                    out_mlp = self.out_mlp,
                                    out_hidden_layers = self.out_hidden_layers
                                    )
        
        # self.qp_norm = nn.LayerNorm(2*slot_size, eps=1e-6)
        # 求解当前时间slot的哈密顿量 .由于q,p的存在，因此embed_dim*2
        # 考虑是否使用mlp将q,p拼接后，经过一个mlp后再输入H网络？？？？？？
        self.H_net = HamiltonianNet(
                                    embed_dim = self.embed_dim * 2,  # 
                                    num_heads =self.num_heads,
                                    qkv_size = self.qkv_size * 2,
                                    mlp_size = self.mlp_size,
                                    num_layers = self.H_layers,
                                    pre_norm= self.pre_norm,
                                    weight_init= self.weight_init,
                                    dropout_rate = self.dropout_rate,
                                    )
        # 积分器
        self.integrator = Integrator()
        # 输入or输出值标准化
        self.mlp_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, slot: Array ,buffer: Array) :
        '''
        Args:
            slot: 当前时间的slot, shape (batch_size, num_slot, slot_size)
        Returns:
            q_next: 下一个时间的slot坐标, shape (batch_size, slot_size)
            p_next: 下一个时间的slot动量, shape (batch_size, slot_size)
        '''
        # slot = slot.requires_grad_(True)
        if self.pre_norm:
            slot = self.mlp_norm(slot)
            buffer = self.mlp_norm(buffer)
        
        # 求解当前时间slot的坐标q和动量p 
        q, p = self.qp_net(slot ,buffer) # (batch_size, num_slot, 2*slot_size)   
        q.requires_grad_(True)
        p.requires_grad_(True)
        # 求解当前时间slot的哈密顿量
        with torch.enable_grad():
            q_next, p_next = self.integrator.step(q, p, self.H_net) # (batch_size, slot_size)
       
        if not self.pre_norm:
            # we just need q_nest, so we don't need to normalize p_next
            q_next = self.mlp_norm(q_next)

        return q_next, p_next