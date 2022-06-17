import jax
import jax.numpy as jnp
from jax.numpy import einsum

import flax.linen as nn

from einops import rearrange

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    dim: int
    eps: float = 1e-5

    @nn.compact
    def call(self, x):
        eps = self.eps

        self.g = tf.Variable(tf.ones([1, 1, 1, dim]))
        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])
        self.b = tf.Variable(tf.zeros([1, 1, 1, dim]))
        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])

        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)

        x = (x - mean) / tf.sqrt((var + eps)) * self.g + self.b

        return x

class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    dim: int
    mult: int
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features = self.dim * self.mult, kernel_size = (1, 1), strides = (1, 1), use_bias = False)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        x = nn.Conv(features = self.dim, kernel_size = (1, 1), strides = (1, 1), use_bias = False)(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        return x


class DepthWiseConv2d(nn.Module):
    dim_in 
    dim_out 
    kernel_size 
    stride 
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        net = []
        net += [nn.Conv(features = self.dim_in, kernel_size = self.kernel_size, strides = self.stride, padding = 'SAME', groups = self.dim_in, use_bias = self.bias)]
        net += [nn.BatchNorm(use_running_average = False, momentum = 0.9, epsilon = 1e-5)]
        net += [nn.Conv(features = self.dim_out, kernel_size = (1, 1), strides = (1, 1), use_bias = self.bias)] 
        x = nn.Sequential(net)(x)
        return x

class Attention(nn.Module): 
    dim: int 
    proj_kernel 
    kv_proj_stride 
    heads: int = 8 
    dim_head: int = 64 
    dropout: float = 0.

    @nn.compact
    def call(self, x):
        inner_dim = self.dim_head * self.heads
        heads = self.heads
        scale = self.dim_head ** -0.5

        b, _, y, n = x.shape
        h = heads
        q = DepthWiseConv2d(self.dim, inner_dim, self.proj_kernel, stride = (1, 1), bias = False)(x)
        kv = DepthWiseConv2d(self.dim, inner_dim * 2, self.proj_kernel, stride = self.kv_proj_stride, bias = False)(x)
        k, v = jnp.split(kv, 2, axis = -1)
        qkv = (q, k, v)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> (b h) (x y) d', h = h), qkv)

        dots = einsum('b i d, b j d -> b i j', q, k) * scale
        attn = nn.softmax(dots, axis = -1)

        x = einsum('b i j, b j d -> b i d', attn, v)
        x = rearrange(x, '(b h) (x y) d -> b x y (h d)', h = h, y=y)

        x = nn.Conv(features = self.dim_out, kernel_size = (1, 1), strides = (1, 1), use_bias = False)(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)

        return x

class Transformer(nn.Module):
    dim: int 
    proj_kernel, 
    kv_proj_stride, 
    depth: int 
    heads: int 
    dim_head: int = 64 
    mlp_mult: int = 4
    dropout: float = 0.

    @nn.compact
    def call(self, x):

        layers = []

        for _ in range(self.depth):
            layers.append([
                PreNorm(self.dim, Attention(self.dim, proj_kernel = self.proj_kernel, kv_proj_stride = self.kv_proj_stride, heads = self.heads, dim_head = self.dim_head, dropout = self.dropout)),
                PreNorm(self.dim, FeedForward(self.dim, self.mlp_mult, dropout = self.dropout))
            ])

        for i, (attn, ff) in enumerate(layers):
            x = attn(x) + x
            x = ff(x) + x

        return x

class CvT(nn.Module):
    num_classes,
    s1_emb_dim=64,
    s1_emb_kernel=7,
    s1_emb_stride=4,
    s1_proj_kernel=3,
    s1_kv_proj_stride=2,
    s1_heads=1,
    s1_depth=1,
    s1_mlp_mult=4,
    s2_emb_dim=192,
    s2_emb_kernel=3,
    s2_emb_stride=2,
    s2_proj_kernel=3,
    s2_kv_proj_stride=2,
    s2_heads=3,
    s2_depth=2,
    s2_mlp_mult=4,
    s3_emb_dim=384,
    s3_emb_kernel=3,
    s3_emb_stride=2,
    s3_proj_kernel=3,
    s3_kv_proj_stride=2,
    s3_heads=6,
    s3_depth=10,
    s3_mlp_mult=4,
    dropout=0.

    @nn.compact
    def __call__(self, img, training=True, **kwargs):

        kwargs = dict(locals())

        cvt_layers = nn.Sequential()

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            cvt_layers.add(nn.Sequential([
                nn.Conv2D(filters=config['emb_dim'], kernel_size=config['emb_kernel'], padding='SAME', strides=config['emb_stride']),
                LayerNorm(config['emb_dim']),
                Transformer(dim=config['emb_dim'], proj_kernel=config['proj_kernel'],
                            kv_proj_stride=config['kv_proj_stride'], depth=config['depth'], heads=config['heads'],
                            mlp_mult=config['mlp_mult'], dropout = self.dropout)
            ]))


        cvt_layers.add(nn.Sequential([
            nn.GlobalAvgPool2D(),
            nn.Dense(units = num_classes)
       ]))

        x = cvt_layers(img)
        return x


if __name__ == '__main__':

    import numpy

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 256, 256, 3))

    v = CvT(
        num_classes = 1000,
        s1_emb_dim = 64,        # stage 1 - dimension
        s1_emb_kernel = 7,      # stage 1 - conv kernel
        s1_emb_stride = 4,      # stage 1 - conv stride
        s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
        s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
        s1_heads = 1,           # stage 1 - heads
        s1_depth = 1,           # stage 1 - depth
        s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
        s2_emb_dim = 192,       # stage 2 - (same as above)
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,       # stage 3 - (same as above)
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 4,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.
    )

    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2), 
                'emb_dropout': jax.random.PRNGKey(3)}

    params = v.init(init_rngs, img)
    output = v.apply(params, img, rngs=init_rngs)
    print(output.shape)

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: numpy.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")
