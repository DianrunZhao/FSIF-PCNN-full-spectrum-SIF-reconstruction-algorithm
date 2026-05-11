import csv
import os
import tensorflow as tf
from numpy import *
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.callbacks import ModelCheckpoint
import pickle
from keras import layers, models
from pathlib import Path


# Print verification information
print("XLA_FLAGS:", os.environ.get('XLA_FLAGS', 'no set'))
os.environ.pop("TF_XLA_FLAGS", None)

tf.config.optimizer.set_jit(True)

print(f"TensorFlow Version: {tf.__version__}")
print(f"CUDA_VERSION: {tf.sysconfig.get_build_info()['cuda_version']}")
print(f"CUDNN_VERSION: {tf.sysconfig.get_build_info()['cudnn_version']}")


policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)
print("Mixed-precision policy:", tf.keras.mixed_precision.global_policy())


def save(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

def parse_tfrecord_fn_cnn(example_proto, global_mean, global_var,sigmaF,sigmaR, global_mean1=None, global_var1=None):

    feature_description = {
        'data': tf.io.FixedLenFeature([1 + 990 * 3], tf.float32),
        'labels': tf.io.FixedLenFeature([990 * 2], tf.float32)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    all_data = example['data']  # shape=(2971,)
    raw_spectral = all_data[1:]  # shape=(2970,)


    normalized = (raw_spectral - global_mean) / tf.sqrt(global_var + 1e-8)

    spectral_2d = tf.reshape(normalized, (3, 990))  # shape=(990,3)
    spectral_2d = tf.transpose(spectral_2d)  # (990,3)

    labels = example['labels']  # shape=(1980,)
    # 1. label
    sif_true = labels[:990] / (sigmaF)
    er_true  = labels[-990:] / (sigmaR)
    # zeros     = tf.zeros_like(sif_true)
    return  spectral_2d, {'sif': sif_true, 'r': er_true}



def create_cnn_dataset(tfrecord_dir, global_mean, global_var,sigmaF,sigmaR, batch_size=32, shuffle=False, num_files=1,rp=False,global_mean1=None, global_var1=None):
    """((tower_id, spectral_2D), label)"""
    file_list = [os.path.join(tfrecord_dir, f"file_{i}.tfrecord") for i in range(1, num_files + 1)]
    ds_files = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        ds_files = ds_files.shuffle(buffer_size=len(file_list), seed=42)

    ds = ds_files.interleave(
        lambda x: tf.data.TFRecordDataset(x,num_parallel_reads=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE,
        # block_length=16,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        ds = ds.shuffle(buffer_size=48000*1,reshuffle_each_iteration=True, seed=42)
    # ds = ds.repeat()
    parse_fn = lambda x: parse_tfrecord_fn_cnn(x, global_mean, global_var,sigmaF,sigmaR, global_mean1, global_var1)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.cache() 
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

from keras.callbacks import Callback


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        if isinstance(opt, tf.keras.mixed_precision.LossScaleOptimizer):
            if hasattr(opt, 'optimizer'):
                inner_opt = opt.optimizer
            elif hasattr(opt, 'inner_optimizer'):
                inner_opt = opt.inner_optimizer
            else:
                inner_opt = opt
        else:
            inner_opt = opt

        lr = inner_opt.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(inner_opt.iterations)

        try:
            lr_val = tf.keras.backend.get_value(lr)
        except:
            lr_val = lr
        print(f"\nEpoch {epoch + 1}: learning rate = {lr_val:.6g}")


from keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense



##############################################
# EncoderBlock
##############################################
def EncoderBlock(inputs, filters, kernel_sizes=[3, 5, 7], bottleneck_ratio=0.5, name_prefix="Enc",use_mhsa=False,
                 num_heads=4, key_dim=64,seq_len=None, drop_path=0.):
    x = MultiScaleBlock2(inputs, filters, kernel_sizes, bottleneck_ratio=bottleneck_ratio,name_prefix=f"{name_prefix}_MSC")
    x = ResidualBlock2(x, filters, bottleneck_ratio=bottleneck_ratio, name_prefix=f"{name_prefix}_Res")

    skip = x
    pool = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        strides=2,
        padding='same',
        name=f"{name_prefix}_downconv", use_bias=False
    )(x)

    pool = layers.BatchNormalization(name=f"{name_prefix}_down_bn")(pool)
    # pool = layers.Activation('relu', name=f"{name_prefix}_down_relu")(pool)
    pool = tf.keras.activations.gelu(pool, approximate=True)

    return skip, pool



##############################################
# DecoderBlock
##############################################
def DecoderBlock(inputs, skip, filters,kernel_sizes=[3,5,7],bottleneck_ratio=0.5, name_prefix="Dec", target_len=None,use_mhsa=False,
                 num_heads=4, key_dim=64):
    x = layers.Conv1DTranspose(filters, 4, strides=2, padding='same',
                            use_bias=False, name=f'{name_prefix}_up')(inputs)
    x = layers.BatchNormalization(
                            name=f'{name_prefix}_up_bn')(x)
    x = tf.keras.activations.gelu(x, approximate=True)

    if target_len is not None:
        x = Crop1D(target_len)(x)

    x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip])
    x = MultiScaleBlock2(x, filters,kernel_sizes,bottleneck_ratio=bottleneck_ratio, name_prefix=f"{name_prefix}_MSC")
    x = ResidualBlock2(x, filters,bottleneck_ratio=bottleneck_ratio, d=1, name_prefix=f"{name_prefix}_Res")
    return x

def Crop1D(target_len):
    """
    """
    def crop_fn(x):
        # x.shape=(batch, seq, channels)
        seq = tf.shape(x)[1]
        return x[:, :target_len, :]
    return layers.Lambda(crop_fn)


def SE1D(x, r=8, name="se"):
    '''
    Function: Enables the network to automatically learn "which channels are important"

    - For (B,seq,C), first perform GlobalAvgPool1D → (B,1,C)

    - Two FCs: C → C/r → C, finally use sigmoid to obtain the weights
    '''
    c = x.shape[-1]
    # s = tf.reduce_mean(x, axis=1, keepdims=True)                 # GAP
    s = layers.GlobalAveragePooling1D(keepdims=True, name=f"{name}_gap")(x)

    s = layers.Dense(c//r, use_bias=False, name=f"{name}_fc1")(s)
    s = tf.keras.activations.gelu(s, approximate=True)

    s = layers.Dense(c,    use_bias=False, name=f"{name}_fc2")(s)
    s = tf.keras.activations.sigmoid(s)          # sigmoid

    return layers.Multiply(name=f"{name}_scale")([x, s])


def MultiScaleBlock2(inputs, filters, kernel_sizes=[3, 5,7], bottleneck_ratio=0.5,name_prefix="MSC"):
    branches = []
    bottleneck_channels = max(1, int(filters * bottleneck_ratio))
    # Bottleneck
    x = layers.Conv1D(bottleneck_channels, 1,use_bias=False,
                        name=f"{name_prefix}_bt_pre")(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.gelu(x, approximate=True)

    for k in kernel_sizes:

        branch = layers.Conv1D(bottleneck_channels, kernel_size=k, padding='same',use_bias=False,
                                name=f"{name_prefix}_conv{k}")(x)
        # BN
        branch = layers.BatchNormalization(name=f"{name_prefix}_bn{k}")(branch)
        branch = tf.keras.activations.gelu(branch, approximate=True)

        branches.append(branch)
    x = layers.Concatenate(name=f"{name_prefix}_concat")(branches)

    x = layers.Conv1D(filters, kernel_size=1, padding='same', use_bias=False,
                      name=f"{name_prefix}_conv1x1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1x1")(x)
    x = tf.keras.activations.gelu(x, approximate=True)
    return x


def ResidualBlock2(x, filters, kernel_size=3, stride=1, d=1, bottleneck_ratio=0.5, name_prefix="Res_bottleneck"):
    shortcut = x
    # ─── 1 ───
    x = layers.Conv1D(filters, kernel_size, stride, padding='same',
                      use_bias=False, name=f'{name_prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = tf.keras.activations.gelu(x, approximate=True)

    # ─── 2 ───
    x = layers.Conv1D(filters, kernel_size, 1, padding='same', dilation_rate=d,
                      use_bias=False, name=f'{name_prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = tf.keras.activations.gelu(x, approximate=True)

    # ─── 3 ───
    x = layers.Conv1D(filters, kernel_size, 1, padding='same',
                      use_bias=False, name=f'{name_prefix}_conv3')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
    # ─── Shortcut ───
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(filters, 1, stride, padding='same',
                                 use_bias=False, name=f'{name_prefix}_proj')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name_prefix}_proj_bn')(shortcut)
    # ─── Residual summation + activation───
    x = layers.Add()([x, shortcut])
    x = tf.keras.activations.gelu(x, approximate=True)
    x = SE1D(x, r=8, name=f"{name_prefix}_se")
    return x

##############################################
# Main Network: Physically Constrained Multi-Scale U-Net
##############################################
def create_advanced_model(seq_len=990, in_channels=3, sigmaF=None,sigmaR=None,global_mean=None, global_var=None,global_mean1=None, global_var1=None):
    """
    Input:
    - spectral_input: (990,3) [E, L, noise_std]
    Output:
    - (990,2): First channel is SIF, second channel is reflectance
    Design Concept:
    1. Input Branch: Concatenated to obtain (990,3).
    2. Encoder: Multi-scale convolution and residual blocks are used for progressive downsampling to extract global high-order features and suppress noise.
    3. Decoder: Upsampling and skip connections are used to recover details, ultimately restoring the resolution to the 990 band.
    4. Prediction Branch: Divided into two branches, predicting reflectance R (constrained by sigmoid) and SIF (ReLU ensures non-negativity) respectively.
    5. Physical Constraints: The input E is multiplied by R to obtain E*R, and then added to SIF to form the final prediction, conforming to the physical relationship L≈E*R+SIF.
    """
    # Input
    spectral_input = Input(shape=(seq_len, in_channels), name="spectral_input")
    x = spectral_input#layers.Concatenate(axis=-1, name="input_concat")([spectral_input, tower_emb])  # (990, 3+1)= (990,4)

    # EncoderBlock
    skip1, pool1 = EncoderBlock(x, 64,kernel_sizes=[3,7,11], bottleneck_ratio=0.5, name_prefix="Enc1",)  # skip1: (990,32), pool1: (495,32)
    skip2, pool2 = EncoderBlock(pool1, 128,kernel_sizes=[3,7,11], bottleneck_ratio=0.25, name_prefix="Enc2")  # skip2: (495,64), pool2: (≈247,64)
    skip3, pool3 = EncoderBlock(pool2, 256,kernel_sizes=[3,7,11],bottleneck_ratio=0.25,  name_prefix="Enc3")  # skip3: (≈247,128), pool3: (≈124,128)
    skip4, pool4 = EncoderBlock(pool3, 512,kernel_sizes=[3,7,11],bottleneck_ratio=0.25,  name_prefix="Enc4")  # skip3: (≈247,128), pool3: (≈124,128)

    # Bottleneck
    bn = MultiScaleBlock2(pool4, 1024,bottleneck_ratio=0.25,kernel_sizes=[3,7,11], name_prefix="Bottleneck_MSC")
    bn = ResidualBlock2(bn, 1024,bottleneck_ratio=0.25, name_prefix="Bottleneck_Res")

    # DecoderBlock
    dec4 = DecoderBlock(bn, skip4, 512, kernel_sizes=[3,7,11],bottleneck_ratio=0.25,name_prefix="Dec1")  # (≈988,32)

    dec3 = DecoderBlock(dec4, skip3, 256,kernel_sizes=[3,7,11],bottleneck_ratio=0.25, name_prefix="Dec2")  # (≈247,128)
    dec2 = DecoderBlock(dec3, skip2, 128,kernel_sizes=[3,7,11],bottleneck_ratio=0.25, name_prefix="Dec3", target_len=495)  # (≈247,128)
    dec1 = DecoderBlock(dec2, skip1, 64, kernel_sizes=[3,7,11],bottleneck_ratio=0.5,name_prefix="Dec4")  # (≈494,64)

    sif_smooth = GaussianSmoother1D(
        ksize=121, init_sigma=12, min_sigma=3, max_sigma=20, name="sif_gauss")

    r_smooth = GaussianSmoother1D(
        ksize=121, init_sigma=7, min_sigma=0, max_sigma=20, name="r_gauss")

    # SIF branch (using ReLU to ensure non-negativity)
    SIF_branch = layers.Conv1D(64, 3, padding='same', name="SIF_conv1")(dec1)
    SIF_branch = layers.BatchNormalization(
                                    name="Fcconv1_bn")(SIF_branch)
    SIF_branch = tf.keras.activations.gelu(SIF_branch, approximate=True)

    SIF_branch = layers.Conv1D(32, 3, padding='same', name="SIF_conv2")(SIF_branch)
    SIF_branch = layers.BatchNormalization(
                            name="Fcconv2_bn")(SIF_branch)
    SIF_branch = tf.keras.activations.gelu(SIF_branch, approximate=True)

    SIF_pred1 = layers.Conv1D(1, 3, padding='same', activation='softplus', name="SIF_output")(SIF_branch)  # (990,1)

    SIF_pred1 = sif_smooth(SIF_pred1)      # (B,990,1)
    sif_flat1 = layers.Flatten(name="sif_flat")(SIF_pred1)  #(batch, 990)

    E_in = layers.Lambda(lambda t: t[..., 0])(spectral_input)
    L_in = layers.Lambda(lambda t: t[..., 1])(spectral_input)

    mu_E   = global_mean[:990]
    mu_L   = global_mean[990:990*2]
    sigmaE = np.sqrt(global_var[:990]       + 1e-8).astype(np.float32)
    sigmaL = np.sqrt(global_var[990:990*2]  + 1e-8).astype(np.float32)
    
    E_raw  = Affine1D(sigmaE, mu_E , name='E_raw')(E_in)
    L_raw  = Affine1D(sigmaL, mu_L , name='L_raw')(L_in)

    R_from_sif = RFromSIF( name="R_from_SIF")(
        [E_raw, sif_flat1, L_raw])

    R_3d = layers.Reshape((seq_len, 1), name="R_expand")(R_from_sif)

    R_pred    = r_smooth(R_3d)
    r_flat1 = layers.Flatten(name="er_flat")(R_pred)  #(batch, 990)

    sif_flat = DivideBySigma(sigmaF, name='sif')(sif_flat1)
    r_flat   = DivideBySigma(sigmaR, name='r')(r_flat1)

    model=models.Model(spectral_input,
                        {'sif': sif_flat, 'r': r_flat},
                        name='PG-UNet')

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, beta_2=0.99, clipnorm=1.0)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)

    init_sif, init_r = 1.0, 1/30.0

    log_var_sif = model.add_weight(
        name='log_var_sif', shape=(),  initializer=tf.constant_initializer(np.log(init_sif)),trainable=True)
    log_var_r   = model.add_weight(
        name='log_var_r',   shape=(), initializer=tf.constant_initializer(np.log(init_r)), trainable=True)

    def nll_loss(which):
        def f(y_true, y_pred):
            lv = log_var_sif if which=='sif' else log_var_r
            base = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)  # MSE per sample
            return tf.exp(-lv) * base + lv
        return f

    model.compile(
        optimizer      = optimizer,
        # loss           = {'sif':'mse', 'r':'mse', 'phys':'mse'},#, 'phys':phys_loss_obj
        loss={'sif': nll_loss('sif'), 'r': nll_loss('r')},

        loss_weights   = {'sif':1, 'r':1},
        metrics      = {'sif':['mse',UnnormMSE ( sigmaF, name='F_raw_mse'),
            UnnormRRMSE(sigmaF, name='F_raw_rrmse')],
                    'r'  :['mse',UnnormMSE (sigmaR, name='R_raw_mse'),
            UnnormRRMSE(sigmaR, name='R_raw_rrmse')],
                    'phys':['mse']})

    print(model.summary())
    return model

# log_var_sif = tf.Variable(0.0, trainable=True, name="log_var_sif")
# log_var_r   = tf.Variable(0.0, trainable=True, name="log_var_r")

from keras.saving import register_keras_serializable

@register_keras_serializable(package="custom")
class RFromSIF(tf.keras.layers.Layer):
    """
    Derivation of R from SIF: R = (L - SIF) / (c * max(|E|, e_floor))
    - e_floor prevents division to zero, c is a learnable scaling constant, absorbing unit/geometric factors.
    """
    def __init__(self, e_floor=1e-8, c_init=1.0, learn_c=False, clip01=True, **kw):
        super().__init__(**kw)
        self.e_floor = float(e_floor)
        self.c_init  = float(c_init)
        self.learn_c = bool(learn_c)
        self.clip01  = bool(clip01)

    def build(self, _):
        self.c = self.add_weight(
            name="c_ER", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self.c_init),
            trainable=self.learn_c,
        )

    def call(self, inputs):
        E_raw, SIF_raw, L_raw = inputs   # (B,990), (B,990), (B,990)
        denom = E_raw#tf.maximum(tf.abs(E_raw), tf.cast(self.e_floor, E_raw.dtype))
        R = (L_raw - SIF_raw) / ( denom)#self.c *
        if self.clip01:
            R = tf.clip_by_value(R, 0.0, 1)
        return R



@tf.keras.utils.register_keras_serializable(package="custom")
class GaussianSmoother1D(tf.keras.layers.Layer):
    """
    1D Gaussian smoothing (spectral axis)
    - ksize: odd number (unit = band)
    - sigma: parameterized using softplus; then clipped to [min_sigma, max_sigma]
    """
    def __init__(self, ksize=21, init_sigma=2.0, min_sigma=0.6, max_sigma=6.0,
                 trainable_sigma=True, padding="REFLECT", **kw):
        super().__init__(**kw)
        assert ksize % 2 == 1, "ksize must be odd"
        self.ksize = int(ksize)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.padding = padding
        self._half = self.ksize // 2
        self._x = None


        init = np.log(np.exp(init_sigma) - 1.0).astype(np.float32)
        self.log_sp_sigma = self.add_weight(
            name="log_sp_sigma",
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(init),
            trainable=trainable_sigma,
        )

    def build(self, input_shape):
        self._x = tf.range(-self._half, self._half + 1, dtype=tf.float32)  # (K,)
        super().build(input_shape)

    def call(self, y):
        # y: (B, L, 1)
        y32 = tf.cast(y, tf.float32)

        sigma = tf.nn.softplus(self.log_sp_sigma)
        sigma = tf.clip_by_value(sigma, self.min_sigma, self.max_sigma)

        k = tf.exp(-0.5 * (self._x / sigma) ** 2)
        k = k / tf.reduce_sum(k)
        k = tf.reshape(k, [self.ksize, 1, 1])  # (K, Cin=1, Cout=1)

        if self.padding == "REFLECT":
            ypad = tf.pad(y32, [[0, 0], [self._half, self._half], [0, 0]], mode="REFLECT")
            out = tf.nn.conv1d(ypad, k, stride=1, padding="VALID")
        else:
            out = tf.nn.conv1d(y32, k, stride=1, padding="SAME")

        return tf.cast(out, y.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "ksize": self.ksize,
            "min_sigma": self.min_sigma,
            "max_sigma": self.max_sigma,
            "padding": self.padding,
        })
        return cfg


from keras.saving import register_keras_serializable

@register_keras_serializable(package="custom")
class DivideBySigma(tf.keras.layers.Layer):
    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        self._sigma = np.asarray(sigma, np.float32)

    def call(self, x):
        sigma = tf.convert_to_tensor(self._sigma, dtype=x.dtype)
        return x / (sigma + 1e-8)

    def get_config(self):
        cfg = super().get_config()
        cfg["sigma"] = self._sigma.tolist()
        return cfg



class Affine1D(tf.keras.layers.Layer):
    def __init__(self, scale, bias, **kw):
        super().__init__(**kw)
        self._scale_np = np.asarray(scale, np.float32)
        self._bias_np  = np.asarray(bias , np.float32)

    def call(self, x):
        s = tf.cast(self._scale_np, x.dtype)
        b = tf.cast(self._bias_np , x.dtype)
        return x * s + b

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"scale": self._scale_np.tolist(),
                    "bias":  self._bias_np.tolist()})
        return cfg


import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable


# ───────────────────────── UnnormMSE ──────────────────────────
@register_keras_serializable(package="custom")
class UnnormMSE(tf.keras.metrics.Mean):
    """
        Mean squared error after inverse normalization:
        MSE = mean[(ŷ·σ − y·σ)²]
    """
    def __init__(self, sigma, name="unnorm_mse", **kwargs):
        super().__init__(name=name, **kwargs)

        self._sigma_init = np.asarray(sigma, np.float32).tolist()

        self.sigma = self.add_weight(
            name="sigma",
            shape=(len(self._sigma_init),),
            dtype=tf.float32,
            initializer=tf.constant_initializer(self._sigma_init),
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true, tf.float32) * self.sigma
        yp = tf.cast(y_pred, tf.float32) * self.sigma

        per_sample_mse = tf.reduce_mean(tf.square(yp - yt), axis=1)   # (B,)
        super().update_state(per_sample_mse, sample_weight=sample_weight)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"sigma": self._sigma_init})
        return cfg


# ───────────────────────── UnnormRRMSE ──────────────────────────
@register_keras_serializable(package="custom")
class UnnormRRMSE(tf.keras.metrics.Mean):
    """
        Relative root mean square error after inverse normalization:
        RRMSE = √mean[ ((ŷ·σ − y·σ)/(y·σ+ε))² ]
    """
    def __init__(self, sigma, eps=1e-3, name="unnorm_rrmse", **kwargs):
        super().__init__(name=name, **kwargs)

        self._sigma_init = np.asarray(sigma, np.float32).tolist()
        self._eps_init   = float(eps)

        self.sigma = self.add_weight(
            name="sigma",
            shape=(len(self._sigma_init),),
            dtype=tf.float32,
            initializer=tf.constant_initializer(self._sigma_init),
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true, tf.float32) * self.sigma
        yp = tf.cast(y_pred, tf.float32) * self.sigma
        eps = tf.cast(self._eps_init, tf.float32)

        per_band = tf.square((yp - yt) / (yt + eps))
        per_sample_rrmse = tf.sqrt(tf.reduce_mean(per_band, axis=1))  # (B,)
        super().update_state(per_sample_rrmse, sample_weight=sample_weight)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"sigma": self._sigma_init, "eps": self._eps_init})
        return cfg


class SigmaLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for name in ['sif_gauss','r_gauss']:
            s = tf.nn.softplus(self.model.get_layer(name).log_sp_sigma).numpy()
            print(f"[epoch {epoch+1}] {name}.sigma = {s:.3f}")


def train_cnn_model():
    import datetime
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                # tf.config.optimizer.set_jit(True)
                tf.config.experimental.set_memory_growth(gpus[0], True)
                tf.config.threading.set_inter_op_parallelism_threads(2)
                tf.config.threading.set_intra_op_parallelism_threads(8)
                print("GPU set successfully!")
            except RuntimeError as e:
                print(e)


        #Set TensorBoard callback
        log_dir = r"logs/fit" "//" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir,
                                           histogram_freq=1,  #Enable weight histogram
                                           # write_grads=True,
                                           # profile_batch='50,60', #Disable performance analysis
                                           write_graph=True,  # Enable model graph
                                           write_images=True,  # Save model weights as images
                                           # update_freq='epoch',  # Write to the log every epoch
                                           # embeddings_freq=5, # Enable visualization of embedding layers (if any)
                                           )

        checkpoint_save_best_path = './model/FSIF-PINN_model_v1.h5'
        cp_callback_save_best = ModelCheckpoint(filepath=checkpoint_save_best_path, save_weights_only=False,
                                                save_best_only=True, monitor='val_loss', mode='min', verbose=2)

        base = Path("/mnt/plant/aMLA-FULL/Train_MYA_SCLN3-BG/tfrecords_No_R")

        global_mean = load(base / "mean_new-TF-BG16.txt").astype(np.float32)
        global_var  = load(base / "var_new-TF-BG16.txt").astype(np.float32)

        global_meanl = load(base / "mean_new-TF-label-R-BG16.txt").astype(np.float32)
        global_varl  = load(base / "var_new-TF-label-R-BG16.txt").astype(np.float32)

        global_mean1 = tf.constant(global_mean, dtype=tf.float32)             # shape=(2970,)
        global_var1  = tf.constant(global_var,  dtype=tf.float32)             # shape=(2970,)


        sigmaF = np.maximum(np.sqrt(global_varl[:990] ), 0.1).astype(np.float32)
        sigmaR = np.maximum(np.sqrt(global_varl[-990:]), 0.01).astype(np.float32)


        batch_size = 2048
        lr_logger = LearningRateLogger()

        # 2) Create training/validation dataset
        nff=16
        train_dataset = create_cnn_dataset(
            tfrecord_dir=base / "Training",
            global_mean=global_mean1,
            global_var=global_var1,global_mean1=global_meanl, global_var1=global_varl,sigmaF=sigmaF,sigmaR=sigmaR,
            # global_mean1=global_mean1,
            # global_var1=global_var1,
            batch_size=batch_size,
            shuffle=True,
            num_files=nff,rp=True
        )
        
        val_dataset = create_cnn_dataset(
            tfrecord_dir=base / "Validation",
            global_mean=global_mean1,
            global_var=global_var1,global_mean1=global_meanl, global_var1=global_varl,sigmaF=sigmaF,sigmaR=sigmaR,
            # global_mean1=global_mean1,
            # global_var1=global_var1,
            batch_size=batch_size,
            shuffle=False,
            num_files=2
        )

        # test
        for  spec, label in train_dataset.take(1):
            # print("tower_id shape:", tid.shape)     # (batch,)
            print("spectral shape:", spec.shape)    # (batch, 990, 3)
            # print("label shape:", label.shape)      # (batch, 1980)
            print({k:v.shape for k,v in label.items()})

        def create_enhanced_lr_schedule(total_epochs=200, lr_max=0.0001, lr_min=1e-8,
                                        warmup_epochs=10, cycle_length=30):
            def lr_schedule(epoch, current_lr):
                # warm up
                # if epoch < warmup_epochs:
                #     return lr_max * (epoch + 1) / warmup_epochs
                if epoch <= warmup_epochs:
                    return lr_min + (lr_max - lr_min) * (epoch / warmup_epochs) ** 0.8

                after_warmup = epoch - warmup_epochs
                cycle_idx = (after_warmup) // cycle_length

                cycle_epoch = after_warmup % cycle_length
                cosine_decay = 0.5 * (1 + np.cos(np.pi * cycle_epoch / cycle_length))

                # Lr dacay
                decayed_lr = lr_min + (lr_max - lr_min) * cosine_decay

                return decayed_lr * (0.2 ** cycle_idx)

            return lr_schedule

        Elr_cos = tf.keras.callbacks.LearningRateScheduler(
            create_enhanced_lr_schedule(
                lr_max=0.001,  # The initial learning rate should match the optimizer settings.
                warmup_epochs=10,  # Increase warmup time appropriately
                cycle_length=80  # Ensure a sufficient descent period
            )
        )


        model = create_advanced_model(seq_len=990, in_channels=3, 
                                      global_mean=global_mean, global_var=global_var,global_mean1=global_meanl, global_var1=global_varl,sigmaF=sigmaF,sigmaR=sigmaR)

        # 5) Start training
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1000,
            callbacks=[tensorboard_callback, cp_callback_save_best,lr_logger,Elr_cos,SigmaLogger()],#lam_cb,
            verbose=2,
        )

        # Save the final model
        model.save("FSIF-PINN_v1.h5")
        print("Training Done. Save model")

    except Exception as e:
        import traceback, sys
        print(f"An error occurred during training: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    train_cnn_model()