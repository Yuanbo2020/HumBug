import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.config as config


def move_data_to_gpu(x, cuda, using_float=False):
    if using_float:
        x = torch.Tensor(x)
    else:
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)

        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)

        else:
            raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x

#####################################################################################################################
from torch import hub
import numpy as np
from framework import vggish_params
from framework import vggish_input


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)  # torch.Size([32, 512, 6, 4])
        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        # print(x.shape)  # torch.Size([32, 4, 6, 512])
        x = torch.transpose(x, 1, 2)
        # print(x.shape)  # torch.Size([32, 6, 4, 512])
        x = x.contiguous()
        # print(x.shape)   # torch.Size([32, 6, 4, 512])
        x = x.view(x.size(0), -1)
        # print(x.shape)  # torch.Size([32, 12288])
        """
        由于这里是 12288，所以初始化的时候不能选择skip layer，要从0开始
        self.vggish.embeddings = nn.Sequential(*(list(self.vggish.embeddings.children())[0:]))  # skip layers
        """
        return self.embeddings(x)


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


class VGGish(VGG):
    def __init__(self, urls, device=None, pretrained=True, preprocess=True, postprocess=True, progress=True):
        super().__init__(make_layers())
        if pretrained:
            state_dict = hub.load_state_dict_from_url(urls['vggish'], progress=progress)
            super().load_state_dict(state_dict)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.preprocess = preprocess
        self.postprocess = postprocess
        if self.postprocess:
            self.pproc = Postprocessor()
            if pretrained:
                state_dict = hub.load_state_dict_from_url(urls['pca'], progress=progress)
                # TODO: Convert the state_dict to torch
                state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float
                )
                state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
                )

                self.pproc.load_state_dict(state_dict)
        self.to(self.device)

    def forward(self, x, fs=None):
        if self.preprocess:
            x = self._preprocess(x, fs)
        # x = x.to(self.device)
        x = VGG.forward(self, x)
        if self.postprocess:
            x = self._postprocess(x)
        return x

    def _preprocess(self, x, fs):
        if isinstance(x, np.ndarray):
            x = vggish_input.waveform_to_examples(x, fs)
        elif isinstance(x, str):
            x = vggish_input.wavfile_to_examples(x)
        else:
            raise AttributeError
        return x

    def _postprocess(self, x):
        return self.pproc(x)


class VGGishDropoutFeatB(nn.Module):
    def __init__(self, preprocess=False, dropout=0.2, MC_dropout=True):
        super(VGGishDropoutFeatB, self).__init__()
        self.model_urls = config.vggish_model_urls
        self.vggish = VGGish(self.model_urls, pretrained=config.pretrained, postprocess=False, preprocess=preprocess)
        # self.vggish = nn.Sequential(*(list(self.vggish.children())[2:])) # skip layers

        # self.vggish.embeddings = nn.Sequential(*(list(self.vggish.embeddings.children())[2:]))  # skip layers
        # 2025-1-13 自己修改
        self.vggish.embeddings = nn.Sequential(*(list(self.vggish.embeddings.children())[0:]))  # skip layers

        self.MC_dropout = MC_dropout

        self.dropout = dropout
        # self.dropout = nn.Dropout(dropout)  # hyb 自己实现的，这样的话，model.eval()时，输出都是固定的
        self.n_channels = 1  # For building data correctly with dataloaders
        self.fc1 = nn.Linear(128, 1)  # for multiclass
        # For application to embeddings, see:
        # https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_train_demo.py

    def forward(self, x):
        x = x.view(-1, 1, config.win_size, config.mel_bins)   # Feat B
        # print(x.size())  # torch.Size([32, 1, 100, 64])
        x = self.vggish.forward(x)
        # print(x.size())
        # x = self.fc1(F.dropout(x, p=self.dropout))  # 这里的training=True,是一直开着的，与model.eval()无关

        if self.MC_dropout:
            x = self.fc1(F.dropout(x, p=self.dropout))
        else:
            x = self.fc1(F.dropout(x, p=self.dropout, training=self.training))

        """
        # x = self.fc1(self.dropout(x))  # hyb 自己实现的，这样的话，model.eval()时，输出都是固定的
        https://zhuanlan.zhihu.com/p/575456981
        推荐使用 nn.Dropout。因为一般只有训练时才使用 Dropout，在验证或测试时不需要使用 Dropout。
        使用 nn.Dropout时，如果调用 model.eval() ，模型的 Dropout 层都会关闭；
        但如果使用 nn.functional.dropout，在调用 model.eval() 时，不会关闭 Dropout。----以此来实现MC dropout
        即使要用F.dropout，也要加 self.training:  embedding = F.dropout(x, p=0.5, training=self.training)

        p为对于input中各个元素zero out的概率，也就是说当p=1时，output为全0。
        inplace参数，表示是否对tensor本身操作，若选择True,将会设置tensor为0。
        """
        x = torch.sigmoid(x)
        return x




############################################################################################################
def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class PANN(nn.Module):
    def __init__(self, class_num, dropout, MC_dropout, batchnormal=False):

        super(PANN, self).__init__()

        self.MC_dropout = MC_dropout

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)
            # self.bn0_loudness = nn.BatchNorm2d(1)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.dropout = dropout

        last_units = 2048
        self.fc1 = nn.Linear(last_units, last_units, bias=True)

        # # ------------------- classification layer -----------------------------------------------------------------
        self.fc_final_event = nn.Linear(last_units, class_num, bias=True)
        # ##############################################################################################################

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input):
        # print(input.shape) # torch.Size([32, 1, 100, 64])

        # (_, seq_len, mel_bins) = input.shape
        # x = input.view(-1, 1, seq_len, mel_bins)
        # '''(samples_num, feature_maps, time_steps, freq_num)'''

        x = input
        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # -------------------------------------------------------------------------------------------------------------
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([32, 64, 50, 32])
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([32, 128, 25, 16])
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([32, 256, 12, 8])
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([32, 512, 6, 4])
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([32, 1024, 3, 2])
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        # print('x.size():', x.size())  # x.size(): torch.Size([32, 2048, 3, 2])
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        # print('6x.size():', x.size())  # torch.Size([32, 2048, 3])

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2  # torch.Size([64, 2048])

        # x = F.dropout(x, p=0.5, training=self.training)
        common_embeddings = F.relu_(self.fc1(x))
        # clipwise_output = torch.sigmoid(self.fc_audioset(x))

        if self.MC_dropout:
            event = self.fc_final_event(F.dropout(common_embeddings, p=self.dropout))
        else:
            event = self.fc_final_event(F.dropout(common_embeddings, p=self.dropout, training=self.training))
        # 这里的training=True,是一直开着的，与model.eval()无关
        event = torch.sigmoid(event)

        return event




#################################### mha ########################################################
import numpy as np
# transformer
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention

class ScaledDotProductAttention_nomask(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_nomask, self).__init__()

    def forward(self, Q, K, V, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention_nomask(nn.Module):
    def __init__(self, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                 output_dim=d_model):
        super(MultiHeadAttention_nomask, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * d_v, output_dim)

    def forward(self, Q, K, V, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention_nomask()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, output_dim=d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_nomask(output_dim=output_dim)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim=d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(output_dim) for _ in range(n_layers)])
        self.mel_projection = nn.Linear(input_dim, d_model)

    def forward(self, enc_inputs):
        # print(enc_inputs.size())  # torch.Size([64, 54, 8, 8])
        size = enc_inputs.size()
        enc_inputs = enc_inputs.reshape(size[0], size[1], -1)
        enc_outputs = self.mel_projection(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
#################################################################################################


class CNN_Transformer(nn.Module):
    def __init__(self, class_num, dropout, MC_dropout):

        super(CNN_Transformer, self).__init__()

        self.dropout = dropout
        self.MC_dropout = MC_dropout

        out_channels = 64
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        out_channels = 128
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        out_channels = 256
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        d_model = 512
        self.mha = Encoder(input_dim=256, n_layers=1, output_dim=d_model)

        last_units = 512*12
        self.fc_final = nn.Linear(last_units, class_num, bias=True)

    def forward(self, input):
        # print(input.shape)  # torch.Size([32, 1, 100, 64])
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        # (_, seq_len, mel_bins) = input.shape
        # x = input.view(-1, 1, seq_len, mel_bins)
        # '''(samples_num, feature_maps, time_steps, freq_num)'''

        x = input
        # print(x.size())  # torch.Size([32, 1, 100, 64])
        x = F.relu_(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 4))
        # print(x.size())  # torch.Size([32, 64, 24, 20])

        x = F.relu_(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 4))
        # print(x.size())  # torch.Size([32, 128, 25, 4])

        x = F.relu_(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(2, 4))
        # print(x.size())  # torch.Size([32, 256, 12, 1])

        x = x.transpose(1, 2)  # torch.Size([64, 6, 256, 1])
        x, x_self_attns = self.mha(x)  # already have reshape
        # print(x.size())  # torch.Size([32, 12, 512])

        x = torch.flatten(x, start_dim=1)

        if self.MC_dropout:
            event = self.fc_final(F.dropout(x, p=self.dropout))
        else:
            event = self.fc_final(F.dropout(x, p=self.dropout, training=self.training))

        event = torch.sigmoid(event)

        return event



############################################################################
from framework.Yamnet_params import YAMNetParams

class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF Slim
    """
    def __init__(self, *args, **kwargs):
        # remove padding argument to avoid conflict
        padding = kwargs.pop("padding", "SAME")
        # initialize nn.Conv2d
        super().__init__(*args, **kwargs)
        self.padding = padding
        assert self.padding == "SAME"
        self.num_kernel_dims = 2
        self.forward_func = lambda input, padding: F.conv2d(
            input, self.weight, self.bias, self.stride,
            padding=padding, dilation=self.dilation, groups=self.groups,
        )

    def tf_SAME_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.kernel_size[dim]

        dilate = self.dilation
        dilate = dilate if isinstance(dilate, int) else dilate[dim]
        stride = self.stride
        stride = stride if isinstance(stride, int) else stride[dim]

        effective_kernel_size = (filter_size - 1) * dilate + 1
        out_size = (input_size + stride - 1) // stride
        total_padding = max(
            0, (out_size - 1) * stride + effective_kernel_size - input_size
        )
        total_odd = int(total_padding % 2 != 0)
        return total_odd, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return self.forward_func(input, padding=0)
        odd_1, padding_1 = self.tf_SAME_padding(input, dim=0)
        odd_2, padding_2 = self.tf_SAME_padding(input, dim=1)
        if odd_1 or odd_2:
            # NOTE: F.pad argument goes from last to first dim
            input = F.pad(input, [0, odd_2, 0, odd_1])

        return self.forward_func(
            input, padding=[ padding_1 // 2, padding_2 // 2 ]
        )


class CONV_BN_RELU(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(
            conv.out_channels, eps=YAMNetParams.BATCHNORM_EPSILON
        )  # NOTE: yamnet uses an eps of 1e-4. This causes a huge difference
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.fused = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False
            )
        )

    def forward(self, x):
        return self.fused(x)


class SeparableConv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.depthwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=input_dim, groups=input_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False,
            ),
        )
        self.pointwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=1, stride=1,
                padding='SAME', bias=False,
            ),
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class YAMNet(nn.Module):
    def __init__(self, class_num, dropout, MC_dropout):
        super().__init__()
        net_configs = [
            # (layer_function, kernel, stride, num_filters)
            (Conv, [3, 3], 2, 32),
            (SeparableConv, [3, 3], 1, 64),
            (SeparableConv, [3, 3], 2, 128),
            (SeparableConv, [3, 3], 1, 128),
            (SeparableConv, [3, 3], 2, 256),
            (SeparableConv, [3, 3], 1, 256),
            (SeparableConv, [3, 3], 2, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 1, 512),
            (SeparableConv, [3, 3], 2, 1024),
            (SeparableConv, [3, 3], 1, 1024)
        ]

        self.dropout = dropout
        self.MC_dropout = MC_dropout

        input_dim = 1
        self.layer_names = []
        for (i, (layer_mod, kernel, stride, output_dim)) in enumerate(net_configs):
            name = 'layer{}'.format(i + 1)
            self.add_module(name, layer_mod(kernel, stride, input_dim, output_dim))
            input_dim = output_dim
            self.layer_names.append(name)

        self.bn0 = nn.BatchNorm2d(config.mel_bins)

        last_units = 1024
        self.fc_final = nn.Linear(last_units, class_num, bias=True)

    def forward(self, x):
        # print(x.size())  # torch.Size([64, 1, 100, 64])

        for name in self.layer_names:
            mod = getattr(self, name)
            x = mod(x)
        x = F.adaptive_avg_pool2d(x, 1)
        # print(x.size())  # torch.Size([32, 1024, 1, 1])
        x = x.reshape(x.shape[0], -1)
        # print(x.size())  # torch.Size([32, 1024])

        if self.MC_dropout:
            event = self.fc_final(F.dropout(x, p=self.dropout))
        else:
            event = self.fc_final(F.dropout(x, p=self.dropout, training=self.training))
        # 这里的training=True,是一直开着的，与model.eval()无关
        event = torch.sigmoid(event)

        return event


################################################################################
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, class_num, dropout, MC_dropout):

        super(MobileNetV2, self).__init__()

        self.dropout = dropout
        self.MC_dropout = MC_dropout

        width_mult = 1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        d_embeddings = 1280
        self.fc_final = nn.Linear(d_embeddings, class_num, bias=True)


    def forward(self, input):
        """
        Input: (batch_size, data_length)"""
        # print(input.shape)  # torch.Size([64, 1, 480, 64])

        x = self.features(input)

        x = torch.mean(x, dim=3)

        x = torch.mean(x, dim=2)

        if self.MC_dropout:
            event = self.fc_final(F.dropout(x, p=self.dropout))
        else:
            event = self.fc_final(F.dropout(x, p=self.dropout, training=self.training))
        # 这里的training=True,是一直开着的，与model.eval()无关
        event = torch.sigmoid(event)

        return event


# ----------------------------------------------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x




class MTRCNN(nn.Module):
    def __init__(self, class_num, dropout, MC_dropout, batchnormal):

        super(MTRCNN, self).__init__()


        self.dropout = dropout
        self.MC_dropout = MC_dropout

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        frequency_num = 6
        frequency_emb_dim = 1
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1 = ConvBlock_single_layer(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, padding=(0,0), dilation=(2, 1))
        self.conv_block3 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, padding=(0,0), dilation=(3, 1))
        self.k_3_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 5
        kernel_size = (5, 5)
        self.conv_block1_kernel_5 = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0,2))
        self.conv_block2_kernel_5 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(2, 1))
        self.conv_block3_kernel_5 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(0, 1), dilation=(3, 1))
        self.k_5_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 7
        kernel_size = (7, 7)
        self.conv_block1_kernel_7 = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 3))
        self.conv_block2_kernel_7 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, kernel_size=kernel_size,
                                                       padding=(2, 2), dilation=(2, 1))
        self.conv_block3_kernel_7 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, kernel_size=kernel_size,
                                                       padding=(2, 2), dilation=(3, 1))
        self.k_7_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # # -------------- kernel 9
        # kernel_size = (9, 9)
        # self.conv_block1_kernel_9 = ConvBlock_single_layer(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=(0, 4))
        # self.conv_block2_kernel_9 = ConvBlock_dilation_single_layer(in_channels=16, out_channels=32, kernel_size=kernel_size,
        #                                                padding=(0, 3), dilation=(2, 1))
        # self.conv_block3_kernel_9 = ConvBlock_dilation_single_layer(in_channels=32, out_channels=64, kernel_size=kernel_size,
        #                                                padding=(0, 3), dilation=(3, 1))
        # self.k_9_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        ##################################### gnn ####################################################################


        # ----------------------------------------------------------------------------------------------------


        scene_event_embedding_dim = 128
        # embedding layers
        self.fc_embedding_event = nn.Linear(64*3, scene_event_embedding_dim, bias=True)
        # -----------------------------------------------------------------------------------------------------------

        # ------------------- classification layer -----------------------------------------------------------------
        # self.fc_final_arousal = nn.Linear(scene_event_embedding_dim, class_num_arousal, bias=True)
        #
        # self.fc_final_vanlence = nn.Linear(scene_event_embedding_dim, class_num_vanlence, bias=True)

        self.fc_final = nn.Linear(scene_event_embedding_dim, class_num, bias=True)

        ##############################################################################################################

        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)

        init_layer(self.fc_embedding_event)

        # # classification layer -------------------------------------------------------------------------------------
        # init_layer(self.fc_final_arousal)
        # init_layer(self.fc_final_vanlence)

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, x):
        # print(input.shape) # torch.Size([32, 1,  3001, 64])

        # (_, seq_len, mel_bins) = input.shape
        # x = input.view(-1, 1, seq_len, mel_bins)
        # '''(samples_num, feature_maps, time_steps, freq_num)'''


        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        batch_x = x

        # print(x.size())  #  torch.Size([64, 1, 100, 64])
        x_k_3 = self.conv_block1(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block2(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)

        x_k_3 = self.conv_block3(x_k_3, pool_size=(2, 2), pool_type='avg')
        x_k_3 = F.dropout(x_k_3, p=0.2, training=self.training)
        # print('x_k_3: ', x_k_3.size())  # x_k_3:  torch.Size([64, 64, 8, 6])

        x_k_3 = self.mean_max(x_k_3)
        # print('x_k_3: ', x_k_3.size())  # x_k_3:  torch.Size([32, 64, 1])
        x_k_3_mel = F.relu_(x_k_3[:, :, 0])
        # print('x_k_3_mel: ', x_k_3_mel.size())  # x_k_3_mel:  torch.Size([64, 64])

        # kernel 5 -----------------------------------------------------------------------------------------------------
        x_k_5 = self.conv_block1_kernel_5(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([64, 16, 48, 32])

        x_k_5 = self.conv_block2_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size())  # torch.Size([64, 32, 20, 15])

        x_k_5 = self.conv_block3_kernel_5(x_k_5, pool_size=(2, 2), pool_type='avg')
        x_k_5 = F.dropout(x_k_5, p=0.2, training=self.training)
        # print(x_k_5.size(), '\n')  # torch.Size([64, 64, 4, 6])

        x_k_5 = self.mean_max(x_k_5)  # torch.Size([32, 64, 1])
        x_k_5_mel = F.relu_(x_k_5[:, :, 0])
        # print('x_k_5_mel: ', x_k_5_mel.size())  # torch.Size([64, 64])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        x_k_7 = self.conv_block1_kernel_7(batch_x, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([64, 16, 47, 32])

        x_k_7 = self.conv_block2_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size())  # torch.Size([64, 32, 17, 15])

        x_k_7 = self.conv_block3_kernel_7(x_k_7, pool_size=(2, 2), pool_type='avg')
        x_k_7 = F.dropout(x_k_7, p=0.2, training=self.training)
        # print(x_k_7.size(), '\n')  # torch.Size([64, 64, 2, 6])

        x_k_7 = self.mean_max(x_k_7)
        # print(x_k_7.size(), '\n')  # torch.Size([32, 64, 1])
        x_k_7_mel = F.relu_(x_k_7[:, :, 0])
        # print('x_k_7_mel: ', x_k_7_mel.size())  #torch.Size([64, 64])

        # -------------------------------------------------------------------------------------------------------------
        event_embs_log_mel = torch.cat([x_k_3_mel, x_k_5_mel,
                                        x_k_7_mel], dim=-1)
        # print(event_embs_log_mel.size())  # torch.Size([64, 192])  (node_num, batch, edge_dim)

        # -------------------------------------------------------------------------------------------------------------
        event_embeddings = F.gelu(self.fc_embedding_event(event_embs_log_mel))
        # -------------------------------------------------------------------------------------------------------------

        if self.MC_dropout:
            event = self.fc_final(F.dropout(event_embeddings, p=self.dropout))
        else:
            event = self.fc_final(F.dropout(event_embeddings, p=self.dropout, training=self.training))
        # 这里的training=True,是一直开着的，与model.eval()无关
        event = torch.sigmoid(event)

        return event



