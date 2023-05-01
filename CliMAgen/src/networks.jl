"""
    NoiseConditionalScoreNetwork

The struct containing the parameters and layers
of the Noise Conditional Score Network architecture,
with the option to include a mean-bypass layer.

# References
Unet: https://arxiv.org/abs/1505.04597
"""
struct NoiseConditionalScoreNetwork
    "The layers of the network"
    layers::NamedTuple
    "A boolean indicating if non-noised context channels are present"
    context::Bool
    "A boolean indicating if a mean-bypass layer should be used"
    mean_bypass::Bool
    "A boolean indicating if the output of the mean-bypass layer should be scaled"
    scale_mean_bypass::Bool
    "A boolean indicating if the input is demeaned before being passed to the U-net"
    shift_input::Bool
    "A boolean indicating if the output of the Unet is demeaned"
    shift_output::Bool
    "A boolean indicating if a groupnorm should be used in the mean-bypass layer"
    gnorm::Bool
end

"""
    NoiseConditionalScoreNetwork(; context=false,
                                 mean_bypass=false, 
                                 scale_mean_bypass=false,
                                 shift_input=false,
                                 shift_output=false,
                                 gnorm=false,
                                 nspatial=2,
                                 dropout_p=0.0f0,
                                 num_residual=8,
                                 noised_channels=1,
                                 context_channels=0,
                                 channels=[32, 64, 128, 256],
                                 embed_dim=256,
                                 scale=30.0f0,
                                 proj_kernelsize=3,
                                 outer_kernelsize=3,
                                 middle_kernelsize=3,
                                 inner_kernelsize=3)

Returns a NoiseConditionalScoreNetwork, given
- context: boolean indicating whether or not contextual information is 
           present
- mean_bypass: boolean indicating if a mean-bypass layer should be used
- scale_mean_bypass: boolean indicating if the output of the mean-bypass 
                     layer should be scaled
- shift_input: boolean indicating if the input is demeaned before being 
               passed to the U-net
- shift_output: boolean indicating if the output of the Unet is demeaned
- gnorm: boolean indicating if a groupnorm should be used in the mean-bypass 
         layer
- nspatial: integer giving the number of spatial dimensions; images are assumed 
            to be square.
- dropout_p: float giving the dropout probability
- num_residual: integer giving the the number of residual blocks in the center of 
                the Unet
- noised_channels: integer giving the number of channels that are being noised
- context_channels: integer giving the number of context channels (not noised)
- channels: array of integers containing the number of channels for each layer of 
            the Unet during downsampling/upsampling
- embed_dim: integer of the time-embedding dimension
- scale: float giving the scale of the time-embedding layers
- proj_kernelsize: integer giving the kernel size in projection layers
- outer_kernelsize: integer giving the kernel size in the outermost down/upsample 
                    layers
- middle_kernelsize: integer giving the kernel size in the middle down/upsample 
                     layers
- inner_kernelsize: integer giving the kernel size in the innermost down/upsample 
                    layers
"""
function NoiseConditionalScoreNetwork(; context=false,
                                      mean_bypass=false, 
                                      scale_mean_bypass=false,
                                      shift_input=false,
                                      shift_output=false,
                                      gnorm=false,
                                      nspatial=2,
                                      dropout_p=0.0f0,
                                      num_residual=8,
                                      noised_channels=1,
                                      context_channels=0,
                                      channels=[32, 64, 128, 256],
                                      embed_dim=256,
                                      scale=30.0f0,
                                      proj_kernelsize=3,
                                      outer_kernelsize=3,
                                      middle_kernelsize=3,
                                      inner_kernelsize=3)
    if scale_mean_bypass & !mean_bypass
        @error("Attempting to scale the mean bypass term without adding in a mean bypass connection.")
    end
    if gnorm & !mean_bypass
        @error("Attempting to gnorm without adding in a mean bypass connection.")
    end
    if context & (context_channels == 0)
        @error("Attempting to use context-aware network without context input.")
    end
    if !context & (context_channels > 0)
        @error("Attempting to use context-unaware network with context input.")
    end
    
    inchannels = noised_channels+context_channels
    outchannels = noised_channels
    # Mean processing as indicated by boolean mean_bypass
    if mean_bypass
        if gnorm
            mean_bypass_layers = (
                mean_skip_1 = Conv((1, 1), inchannels => embed_dim),
                mean_skip_2 = Conv((1, 1), embed_dim => embed_dim),
                mean_skip_3 = Conv((1, 1), embed_dim => outchannels),
                mean_gnorm_1 = GroupNorm(embed_dim, 32, swish),
                mean_gnorm_2 = GroupNorm(embed_dim, 32, swish),
                mean_dense_1 = Dense(embed_dim, embed_dim),
                mean_dense_2 = Dense(embed_dim, embed_dim),
            )
        else
            mean_bypass_layers = (
                mean_skip_1 = Conv((1, 1), inchannels => embed_dim),
                mean_skip_2 = Conv((1, 1), embed_dim => embed_dim),
                mean_skip_3 = Conv((1, 1), embed_dim => outchannels),
                mean_dense_1 = Dense(embed_dim, embed_dim),
                mean_dense_2 = Dense(embed_dim, embed_dim),
            )
        end
    else
        mean_bypass_layers = ()
    end
    
    layers = (gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
              linear=Dense(embed_dim, embed_dim, swish),
              
              # Lifting
              conv1=Conv((3, 3), inchannels => channels[1], stride=1, pad=SamePad()),
              dense1=Dense(embed_dim, channels[1]),
              gnorm1=GroupNorm(channels[1], 4, swish),
              
              # Encoding
              conv2=Downsampling(channels[1] => channels[2], nspatial, kernel_size=3),
              dense2=Dense(embed_dim, channels[2]),
              gnorm2=GroupNorm(channels[2], 32, swish),
              
              conv3=Downsampling(channels[2] => channels[3], nspatial, kernel_size=3),
              dense3=Dense(embed_dim, channels[3]),
              gnorm3=GroupNorm(channels[3], 32, swish),
              
              conv4=Downsampling(channels[3] => channels[4], nspatial, kernel_size=3),
              dense4=Dense(embed_dim, channels[4]),
              
              # Residual Blocks
              resnet_blocks = 
              [ResnetBlockNCSN(channels[end], nspatial, embed_dim; p = dropout_p) for _ in range(1, length=num_residual)],
              
              # Decoding
              gnorm4=GroupNorm(channels[4], 32, swish),
              tconv4=Upsampling(channels[4] => channels[3], nspatial, kernel_size=inner_kernelsize),
              denset4=Dense(embed_dim, channels[3]),
              tgnorm4=GroupNorm(channels[3], 32, swish),
              
              tconv3=Upsampling(channels[3]+channels[3] => channels[2], nspatial, kernel_size=middle_kernelsize),
              denset3=Dense(embed_dim, channels[2]),
              tgnorm3=GroupNorm(channels[2], 32, swish),
              
              tconv2=Upsampling(channels[2]+channels[2] => channels[1], nspatial, kernel_size=outer_kernelsize),
              denset2=Dense(embed_dim, channels[1]),
              tgnorm2=GroupNorm(channels[1], 32, swish),
              
              # Projection
              tconv1=Conv((proj_kernelsize, proj_kernelsize), channels[1] + channels[1] => outchannels, stride=1, pad=SamePad()),
              mean_bypass_layers...
              )
    
    return NoiseConditionalScoreNetwork(layers, context, mean_bypass, scale_mean_bypass, shift_input, shift_output, gnorm)
end

@functor NoiseConditionalScoreNetwork

"""
    (net::NoiseConditionalScoreNetwork)(x, c, t)

Evaluates the neural network of the NoiseConditionalScoreNetwork
model on (x,c,t), where `x` is the tensor of noised input,
`c` is the tensor of contextual input, and `t` is a tensor of times.
"""
function (net::NoiseConditionalScoreNetwork)(x, c, t)
    # Embedding
    embed = net.layers.gaussfourierproj(t)
    embed = net.layers.linear(embed)

    # Encoder
    if net.shift_input
        h1 = x .- mean(x, dims=(1,2)) # remove mean of noised variables before input
    else
        h1 = x
    end
    h1 = concatenate_channels(Val(net.context), h1, c)
    h1 = net.layers.conv1(h1)
    h1 = h1 .+ expand_dims(net.layers.dense1(embed), 2)
    h1 = net.layers.gnorm1(h1)
    h2 = net.layers.conv2(h1)
    h2 = h2 .+ expand_dims(net.layers.dense2(embed), 2)
    h2 = net.layers.gnorm2(h2)
    h3 = net.layers.conv3(h2)
    h3 = h3 .+ expand_dims(net.layers.dense3(embed), 2)
    h3 = net.layers.gnorm3(h3)
    h4 = net.layers.conv4(h3)
    h4 = h4 .+ expand_dims(net.layers.dense4(embed), 2)

    # middle
    h = h4
    for block in net.layers.resnet_blocks
        h = block(h, embed)
    end

    # Decoder
    h = net.layers.gnorm4(h)
    h = net.layers.tconv4(h)
    h = h .+ expand_dims(net.layers.denset4(embed), 2)
    h = net.layers.tgnorm4(h)
    h = net.layers.tconv3(cat(h, h3; dims=3))
    h = h .+ expand_dims(net.layers.denset3(embed), 2)
    h = net.layers.tgnorm3(h)
    h = net.layers.tconv2(cat(h, h2, dims=3))
    h = h .+ expand_dims(net.layers.denset2(embed), 2)
    h = net.layers.tgnorm2(h)
    h = net.layers.tconv1(cat(h, h1, dims=3))
    if net.shift_output
        h = h .- mean(h, dims=(1,2)) # remove mean after output
    end

    # Mean processing of noised variable channels
    if net.mean_bypass
        hm = net.layers.mean_skip_1(mean(concatenate_channels(Val(net.context), x, c), dims=(1,2)))
        hm = hm .+ expand_dims(net.layers.mean_dense_1(embed), 2)
        if net.gnorm
            hm = net.layers.mean_gnorm_1(hm)
        end
        hm = net.layers.mean_skip_2(hm)
        hm = hm .+ expand_dims(net.layers.mean_dense_2(embed), 2)
        if net.gnorm
            hm = net.layers.mean_gnorm_2(hm)
        end
        hm = net.layers.mean_skip_3(hm)
        if net.scale_mean_bypass
            scale = convert(eltype(x), sqrt(prod(size(x)[1:ndims(x)-2])))
            hm = hm ./ scale
        end
        # Add back in noised channel mean to noised channel spatial variatons
        return h .+ hm
    else
        return h
    end
end

"""
    concatenate_channels(context::Val{true}, x, c)

Concatenates the context channels `c` with the noised data
channels `x` if `context` is true.
"""
function concatenate_channels(context::Val{true}, x, c)
    return cat(x, c, dims = 3)
end

"""
    concatenate_channels(context::Val{false}, x, c)

Returns `x` if `context` is false.
"""
function concatenate_channels(context::Val{false}, x, c)
    return x
end

"""
    GaussianFourierProjection{FT}

Concrete type used in the Gaussian Fourier Projection method
of embedding a continuous time variable.
"""
struct GaussianFourierProjection{FT}
    "Array used to scale and embed the time variable."
    W::AbstractArray{FT}
end

"""
    GaussianFourierProjection(embed_dim::Int, scale::FT) where {FT}

Outer constructor for the GaussianFourierProjection.

W is not trainable and is sampled once upon construction.
"""
function GaussianFourierProjection(embed_dim::Int, scale::FT) where {FT}
    W = randn(FT, embed_dim ÷ 2) .* scale
    return GaussianFourierProjection(W)
end

@functor GaussianFourierProjection

"""
    (gfp::GaussianFourierProjection{FT})(t) where {FT}


Embeds a continuous time `t`  into a periodic domain
using a random vector of Gaussian noise `gfp.W`.

# References
https://arxiv.org/abs/2006.10739
"""
function (gfp::GaussianFourierProjection{FT})(t) where {FT}
    t_proj = t' .* gfp.W .* FT(2π)
    return [sin.(t_proj); cos.(t_proj)]
end

"""
    Flux.params(::GaussianFourierProjection)

Returns the trainable parameters of the GaussianFourierProjection,
which are `nothing`.
"""
Flux.params(::GaussianFourierProjection) = nothing

"""
    CliMAgen.ResnetBlockNCSN

Struct holding the layers of the ResNet block 
used in the NoiseConditionalScoreNetwork model,
using GroupNorm and GaussianFourierProjection.

References:
https://arxiv.org/abs/1505.04597
https://arxiv.org/abs/1712.09763
"""
struct ResnetBlockNCSN
    "The group-normalization layer for the input"
    norm1
    "The first convolutional layer"
    conv1
    "The second group-normalization layer"
    norm2
    "The second convolutional layer"
    conv2
    "A dense layer for handling the time variable"
    dense
    "A dropout layer"
    dropout
end
"""
     ResnetBlockNCSN(channels::Int, nspatial::Int, nembed::Int; p=0.1f0, σ=Flux.swish)

Constructor for the ResnetBlockNCSN, which preserves the `channel` number and image
size of the input.

Here, `nspatial` is the number of spatial dimensions, `nembed` is the embedding
size used in the GaussianFourierProjection,
`p` is the dropout probability, and `σ` is the nonlinearity used in the group
norms and the dense layer.
"""
function ResnetBlockNCSN(channels::Int, nspatial::Int, nembed::Int; p=0.1f0, σ=Flux.swish)
    # channels needs to be larger than 4 for group norms
    @assert channels ÷ 4 > 0

    # Require same input and output spatial size
    pad = SamePad()

    return ResnetBlockNCSN(
        GroupNorm(channels, min(channels ÷ 4, 32), σ),
        Conv(Tuple(3 for _ in 1:nspatial), channels => channels, pad=pad),
        GroupNorm(channels, min(channels ÷ 4, 32), σ),
        Conv(Tuple(3 for _ in 1:nspatial), channels => channels, pad=pad),
        Dense(nembed => channels, σ),
        Dropout(p),
    )
end

@functor ResnetBlockNCSN

"""
   (net::ResnetBlockNCSN)(x, tembed)

Applies the ResnetBlockNCSN to (x,tembed).
"""
function (net::ResnetBlockNCSN)(x, tembed)
    # add on temporal embeddings to condition on time
    h = net.norm1(x)
    h = net.conv1(h) .+ expand_dims(net.dense(tembed), 2)

    # dropout is needed for low complexity datasets to
    # avoid overfitting
    h = net.norm2(h)
    h = net.dropout(h)
    h = net.conv2(h)

    return h .+ x
end

"""
    CliMAgen.Downsampling(channels::Pair, nspatial::Int; factor::Int=2, kernel_size::Int=3)

Creates a downsampling layer using convolutional kernels.

Here, 
- `channels = inchannels => outchannels` is the pair of incoming and outgoing channels,
- `nspatial` is the number of spatial dimensions of the image,
- `factor` indicates the downsampling factor, and 
- `kernel_size` is in the kernel size.
"""
function Downsampling(channels::Pair, nspatial::Int; factor::Int=2, kernel_size::Int=3)
    conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
    return Conv(conv_kernel, channels, stride=factor, pad=SamePad())
end

"""
    CliMAgen.Upsampling(channels::Pair, nspatial::Int; factor::Int=2, kernel_size::Int=3)

Creates an upsampling layer using nearest-neighbor interpolation and 
convolutional kernels, so that checkerboard artifacts are avoided.

Here, 
- `channels = inchannels => outchannels` is the pair of incoming and outgoing channels,
- `nspatial` is the number of spatial dimensions of the image,
- `factor` indicates the downsampling factor, and 
- `kernel_size` is in the kernel size.

References:
https://distill.pub/2016/deconv-checkerboard/
"""
function Upsampling(channels::Pair, nspatial::Int; factor::Int=2, kernel_size::Int=3)
    conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
    return Chain(
        Flux.Upsample(factor, :nearest),
        Conv(conv_kernel, channels, pad=SamePad())
    )
end
