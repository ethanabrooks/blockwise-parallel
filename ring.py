import numpy as np
from multiprocessing import Queue
from scipy.special import softmax

n = 3  # number of chunks
b = 2  # batch dimension (could also include head dimension, since heads are parallel for self-attention)
s = 7
d = 5
Q = np.random.random((n, b, s, d))
K = np.random.random((n, b, s, d))
V = np.random.random((n, b, s, d))
w1 = np.random.standard_normal((d, d))
b1 = np.random.standard_normal(d)
w2 = np.random.standard_normal((d, d))
b2 = np.random.standard_normal(d)


def layer_norm(x: np.ndarray):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance)


def relu(x: np.ndarray):
    return np.maximum(0, x)


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    return np.einsum("bqd,dw -> bqw", x, w) + b[None, None]


def start_host(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    primary: Queue,
):
    assert list(q.shape) == [b, s, d]
    assert list(k.shape) == [b, s, d]
    assert list(v.shape) == [b, s, d]
    alpha: np.ndarray = np.einsum("bqd,bkd -> bqk", q, k)  # q^T K
    max_i = alpha.max(-1)
    num = np.einsum(
        "bqk,bkd -> bqd", np.exp(alpha - max_i[..., None]), v
    )  # e^{alpha - max_i}^T v
    den = np.exp(alpha - max_i[..., None]).sum(-1)

    for _ in range(n):
        prev = max_i
        (max_i, exp_values, exp_weights) = yield (max_i, num, den)
        max_i = np.maximum(max_i, prev)  # update max_i
        correction = np.exp(prev - max_i)
        breakpoint()
        num = num * correction[..., None] + exp_values
        den = den * correction + exp_weights

    chunk_attn_output = num / den[..., None]
    x = chunk_attn_output

    ###################################### NEW CODE ######################################
    # x = layer_norm(chunk_attn_output)

    # # 2-layer feedforward network
    # x = linear(x, w1, b1)
    # x = relu(x)
    # x = linear(x, w2, b2)

    # # residual connection + layer normalization
    # x = chunk_attn_output + x
    # x = layer_norm(x)
    #################################### END NEW CODE ####################################

    primary.put(x)
    yield None


def ring_transformer():
    primary = Queue()
    generators = []
    for q, k, v in zip(Q, K, V):
        generators.append(start_host(q, k, v, primary))

    msgs = [None for _ in generators]
    for _ in range(n):
        for i, (generator, msg) in enumerate(zip(generators, msgs)):
            msg = generator.send(msg)
            msgs[i] = msg
    for i, (generator, msg) in enumerate(zip(generators, msgs)):
        generator.send(msg)

    outputs = [primary.get() for _ in range(n)]
    return np.stack(outputs)


def trad_transformer():
    Q1 = Q.transpose([1, 0, 2, 3]).reshape(b, -1, d)
    K1 = K.transpose([1, 0, 2, 3]).reshape(b, -1, d)
    V1 = V.transpose([1, 0, 2, 3]).reshape(b, -1, d)
    attn_weights: np.ndarray = softmax(np.einsum("bqd,bkd -> bqk", Q1, K1), -1)  # Q^T K
    assert list(attn_weights.shape) == [b, s * n, s * n]
    attn_outputs2 = np.einsum("bqk,bkd -> bqd", attn_weights, V1)  # q^T K V

    x = attn_outputs2
    # x = layer_norm(attn_outputs2)

    # # 2-layer feedforward network
    # x = linear(x, w1, b1)
    # x = relu(x)
    # x = linear(x, w2, b2)

    # # residual connection
    # x = attn_outputs2 + x
    # x = layer_norm(x)
    return x


def main():
    outputs = ring_transformer()
    outputs = outputs.transpose(1, 0, 2, 3).reshape(
        b, n * s, d
    )  # merge blocks for comparison
    outputs2 = trad_transformer()

    print(outputs[0])
    print(outputs2[0])
    # assert np.allclose(outputs, outputs2)


if __name__ == "__main__":
    main()
