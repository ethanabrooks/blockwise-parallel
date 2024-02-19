from multiprocessing import Queue
import numpy as np
from scipy.special import softmax

np.random.seed(0)
# + id="lgYpN68O2Jvu"
num0 = 0
den0 = 0
max_i0 = -np.inf
n = 3  # number of chunks
b = 2  # batch dimension (could also include head dimension, since heads are parallel for self-attention)
s = 7
d = 5
Q = np.random.random((n, b, s, d))
K = np.random.random((n, b, s, d))
V = np.random.random((n, b, s, d))


def blockwise_parallel_transformer():
    attn_outputs = []

    q: np.ndarray
    for i, q in enumerate(Q):
        assert list(q.shape) == [b, s, d]
        num = np.zeros((b, s, d))  # initialize numerator
        den = np.zeros((b, s))  # initialize denominator
        max_i = -np.inf * np.ones((b, s))  # initialize max_i

        k: np.ndarray
        v: np.ndarray
        for j, (k, v) in enumerate(zip(K, V)):
            assert list(k.shape) == [b, s, d]
            assert list(v.shape) == [b, s, d]
            alpha: np.ndarray = np.einsum("bqd,bkd -> bqk", q, k)  # q^T K
            prev = max_i
            max_i = np.maximum(alpha.max(-1), max_i)  # update max_i
            exp_values = np.einsum(
                "bqk,bkd -> bqd", np.exp(alpha - max_i[..., None]), v
            )  # e^{alpha - max_i}^T v

            # update numerator and denominator
            num = num * np.exp(prev - max_i)[..., None] + exp_values
            den = den * np.exp(prev - max_i) + np.exp(alpha - max_i[..., None]).sum(-1)

        attn_outputs.append(num / den[..., None])

    return np.stack(attn_outputs)


def trad_transformer():
    Q1 = Q.transpose([1, 0, 2, 3]).reshape(b, -1, d)
    K1 = K.transpose([1, 0, 2, 3]).reshape(b, -1, d)
    V1 = V.transpose([1, 0, 2, 3]).reshape(b, -1, d)
    attn_weights: np.ndarray = softmax(np.einsum("bqd,bkd -> bqk", Q1, K1), -1)  # Q^T K
    assert list(attn_weights.shape) == [b, s * n, s * n]
    return np.einsum("bqk,bkd -> bqd", attn_weights, V1)  # q^T K V


def start_host(
    i,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    primary: Queue,
):
    assert list(q.shape) == [b, s, d]
    num = np.zeros((b, s, d))  # initialize numerator
    den = np.zeros((b, s))  # initialize denominator
    max_i = -np.inf * np.ones((b, s))  # initialize max_i

    k: np.ndarray
    v: np.ndarray
    for _ in range(n):
        assert list(k.shape) == [b, s, d]
        assert list(v.shape) == [b, s, d]
        alpha: np.ndarray = np.einsum("bqd,bkd -> bqk", q, k)  # q^T K
        prev = max_i
        max_i = np.maximum(alpha.max(-1), max_i)  # update max_i
        exp_values = np.einsum(
            "bqk,bkd -> bqd", np.exp(alpha - max_i[..., None]), v
        )  # e^{alpha - max_i}^T v

        # update numerator and denominator
        num = num * np.exp(prev - max_i)[..., None] + exp_values
        den = den * np.exp(prev - max_i) + np.exp(alpha - max_i[..., None]).sum(-1)
        (k, v) = yield (k, v)

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
    for i, (q, k, v) in enumerate(zip(Q, K, V)):
        generators.append(start_host(i, q, k, v, primary))

    msgs = [None for _ in generators]
    for _ in range(n):
        for i, (generator, msg) in enumerate(zip(generators, msgs)):
            msg = generator.send(msg)
            msgs[i] = msg
    for i, (generator, msg) in enumerate(zip(generators, msgs)):
        generator.send(msg)

    outputs = [primary.get() for _ in range(n)]
    return np.stack(outputs)


if __name__ == "__main__":
    attn_outputs = blockwise_parallel_transformer()
    # attn_outputs = attn_outputs.transpose(1, 0, 2, 3).reshape(
    #     b, n * s, d
    # )  # merge blocks for comparison
    attn_outputs2 = trad_transformer().reshape(b, n, s, d).transpose(1, 0, 2, 3)
    # attn_outputs2 = ring_transformer()
    assert np.allclose(attn_outputs, attn_outputs2)
    print("Success! The two computations are equivalent.")
    # -
