from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue
from collections import deque
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
Q = np.ones((n, b, s, d)) * np.arange(n)[:, None, None, None]
K = np.ones((n, b, s, d)) * np.arange(n)[:, None, None, None]
V = np.ones((n, b, s, d)) * np.arange(n)[:, None, None, None]
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


def postprocess(x: np.ndarray):
    x0 = x
    x = layer_norm(x)

    # 2-layer feedforward network
    x = linear(x, w1, b1)
    x = relu(x)
    x = linear(x, w2, b2)

    # residual connection + layer normalization
    x = x0 + x
    x = layer_norm(x)
    return x


def blockwise_parallel_transformer():
    outputs = []

    q: np.ndarray
    for i, q in enumerate(Q):
        assert list(q.shape) == [b, s, d]
        num = np.zeros((b, s, d))  # initialize numerator
        den = np.zeros((b, s))  # initialize denominator
        max_i = -np.inf * np.ones((b, s))  # initialize max_i

        k: np.ndarray
        v: np.ndarray
        for j in [2, 1, 0]:
            k = K[j]
            v = V[j]
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

        x = num / den[..., None]
        x = postprocess(x)
        outputs.append(x)

    return np.stack(outputs)


def trad_transformer():
    Q1 = Q.transpose([1, 0, 2, 3]).reshape(b, -1, d)
    K1 = K.transpose([1, 0, 2, 3]).reshape(b, -1, d)
    V1 = V.transpose([1, 0, 2, 3]).reshape(b, -1, d)
    attn_weights: np.ndarray = softmax(np.einsum("bqd,bkd -> bqk", Q1, K1), -1)  # Q^T K
    assert list(attn_weights.shape) == [b, s * n, s * n]
    x = np.einsum("bqk,bkd -> bqd", attn_weights, V1)  # q^T K V
    x = postprocess(x)
    return x


def start_host_sync(
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

    x = num / den[..., None]
    x = postprocess(x)
    primary.put(x)
    yield None


def ring_transformer_sync():
    primary = Queue()
    generators = []
    for q, k, v in zip(Q, K, V):
        generators.append(start_host_sync(q, k, v, primary))

    msgs = deque([None for _ in generators], maxlen=n)
    for _ in range(n + 1):
        msgs.rotate(-1)
        msgs = deque([generator.send(msg) for generator, msg in zip(generators, msgs)])

    outputs = [primary.get() for _ in range(n)]
    return np.stack(outputs)


def start_host(index: int, q: np.ndarray, recv: Queue, send: Queue):
    print(f"Starting host {index}")
    b, s, d = q.shape
    num = np.zeros((b, s, d))  # initialize numerator
    den = np.zeros((b, s))  # initialize denominator
    max_i = -np.inf * np.ones((b, s))  # initialize max_i

    for _ in range(n):
        k: np.ndarray
        v: np.ndarray
        k, v = recv.get()  # Receive k, v from the input queue (previous host)
        assert k.shape == (b, s, d) and v.shape == (b, s, d)

        alpha: np.ndarray = np.einsum("bqd,bkd -> bqk", q, k)  # q^T K
        prev = max_i
        max_i = np.maximum(alpha.max(-1), max_i)  # update max_i
        exp_values = np.einsum("bqk,bkd -> bqd", np.exp(alpha - max_i[..., None]), v)

        num = num * np.exp(prev - max_i)[..., None] + exp_values
        den = den * np.exp(prev - max_i) + np.exp(alpha - max_i[..., None]).sum(-1)

        send.put((k, v))  # Send (k, v) to the output queue for the next host

    x = num / den[..., None]
    x = postprocess(x)
    print(f"Host {index} done")
    return index, x


def ring_transformer():
    num_hosts = len(Q)
    queues = [
        Queue() for _ in range(num_hosts + 1)
    ]  # Create queues for each host pair, plus one extra to complete the ring

    # Initialize the first set of (k, v) pairs in the queues
    for queue, k, v in zip(queues, K, V):
        queue.put((k, v))

    with ThreadPoolExecutor(max_workers=num_hosts) as executor:
        futures = [
            executor.submit(start_host, i, q, queues[i], queues[(i + 1) % num_hosts])
            for i, q in enumerate(Q)
        ]
        outputs = [future.result() for future in futures]

    # Ensure outputs are sorted by index to maintain deterministic order
    return np.stack([x for _, x in sorted(outputs)])


if __name__ == "__main__":
    attn_outputs1 = trad_transformer().reshape(b, n, s, d).transpose(1, 0, 2, 3)
    attn_outputs2 = ring_transformer_sync()
    attn_outputs3 = blockwise_parallel_transformer()
    attn_outputs4 = ring_transformer()
    assert np.allclose(attn_outputs1, attn_outputs2)
    assert np.allclose(attn_outputs2, attn_outputs3)
    assert np.allclose(attn_outputs3, attn_outputs4)
    print("Success! The computations are equivalent.")
    # -
