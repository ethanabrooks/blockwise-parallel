from multiprocessing import Process, Queue
import numpy as np


def layer_norm(x: np.ndarray):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance)


def relu(x: np.ndarray):
    return np.maximum(0, x)


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    return np.einsum("bqd,dw -> bqw", x, w) + b[None, None]


def postprocess(
    x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray
):
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


def start_host(
    index: int,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    primary: Queue,
    input_queue: Queue,
    output_queue: Queue,
    n: int,
    **kwargs
):
    b, s, d = q.shape
    num = np.zeros((b, s, d))  # initialize numerator
    den = np.zeros((b, s))  # initialize denominator
    max_i = -np.inf * np.ones((b, s))  # initialize max_i

    for _ in range(n):
        k, v = input_queue.get()  # Receive k, v from the previous host
        output_queue.put((k, v))  # Send k, v to the next host
        assert k.shape == (b, s, d)
        assert v.shape == (b, s, d)
        alpha = np.einsum("bqd,bkd -> bqk", q, k)  # q^T K
        prev = max_i
        max_i = np.maximum(alpha.max(-1), max_i)  # update max_i
        exp_values = np.einsum(
            "bqk,bkd -> bqd", np.exp(alpha - max_i[..., None]), v
        )  # e^{alpha - max_i}^T v

        # update numerator and denominator
        num = num * np.exp(prev - max_i)[..., None] + exp_values
        den = den * np.exp(prev - max_i) + np.exp(alpha - max_i[..., None]).sum(-1)

    x = num / den[..., None]
    x = postprocess(x, **kwargs)
    primary.put((index, x))


def ring_transformer(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n: int, **kwargs):
    primary = Queue()
    num_hosts = len(Q)
    queues = [Queue() for _ in range(num_hosts)]
    processes = []

    # Create processes
    for i, (q, k, v) in enumerate(zip(Q, K, V)):
        input_queue = queues[i - 1]  # Previous host queue
        output_queue = queues[i]  # Current host queue
        process = Process(
            target=start_host,
            args=(i, q, k, v, primary, input_queue, output_queue, n),
            kwargs=kwargs,
        )
        processes.append(process)

    # Start processes
    for process in processes:
        process.start()

    # Send initial messages to start the communication
    for queue, k, v in zip(queues, K, V):
        queue.put((k, v))

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Collect outputs
    outputs = sorted([primary.get() for _ in range(num_hosts)])
    return np.stack([x for _, x in outputs])
