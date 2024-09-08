import jax
import jax.numpy as jnp

def check_jax_cuda_cudnn():
    # Check if CUDA is available
    cuda_available = jax.lib.xla_bridge.get_backend().platform == 'gpu'
    
    # Perform a simple operation to check if cuDNN is available (matrix multiplication)
    try:
        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (1000, 1000))
        b = jax.random.normal(key, (1000, 1000))
        c = jnp.dot(a, b)  # Should use cuDNN for this operation if available
        c.block_until_ready()  # Ensures the computation is completed
        cudnn_available = True
    except Exception as e:
        cudnn_available = False

    return cuda_available, cudnn_available

cuda_available, cudnn_available = check_jax_cuda_cudnn()
print(f"CUDA available: {cuda_available}")
print(f"cuDNN available: {cudnn_available}")

