import os
import subprocess
import warnings
import tensorflow as tf

# Number the GPU's like nvidia-smi does (must be set before TensorFlow initializes)
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

def find_available(num_gpus: int, max_utilization: int, min_free_memory: int) -> list[int]:
    """Find GPU indices with sufficiently low utilization and sufficient free memory, via `nvidia-smi`

    Returns an empty list (run on the CPU) if `nvidia-smi` is not available.

    Args:
        num_gpus (int): number of available GPU's to find
        max_utilization (int): maximum utilization (%) for a GPU to be considered available
        min_free_memory (int): minimum free memory (MiB) for a GPU to be considered available

    Returns:
        list[int]: indices of the first `num_gpus` GPU's that satisfy the given constraints, in nvidia-smi order, among the GPU's in CUDA_VISIBLE_DEVICES if set
    """

    try:
        stats = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free', '--format=csv'])
    except (FileNotFoundError, subprocess.CalledProcessError):
        warnings.warn('nvidia-smi not available, running on the CPU')
        return []

    stats = stats.decode().splitlines()[1:]
    stats = [x.split(', ') for x in stats]
    stats = [[int(y[0][:-2]), int(y[1][:-4])] for y in stats]

    # Only consider the GPU's visible to TensorFlow, numbered as TensorFlow numbers them
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        stats = [stats[int(i)] for i in visible.split(',') if i.strip().isdigit()]
        if not stats:
            warnings.warn('No GPU visible (CUDA_VISIBLE_DEVICES), running on the CPU')
            return []

    selected_gpus = []

    for gpu_id, stat in enumerate(stats):
        if stat[0] <= max_utilization and stat[1] >= min_free_memory:
            selected_gpus.append(gpu_id)
            if len(selected_gpus) == num_gpus:
                break
    else:
        raise RuntimeError('Could not find available GPUs')

    return selected_gpus

def select_gpus(selected_gpus: list[int], memory_growth: bool=False):
    """Restrict TensorFlow to the given GPU's for this session

    Args:
        selected_gpus (list[int]): indices of GPU's to make visible to TensorFlow, in nvidia-smi order; an empty list runs on the CPU
        memory_growth (bool, optional): whether to allow GPU memory growth instead of allocating all memory upfront. Defaults to False.
    """

    # Fetch available physical GPU's
    physical_gpus = tf.config.list_physical_devices('GPU')

    for i in selected_gpus:
        if i >= len(physical_gpus):
            raise ValueError('GPU {} selected, but TensorFlow only sees {} GPU(s); is CUDA_VISIBLE_DEVICES set?'.format(i, len(physical_gpus)))

    # Remove the GPU's not provided as input
    for i in reversed(range(len(physical_gpus))):
        if i not in selected_gpus:
            physical_gpus.pop(i)

    # Set the selected GPU's as the visible devices for this session
    tf.config.set_visible_devices(physical_gpus, 'GPU')

    # Set memory growth
    if memory_growth is True:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
