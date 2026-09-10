import subprocess
import tensorflow as tf

def find_available(num_gpus: int, max_utilization: int, min_free_memory: int) -> list[int]:
    """Find GPU indices with sufficiently low utilization and sufficient free memory, via `nvidia-smi`

    Args:
        num_gpus (int): number of available GPU's to find
        max_utilization (int): maximum utilization (%) for a GPU to be considered available
        min_free_memory (int): minimum free memory (MiB) for a GPU to be considered available

    Returns:
        list[int]: indices of the first `num_gpus` GPU's that satisfy the given constraints
    """

    stats = subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv')
    stats = stats.decode().splitlines()[1:]
    stats = [x.split(', ') for x in stats]
    stats = [[int(y[0][:-2]), int(y[1][:-4])] for y in stats]

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
        selected_gpus (list[int]): indices of GPU's to make visible to TensorFlow
        memory_growth (bool, optional): whether to allow GPU memory growth instead of allocating all memory upfront. Defaults to False.
    """

    try:
        # Fetch available physical GPU's
        physical_gpus = tf.config.list_physical_devices('GPU')

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

    except Exception as e:
        print(e)
