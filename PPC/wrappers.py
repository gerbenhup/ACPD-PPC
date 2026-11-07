import tensorflow as tf

def time_distributed(module: tf.keras.layers.Layer) -> tf.keras.Model:
    """Apply a module to every step of an extra time axis, e.g. an encoder to every segment of a sequence

    Args:
        module (tf.keras.layers.Layer): module with input shape [B, ...]

    Returns:
        tf.keras.Model: model with input shape [B, T, ...], applying the module to each of the T steps
    """

    # Get old and new input shapes
    input_shape     = list(module.input.shape[1:])
    new_input_shape = [None,] + input_shape

    # Create new wrapped module
    input      = tf.keras.Input(new_input_shape)
    output     = tf.keras.layers.TimeDistributed(module)(input)
    new_module = tf.keras.Model(input, output)

    return new_module

def time_stacked(modules: list[tf.keras.layers.Layer], inputs_concat: list[bool]) -> tf.keras.Model:
    """Combine T modules with the same inputs into one model whose outputs are stacked along a new time axis

    Used to run the Nf forecast models (one per future step) as a single model: [B, Ng] -> [B, Nf, Ne, 2].

    Args:
        modules (list[tf.keras.layers.Layer]): T modules with identical input shapes, e.g. models or layers
        inputs_concat (list[bool]): for each input of the modules, whether every module gets its own slice of a
            stacked input of shape [B, T, ...] (True), or all modules share the same input (False)

    Returns:
        tf.keras.Model: model with the module outputs stacked along axis 1, shape [B, T, ...]
    """

    new_inputs        = [None for _ in range(len(inputs_concat))]
    inputs_per_module = [[] for _ in range(len(modules))]

    # Create new inputs
    for i, (input_, concat) in enumerate(zip(modules[0].inputs, inputs_concat)):
        if concat is True:
            new_inputs[i] = tf.keras.Input((len(modules),) + input_.shape[1:])
        else:
            new_inputs[i] = tf.keras.Input(input_.shape[1:])

    # Determine inputs for every module
    for j in range(len(modules)):
        for i, (input_, concat) in enumerate(zip(new_inputs, inputs_concat)):
            if concat is True:
                inputs_per_module[j].append(tf.keras.layers.Lambda(lambda x, j=j: x[:, j])(input_))
            else:
                inputs_per_module[j].append(input_)

    # Fetch module outputs and stack them in time
    iresults = [module(inputs) for module, inputs in zip(modules, inputs_per_module)]
    sresults = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(iresults)

    # Create new module
    new_module = tf.keras.Model(new_inputs, sresults)

    return new_module
