import tensorflow as tf

def TimeDistributed(module: tf.keras.layers.Layer) -> tf.keras.Model:
    """Wrap module in TimeDistributed wrapper.

    Args:
        module (tf.keras.layers.Layer): module to wrap

    Returns:
        tf.keras.Model: wrapped module
    """

    # Get old and new input shapes
    input_shape     = list(module.input.shape[1:])
    new_input_shape = [None,] + input_shape

    # Create new wrapped module
    input      = tf.keras.Input(new_input_shape)
    output     = tf.keras.layers.TimeDistributed(module)(input)
    new_module = tf.keras.Model(input, output)

    return new_module

def TimeStacked(modules: list[tf.keras.layers.Layer], inputs_concat: list[bool]) -> tf.keras.Model:
    """Wrap multiple modules with the same input sizes into one module with stacked outputs

    Args:
        modules (list[tf.keras.layers.Layer]): list of TensorFlow modules, e.g., models or layers
        inputs_concat (list[bool]): whether to stack the input elements of each seperate module or to use a single input

    Returns:
        tf.keras.Model: stacked modules in a single model
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
                inputs_per_module[j].append(tf.keras.layers.Lambda(lambda x: x[:, j])(input_))
            else:
                inputs_per_module[j].append(input_)

    # Fetch module outputs and stack them in time
    iresults = [module(inputs) for module, inputs in zip(modules, inputs_per_module)]
    sresults = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(iresults)

    # Create new module
    new_module = tf.keras.Model(new_inputs, sresults)

    return new_module
