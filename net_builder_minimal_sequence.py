import tensorflow as tf
from tensorflow.keras import layers


def get_last_layers(model_dict):
    """
    Get the last architecture layers from the model dictionary.

    :param model_dict: Dictionary containing model connections.
    :return: List of last architecture layers.
    """
    # Find the maximum index present in the model_dict
    max_index = max(model_dict.keys()) if model_dict else -1

    # Identify input nodes not connected to any other nodes
    input_nodes = [input_node for input_node in range(max_index + 1) if input_node not in model_dict]

    # # If there are input nodes not connected to any other nodes, return them
    # if input_nodes:
    #     return input_nodes

    # If all input nodes are connected, return the last layers from the model_dict
    return model_dict[max_index], input_nodes



def dfs(tree_encoding, leaf_nodes):
    """
    Build a tree-based model using DFS encoded tree string.

    :param tree_encoding: Encoded tree string.
    :param leaf_nodes: List of leaf nodes.
    :return: Tree-based model.
    """
    model_dict = {}  # Dictionary to store model connections

    # Start the DFS from the end of the encoded string for each leaf node.
    index = 0
    for leaf_node in leaf_nodes:
        input_layer = leaf_node
        index, _, model_dict = dfs_helper(tree_encoding, input_layer, model_dict, index)

    # Create upper connections using Concatenate layers
    for i in range(len(tree_encoding) - 1):
        if i in model_dict and i + 1 in model_dict:
            concat_layer = layers.Concatenate(axis=1)([*model_dict[i], *model_dict[i + 1]])
            model_dict[i + 1] = [concat_layer]

    # Return the output layers and the model dictionary
    #return layers.Flatten()(model_dict[0]), model_dict

    last_layers, loose_input_nodes = get_last_layers(model_dict)
    return last_layers, loose_input_nodes, model_dict

def dfs_helper(tree_encoding, input_layer, model_dict, index):
    """
    Helper function for DFS traversal.

    :param tree_encoding: Encoded tree string.
    :param input_layer: Input layer of the model.
    :param model_dict: Dictionary to store model connections.
    :param index: Current index in the encoded string.
    :return: The next index to process after completing the current subtree,
             output layer of the current node,
             and updated model_dict.
    """
    # Base case: If we reach the end of the encoded string or encounter a leaf node, return.
    if index >= len(tree_encoding) - 1 or tree_encoding[index] == '0':
        m_out = conv_module(input_layer)
        #m_out = layers.Flatten()(m_out)
        model_dict[index] = [m_out]  # Store the output layer in the model_dict
        return index, m_out, model_dict

    # Move to the next character in the encoded string.
    index += 1

    # Apply conv_module to create a convolutional layer for the current node
    m_out = conv_module(input_layer)

    # Add current node output to the model dictionary
    model_dict[index] = [m_out]

    # If the next character is '1', move deeper in the same branch
    if tree_encoding[index] == '1':
        index, m_out, model_dict = dfs_helper(tree_encoding, m_out, model_dict, index)

    # Return the index, output layer, and updated model dictionary
    return index, m_out, model_dict

# Rest of the code remains the same


def build_tree_model(input_shape, tree_encoding, num_classes):
    input_layer = layers.Input(shape=input_shape)
    input_layer2 = layers.Input(shape=input_shape)
    input_layer3 = layers.Input(shape=input_shape)

    inputs_list = [input_layer, input_layer2, input_layer3]

    output_layers, loose_input_nodes, model_dict = dfs(tree_encoding, inputs_list)

    flattened_outputs = [layers.Flatten()(output_layer) for output_layer in output_layers]

    if len(flattened_outputs)>1:
        x = layers.Concatenate(axis=1)(flattened_outputs)
    else:
        x = flattened_outputs[0]

    if loose_input_nodes:
        x1 = layers.Concatenate(axis=1)(inputs_list)
        x1 = conv_module(x1)
        x1 = layers.Flatten()(x1)
        x = layers.Concatenate(axis=1)([x, x1])
    
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    oupt = layers.Dense(num_classes, activation='softmax')(x)

    # #remove loose inputs
    # for inpt in loose_inputs:
    #     inputs_list.pop(inpt)

    model = tf.keras.Model(inputs=inputs_list, outputs=oupt)
    return model, model_dict

def conv_module(x):
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D()(x)
    return x

# # Example usage
# tree_encoding = "111101101000100011001"

# model, model_dict = build_tree_model((64, 256,), tree_encoding, 10)

# print(model.summary())
# print("Model Dictionary:", model_dict)



# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='tree_based_model2.png', show_shapes=True, show_layer_names=True)
