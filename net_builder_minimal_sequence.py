import tensorflow as tf
from tensorflow.keras import layers, models, Input


def dfs_helper(index, encoded_tree, depth, input_layer, model_dict):
    """
    Helper function for DFS.

    :param index: Current index in the binary string.
    :param depth: Current depth in the tree for visualization.
    :param input_layer: Input layer of the neural network.
    :param model_dict: Dictionary to store model connections.
    :return: The next index to process after completing the current subtree.
    """
    # Base case: If we reach the end of the string, return.
    if index >= len(encoded_tree):
        return index, input_layer

    # Move to the next character in the encoded string.
    index += 1

    # Apply conv_module to create a convolutional layer for the current node
    current_node_output = conv_module(input_layer)

    # Add current node output to the model dictionary
    model_dict[index] = current_node_output

    # Create branches for each child node represented by '1's in the binary string
    while index < len(encoded_tree) and encoded_tree[index] == '1':
        # Recursive call to process each child, updating the index and depth
        index, _ = dfs_helper(index, encoded_tree, depth + 1, current_node_output, model_dict)

    # Once we encounter a '0' or reach the end of the string, it means we are done with this node's children and can go back up.
    return index, current_node_output  # Return the index and output of the current node


def dfs(tree_encoding, input):
    """
    Build a tree-based model using DFS encoded tree string.

    :param input: Input of the model.
    :param tree_encoding: Encoded tree string.
    :return: Tree-based model.
    """
    model_dict = {}  # Dictionary to store model connections

    # Start the DFS from the beginning of the encoded string and at depth 0.
    _, output = dfs_helper(0, tree_encoding, 0, input, model_dict)

    # Connect the model branches based on the dictionary
    outputs = []
    for i in range(len(tree_encoding)):
        if i in model_dict:
            outputs.append(model_dict[i])

    # Concatenate the outputs of all branches
    if len(outputs) > 1:
        merged_output = layers.Concatenate(axis=1)(outputs)
    else:
        merged_output = outputs[0] if outputs else None

    # Add additional layers if needed
    if merged_output is not None:
        x = conv_module(merged_output)  # Add additional layers as needed
    else:
        x = conv_module(output)

    # # Flatten and add output layer
    # x = layers.Flatten()(x)
    # output_layer = layers.Dense(num_classes, activation='softmax')(x)

    # # Create the model
    # model = models.Model(inputs=output, outputs=output_layer)

    # return model
    return x




# def dfs(encoded_tree, input_layer, depth=0):
#     """
#     Perform a DFS on an n-ary tree encoded as a binary string.
#     This function assumes '1' represents the start of a new node,
#     and '0' represents the end of a node's children (moving back up the tree).

#     :param encoded_tree: A binary string encoding of the n-ary tree.
#     :param input_layer: The input layer of the neural network.
#     :param depth: Current depth in the tree.
#     :return: None
#     """    

#     def dfs_helper(index, depth, input_layer):
#         """
#         Helper function for DFS.

#         :param index: Current index in the binary string.
#         :param depth: Current depth in the tree for visualization.
#         :param input_layer: Input layer of the neural network.
#         :return: The next index to process after completing the current subtree.
#         """
#         # Base case: If we reach the end of the string or encounter '0', return.
#         if index >= len(encoded_tree) or encoded_tree[index] == '0':
#             return index - 1, input_layer

#         # Move to the next character in the encoded string.
#         index += 1

#         # Apply conv_module to create a convolutional layer for the current node
#         current_node_output = conv_module(input_layer)

#         # Initialize a list to hold the outputs of convolutional layers for each child branch
#         branch_outputs = []

#         # Create branches for each child node represented by '1's in the binary string
#         while index < len(encoded_tree) and encoded_tree[index] == '1':
#             # Recursive call to process each child, updating the index and depth
#             index, child_output = dfs_helper(index, depth + 1, current_node_output)
#             branch_outputs.append(child_output)

#         # Once we encounter a '0' or reach the end of the string, it means we are done with this node's children and can go back up.
#         # Return the output of the current node
#         return index, current_node_output  # Return the index and output of the current node

#     # Start the DFS from the beginning of the encoded string and at depth 0.
#     _, out = dfs_helper(0, depth, input_layer)

#     return out


def build_tree_model(input_shape, tree_encoding, num_classes):
    num_filters = 1024
    kernel_sizes = [7, 7, 3, 3, 3, 3]
    pool_size = 3
    stride_length = 3

    inputs = Input(shape=input_shape)
    x = layers.Conv1D(filters=num_filters, kernel_size=kernel_sizes[0], activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=pool_size, strides=stride_length)(x)
    x = layers.Conv1D(filters=num_filters, kernel_size=kernel_sizes[1], activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=pool_size, strides=stride_length)(x)

    # Perform DFS traversal to respect the tree structure
    x = dfs(tree_encoding, x)

    # flattened_outputs = []
    # for xi in xs:
    #     xout = layers.Flatten()(xi)
    #     flattened_outputs.append(xout)

    # x = layers.Concatenate(axis=-1)(flattened_outputs)

    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output_layer)
    return model


def conv_module(x):
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    return x


# Example usage
model = build_tree_model((250, 68), "111001000", 52)
model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='tree_based_model.png', show_shapes=True, show_layer_names=True)
