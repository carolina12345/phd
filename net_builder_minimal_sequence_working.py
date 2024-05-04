import tensorflow as tf
from tensorflow.keras import layers

def dfs(tree_encoding, leaf_nodes):
    """
    Build a tree-based model using DFS encoded tree string (upside down).

    :param leaf_nodes: List of leaf nodes (several root nodes in the upside-down view).
    :param tree_encoding: Encoded tree string.
    :return: Tree-based model.
    """
    model_dict = {}  # Dictionary to store model connections
    output_layers = []

    # Start the DFS from the end of the encoded string for each leaf node.
    for leaf_node in leaf_nodes:
        input_layer = leaf_node
        _, output = dfs_helper(tree_encoding, input_layer, model_dict, 0)
        output_layers.append(output)

    return output_layers


def dfs_helper(structure, input_layer, model_dict, index):
    """
    Helper function for DFS.

    :param structure: Node structure indicating how many children each node has.
    :param input_layer: Input layer of the model.
    :param model_dict: Dictionary to store model connections.
    :return: The output layer of the current subtree.
    """
    if structure[index] == 0:  # Leaf node
        m_out = conv_module(input_layer)
        return structure, m_out

    structure, left_child = dfs_helper(structure, input_layer, model_dict, index+1)

    m_out = conv_module(left_child)
    return structure, m_out


# # Adjusted helper function for DFS traversal
# def dfs_helper(index, encoded_tree, depth, input_layer, model_dict):
#     """
#     Helper function for DFS (upside down).

#     :param index: Current index in the binary string.
#     :param depth: Current depth in the tree for visualization.
#     :param input_layer: Input layer of the model.
#     :param model_dict: Dictionary to store model connections.
#     :return: The next index to process after completing the current subtree.
#     """
#     # Base case: If we reach the beginning of the string, return.
#     if index < 0:
#         #m_out = conv_module(input_layer)
#         return index, input_layer

#     # Move to the previous character in the encoded string.
#     index -= 1

#     # Apply conv_module to create a convolutional layer for the current node
#     m_out = conv_module(input_layer)

#     # Add current node output to the model dictionary
#     try:
#         model_dict[index].append(m_out)
#     except:
#         model_dict[index] = [m_out]

#     # Create branches for each child node represented by '1's in the binary string
#     while index >= 0 and encoded_tree[index] == '1':
#         # Recursive call to process each child, updating the index and depth
#         #index, _ = dfs_helper(index, encoded_tree, depth + 1, current_node_output, model_dict)
#         index, m_out = dfs_helper(index, encoded_tree, depth + 1, m_out, model_dict)
#         model_dict[index].append(m_out)

#     # Once we encounter a '0' or reach the beginning of the string, it means we are done with this node's children and can go back up.
#     return index, m_out, model_dict  # Return the index and output of the current node


####################################################################################

# def dfs(tree_structure, input_layer):
#     """
#     Build a tree-based model using DFS based on the provided tree structure.

#     :param tree_structure: List representing the tree structure.
#     :param input_layer: Input layer of the model.
#     :return: List of output layers corresponding to each leaf node.
#     """
#     model_dict = {}  # Dictionary to store model connections
#     output_layers = []

#     # Start the DFS from the root node.
#     for i, structure in enumerate(tree_structure):
#         _, output = dfs_helper(structure, input_layer, model_dict)
#         output_layers.append(output)

#     return output_layers

# def dfs_helper(structure, input_layer, model_dict):
#     """
#     Helper function for DFS.

#     :param structure: Node structure indicating how many children each node has.
#     :param input_layer: Input layer of the model.
#     :param model_dict: Dictionary to store model connections.
#     :return: The output layer of the current subtree.
#     """
#     if structure == 0:  # Leaf node
#         m_out = conv_module(input_layer)
#         return structure, m_out

#     left_child, right_child = dfs_helper(structure - 1, input_layer, model_dict), dfs_helper(structure - 1, input_layer, model_dict)

#     concat_layer = layers.Concatenate()([left_child[1], right_child[1]])
#     m_out = conv_module(concat_layer)
#     return structure, m_out

def build_tree_model(input_shape, tree_structure, num_classes):

    # Input layer
    input_layer = layers.Input(shape=input_shape)

    # Char level branch
    conv1d_char = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)

    # Word/Token Level Branch
    conv1d_word = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(conv1d_char)

    # Sentence Level Branch
    conv1d_sentence = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(conv1d_word)

    # Document Level Branch
    conv1d_document = layers.Conv1D(filters=512, kernel_size=7, activation='relu')(conv1d_sentence)


    xs = dfs(tree_structure, [conv1d_char, conv1d_word, conv1d_sentence, conv1d_document])

    flattened_outputs = [layers.Flatten()(xi) for xi in xs]

    x = layers.Concatenate(axis=-1)(flattened_outputs)

    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def conv_module(x):
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D()(x)
    return x

def build_tree_model(input_shape, tree_encoding, num_classes):
    num_filters = 1024
    kernel_sizes = [7, 7, 3, 3, 3, 3]
    pool_size = 3
    stride_length = 3

    # Input layer
    input_layer = layers.Input(shape=input_shape)

    # Char level branch
    conv1d_char = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)

    # Word/Token Level Branch
    conv1d_word = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(conv1d_char)

    # Sentence Level Branch
    conv1d_sentence = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(conv1d_word)

    # Document Level Branch
    conv1d_document = layers.Conv1D(filters=512, kernel_size=7, activation='relu')(conv1d_sentence)
    # max_pooling_document = layers.GlobalMaxPooling1D()(conv1d_document)

    #########################################

    # conv1d_char = layers.Flatten()(conv1d_char)

    # conv1d_word = layers.Flatten()(conv1d_word)
    # conv1d_sentence = layers.Flatten()(conv1d_sentence)
    # max_pooling_document = layers.Flatten()(max_pooling_document)


    #########################################

    # # Concatenate branches
    # x = layers.Concatenate(axis=-1)([conv1d_char, conv1d_word, conv1d_sentence, max_pooling_document])

    # x = layers.Reshape((109824, 1))(x)

    # Perform DFS traversal to respect the tree structure
    xs = dfs(tree_encoding, [conv1d_char, conv1d_word, conv1d_sentence, conv1d_document])

    flattened_outputs = []
    for xi in xs:
        xout = layers.Flatten()(xi)
        flattened_outputs.append(xout)

    x = layers.Concatenate(axis=-1)(flattened_outputs)

    #x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def conv_module(x):
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D()(x)
    #x = layers.Dropout(0.5)(x)
    return x


# Example usage
tree_structure = [1, 1, 0, 1, 0, 0, 0]
model = build_tree_model((250, 68), tree_structure, 52)
model.summary()


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='tree_based_model.png', show_shapes=True, show_layer_names=True)
