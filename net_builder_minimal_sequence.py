import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from transformer_keras_io import *
from tensorflow.keras import layers, models, Input



def dfs(encoded_tree, input_layer):
    """
    Perform a DFS on an n-ary tree encoded as a binary string.
    This function assumes '1' represents the start of a new node,
    and '0' represents the end of a node's children (moving back up the tree).

    :param encoded_tree: A binary string encoding of the n-ary tree.
    :return: None
    """

    def dfs_helper(index, depth, input_layer):
        """
        Helper function for DFS.

        :param index: Current index in the binary string.
        :param depth: Current depth in the tree for visualization.
        :return: The next index to process after completing the current subtree.
        """
        # Base case: If we reach the end of the string, return.
        if index >= len(encoded_tree):
            # x = layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer[-1])
            # x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
            return index, input_layer

        # Assuming '1' is a node, print it or process it here.
        print(f"{'  ' * depth}Node at depth {depth}")
        input_layer = inception_module(input_layer,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32)

        # Move to the next character in the encoded string.
        index += 1

        # Go deeper into the tree as long as we encounter '1's, indicating more children/subtrees.
        #in_layers = [input_layer[-1]]
        #input_layer_local = input_layer[-1]
        while index < len(encoded_tree) and encoded_tree[index] == '1':
            
            # Recursive call to process each child, updating the index each time.
            #in_layers.append(input_layer_local)

            index, input_layer = dfs_helper(index, depth + 1, input_layer)

        # Once we encounter a '0', it means we are done with this node's children and can go back up.
        return index + 1, input_layer  # Skip the '0' and move to the next part of the encoded tree.

    # Start the DFS from the beginning of the encoded string and at depth 0.
    _, out = dfs_helper(0, 0, input_layer)

    return out


def build_tree_model(input_shape, num_classes, tree_encoding, max_length, vocab_size, embedding_dim):
    input_layer = layers.Input(shape=input_shape)
    x = input_layer

    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embedding_dim)
    x = embedding_layer(x)

    # x = layers.Conv1D(64, 7, padding='same', strides=2, activation='relu')(x)
    # x = layers.MaxPooling1D(3, padding='same', strides=2)(x)
    # x = layers.Conv1D(192, 3, padding='same', activation='relu')(x)
    # x = layers.MaxPooling1D(3, padding='same', strides=2)(x)

    x = dfs(tree_encoding, x)

    x = layers.MaxPooling1D(3, padding='same', strides=2)(x)

    # Simplified to avoid a very long implementation
    # Typically, GoogLeNet would have more inception modules here

    x = layers.AveragePooling1D(7, strides=1)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1000, activation='relu')(x)

    # Fully Connected Layer
    x = layers.Dense(256, activation='relu')(x)

    # Output Layer
    #output_layer = Dense(num_classes, activation='softmax')(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model



def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj):
    
    conv_1x1 = layers.Conv1D(filters_1x1, 1, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    conv_3x3 = layers.Conv1D(filters_3x3_reduce, 1, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    conv_3x3 = layers.Conv1D(filters_3x3, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv_3x3)

    conv_5x5 = layers.Conv1D(filters_5x5_reduce, 1, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    conv_5x5 = layers.Conv1D(filters_5x5, 5, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(conv_5x5)

    pool_proj = layers.MaxPooling1D(3, strides=1, padding='same')(x)
    pool_proj = layers.Conv1D(filters_pool_proj, 1, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(pool_proj)

    x = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=-1)

    output = layers.Dropout(0.3)(x)
    print('inception module output ', output.shape)
    return output

#     x = inception_module(x,
#                          filters_1x1=128,
#                          filters_3x3_reduce=128,
#                          filters_3x3=192,
#                          filters_5x5_reduce=32,
#                          filters_5x5=96,
#                          filters_pool_proj=64)
    



# model = build_tree_model((128), 5, "111001000", 150, 1000, 256)

# model.summary()

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='tree_based_model.png', show_shapes=True, show_layer_names=True)