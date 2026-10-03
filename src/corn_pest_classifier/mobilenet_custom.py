from tensorflow import keras


def inverted_residual_block(x, expand_filters, squeeze_filters, stride=1):
    # A sua implementação manual (Expansão, Depthwise, Compressão)
    m = keras.layers.Conv2D(expand_filters, kernel_size=1, padding='same', use_bias=False)(x)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ReLU(max_value=6.0)(m)
    
    m = keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ReLU(max_value=6.0)(m)
    
    m = keras.layers.Conv2D(squeeze_filters, kernel_size=1, padding='same', use_bias=False)(m)
    m = keras.layers.BatchNormalization()(m)
    
    if x.shape[-1] == squeeze_filters and stride == 1:
        return keras.layers.Add()([x, m])
    return m

def build_custom_mobilenet(input_shape=(224, 224, 3), num_classes=2):
    inputs = keras.Input(shape=input_shape)
    
    # Pre-processamento: normaliza pixels de [0, 255] para [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Camada inicial
    x = keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(x)
    
    # Pilha de blocos customizados (Nível 2) - versão simplificada
    x = inverted_residual_block(x, expand_filters=32, squeeze_filters=16, stride=1)
    x = inverted_residual_block(x, expand_filters=96, squeeze_filters=24, stride=2)
    x = inverted_residual_block(x, expand_filters=144, squeeze_filters=32, stride=2)
    
    # Cabeça de classificação
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs, name="Custom_MobileNet_Milho")