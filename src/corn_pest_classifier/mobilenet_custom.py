from tensorflow import keras


def inverted_residual_block(
    x,
    expand_filters: int,
    squeeze_filters: int,
    stride: int = 1,
    weight_decay: float = 1e-4,
    block_id: str = "b",
):
    """Bloco Residual Invertido (MobileNetV2):
    1. Expansão Pointwise (Conv 1x1 + BN + ReLU6)
    2. Filtragem Espacial Depthwise (DepthwiseConv 3x3 + BN + ReLU6)
    3. Projeção Linear / Compressão (Conv 1x1 + BN sem ativação não-linear)
    4. Skip Connection se stride == 1 e canais de entrada == saída.
    """
    reg = keras.regularizers.l2(weight_decay)
    in_channels = x.shape[-1]

    # 1. Expansão (só expande se expand_filters > in_channels)
    if expand_filters != in_channels:
        m = keras.layers.Conv2D(
            expand_filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
            name=f"{block_id}_expand_conv",
        )(x)
        m = keras.layers.BatchNormalization(name=f"{block_id}_expand_bn")(m)
        m = keras.layers.ReLU(max_value=6.0, name=f"{block_id}_expand_relu6")(m)
    else:
        m = x

    # 2. Depthwise Convolution 3x3
    m = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        depthwise_regularizer=reg,
        name=f"{block_id}_depthwise",
    )(m)
    m = keras.layers.BatchNormalization(name=f"{block_id}_depthwise_bn")(m)
    m = keras.layers.ReLU(max_value=6.0, name=f"{block_id}_depthwise_relu6")(m)

    # 3. Projeção Linear (Gargalo Estreito - Linear Bottleneck)
    m = keras.layers.Conv2D(
        squeeze_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name=f"{block_id}_project_conv",
    )(m)
    m = keras.layers.BatchNormalization(name=f"{block_id}_project_bn")(m)

    # 4. Conexão Residual (Atalho de Identidade)
    if in_channels == squeeze_filters and stride == 1:
        return keras.layers.Add(name=f"{block_id}_add")([x, m])
    return m


def build_custom_mobilenet(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    use_imagenet_backbone: bool = True,
    dropout_rate: float = 0.35,
    weight_decay: float = 1e-4,
) -> keras.Model:
    """Constrói a MobileNetV2 Híbrida para classificação de pragas do milho:
    - Combina Transfer Learning (pesos ImageNet congelados na Fase 1) + Bloco Residual Invertido Customizado
      de refinamento de domínio agrícola + Cabeça regularizada (BatchNorm + Dropout + L2).
    - Caso esteja sem internet para baixar os pesos da ImageNet, constrói a pilha profunda customizada de 7 estágios.
    """
    reg = keras.regularizers.l2(weight_decay)
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Pré-processamento nativo MobileNetV2: normaliza pixels de [0, 255] para [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)

    backbone_loaded = False
    if use_imagenet_backbone:
        try:
            base_backbone = keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet",
                alpha=1.0,
            )
            # Fase 1 (Warmup): Congela o extrator de características pré-treinado
            base_backbone.trainable = False
            x = base_backbone(x, training=False)
            # Bloco Residual Invertido Customizado de Adaptação de Domínio Agrícola
            x = inverted_residual_block(
                x,
                expand_filters=512,
                squeeze_filters=256,
                stride=1,
                weight_decay=weight_decay,
                block_id="agro_neck",
            )
            backbone_loaded = True
            print("✓ Backbone MobileNetV2 (ImageNet) + Bloco Residual Agrícola carregados com sucesso!")
        except Exception as e:
            print(f"[Aviso] Não foi possível baixar pesos ImageNet ({e}). Usando pilha profunda customizada.")

    if not backbone_loaded:
        # Stem Inicial com BatchNormalization e ReLU6
        x = keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
            name="stem_conv",
        )(x)
        x = keras.layers.BatchNormalization(name="stem_bn")(x)
        x = keras.layers.ReLU(max_value=6.0, name="stem_relu6")(x)

        # Pilha profunda de Blocos Residuais Invertidos (7 estágios progressivos)
        x = inverted_residual_block(x, expand_filters=32, squeeze_filters=16, stride=1, weight_decay=weight_decay, block_id="b1")
        x = inverted_residual_block(x, expand_filters=96, squeeze_filters=24, stride=2, weight_decay=weight_decay, block_id="b2_1")
        x = inverted_residual_block(x, expand_filters=144, squeeze_filters=24, stride=1, weight_decay=weight_decay, block_id="b2_2")
        x = inverted_residual_block(x, expand_filters=144, squeeze_filters=32, stride=2, weight_decay=weight_decay, block_id="b3_1")
        x = inverted_residual_block(x, expand_filters=192, squeeze_filters=32, stride=1, weight_decay=weight_decay, block_id="b3_2")
        x = inverted_residual_block(x, expand_filters=192, squeeze_filters=64, stride=2, weight_decay=weight_decay, block_id="b4_1")
        x = inverted_residual_block(x, expand_filters=384, squeeze_filters=96, stride=1, weight_decay=weight_decay, block_id="b5_1")
        x = inverted_residual_block(x, expand_filters=576, squeeze_filters=160, stride=2, weight_decay=weight_decay, block_id="b6_1")

        # Expansão final de features antes do pooling
        x = keras.layers.Conv2D(
            320,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
            name="head_conv1x1",
        )(x)
        x = keras.layers.BatchNormalization(name="head_bn")(x)
        x = keras.layers.ReLU(max_value=6.0, name="head_relu6")(x)

    # Cabeça de Classificação Regularizada contra Overfitting
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = keras.layers.BatchNormalization(name="classifier_bn")(x)
    x = keras.layers.Dropout(dropout_rate, name="classifier_dropout_1")(x)
    x = keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=reg,
        name="classifier_dense_128",
    )(x)
    x = keras.layers.Dropout(dropout_rate * 0.5, name="classifier_dropout_2")(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="pest_probabilities",
    )(x)

    return keras.Model(inputs, outputs, name="Custom_MobileNetV2_Milho")


def unfreeze_for_finetuning(model: keras.Model, unfreeze_top_layers: int = 35) -> int:
    """Fase 2 (Fine-Tuning): Descongela seletivamente as últimas `unfreeze_top_layers` camadas do backbone
    mantendo camadas de BatchNormalization congeladas para preservar as estatísticas móveis.
    """
    unfrozen_count = 0
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower():
            layer.trainable = True
            total_sublayers = len(layer.layers)
            cutoff = max(0, total_sublayers - unfreeze_top_layers)
            for idx, sublayer in enumerate(layer.layers):
                if idx < cutoff or isinstance(sublayer, keras.layers.BatchNormalization):
                    sublayer.trainable = False
                else:
                    sublayer.trainable = True
                    unfrozen_count += 1
    if unfrozen_count > 0:
        print(f"🔓 [Fine-Tuning] {unfrozen_count} camadas superiores do backbone descongeladas (BatchNorm preservado)!")
    return unfrozen_count
