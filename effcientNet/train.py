import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
tf.compat.v1.disable_eager_execution()
import json

# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 数据预处理
#验证集占据20%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
x_train, x_test, x_val = x_train / 255.0, x_test / 255.0, x_val / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
y_val = to_categorical(y_val, 10)

# 创建 EfficientNetB0 模型
def build_model(input_shape):
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    # include_top = False: 这表示不包括模型的顶部（输出层），因此我们可以自定义输出层。
    # input_shape: 输入数据的形状，通常为(高度, 宽度, 通道数)。
    # weights = 'imagenet': 使用在ImageNet数据集上预训练的权重，这样可以加速模型训练并提高性能。
    model = models.Sequential()
    # Sequential: 这表示我们在构建一个顺序模型，层将按添加的顺序连接
    model.add(base_model)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    # 这是一个卷积层，具有128个过滤器，过滤器的大小为3x3。
    # activation = 'relu': 使用ReLU激活函数，帮助引入非线性特征。
    # padding = 'same': 这种填充方式使得输入和输出的空间维度相同。
    model.add(layers.BatchNormalization())
    # BatchNormalization: 批归一化层，用于加速训练过程，提高模型的稳定性。
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    # 另一个卷积层，具有64个过滤器，功能类似于第一个卷积层。
    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())
    # 这个层将每个特征图的所有值取平均，以减少特征图的维度并减轻过拟合。它的输出是一个一维向量，长度等于特征图的数量。
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    # Dense(256, activation='relu'): 全连接层，具有256个神经元，用ReLU激活函数。
    # Dropout(0.5): 在训练过程中，随机丢弃50 % 的神经元，以减少过拟合。
    model.add(layers.Dense(10, activation='softmax'))
    # 输出层，具有10个神经元（适用于10类分类问题），使用softmax激活函数，将输出转换为概率分布。
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # optimizer = 'adam': 使用Adam优化器，它通常在深度学习任务中表现良好。
    # loss = 'categorical_crossentropy': 用于多类分类任务的损失函数。
    # metrics = ['accuracy']: 在训练和评估时监控准确率。
    return model

input_shape = (32, 32, 3)
model = build_model(input_shape)

# 训练模型
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(x_val, y_val))



# 保存模型
model.save_weights('efficientnet_cifar10_weights.h5')

import matplotlib.pyplot as plt

# 绘制准确率和损失图
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # 准确率图
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()

    # 损失图
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.legend()

    plt.show()

plot_history(history)



