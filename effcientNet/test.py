import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models, layers

# 加载 CIFAR-10 测试集
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0
y_test = to_categorical(y_test, 10)

# 加载模型
def build_model(input_shape):
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 设置输入形状
input_shape = (32, 32, 3)

# 构建模型
model = build_model(input_shape)

# 加载模型权重
model.load_weights('efficientnet_cifar10_weights.h5')

# 测试模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# 可视化
predictions = model.predict(x_test)

# 可视化一些预测结果
def plot_sample(index):
    plt.imshow(x_test[index])
    plt.title(f"Predicted: {predictions[index].argmax()}, Actual: {y_test[index].argmax()}")
    plt.show()


# 训练过程的可视化函数
for i in range(5):
    plot_sample(i)


# 可视化损失和准确率
def plot_test_metrics(test_loss, test_acc):
    plt.figure(figsize=(6, 4))

    # 绘制损失和准确率
    plt.bar(['Test Loss', 'Test Accuracy'], [test_loss, test_acc], color=['blue', 'orange'])
    plt.ylim(0, 1)
    plt.title('Test Loss and Accuracy')
    plt.ylabel('Value')
    plt.show()


plot_test_metrics(test_loss, test_acc)