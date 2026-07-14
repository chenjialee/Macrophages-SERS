import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Model import SimplifiedCNN
import pretreatment
from sklearn.utils import shuffle
from picture import setfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

p = pretreatment.pppretreatment()

def dddraw(a, b):
    plt.figure()
    plt.plot(a, b, linewidth=1)
    plt.xlabel('Raman shift(cm-1)')
    plt.ylabel('Intensity(a.u.)')
    plt.pause(10)
    plt.close()

def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

def split_data(X, y, train_ratio=0.8, val_ratio=0.1):
    test_ratio = 1 - train_ratio - val_ratio

    combined = np.column_stack((X, y))  # 将特征和标签组合
    shuffled_combined = shuffle(combined, random_state=45)  # 打乱组合后的数据

    X_shuffled = shuffled_combined[:, :-1]  # 特征部分
    y_shuffled = shuffled_combined[:, -1]  # 标签部分

    # 划分数据
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_shuffled, y_shuffled, test_size=(1 - train_ratio), random_state=45
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=45
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def trainNIRCONV3():
    # 参数定义
    batch_size = 12
    num_epochs = 100
    now = datetime.datetime.now()
    formatted_time = now.strftime("%m-%d_%H-%M")
    folder_path = r'D:\python_project\pythonProject1\TAM_SEARS\train_log'

    # input data
    path = r"E:\Study\yjs\y1\data\averaged_Mcell.xlsx"
    data = pd.read_excel(path, header=None)

    # #增强
    spectrum = data.iloc[0, 1:]
    reflectivity = data.iloc[1:, 1:]
    reflectivity = round(reflectivity,4).transpose()
    target = data.iloc[1:, 0].values.astype(int)

    #M0
    M0_reflectivity = data.iloc[1:201,1:]
    M0_reflectivity = round(M0_reflectivity,4).transpose()
    M1_reflectivity = data.iloc[400:, 1:]
    M1_reflectivity = round(M1_reflectivity, 4).transpose()
    M2_reflectivity = data.iloc[200:400,1:]
    M2_reflectivity = round(M2_reflectivity,4).transpose()
    dddraw(spectrum, M0_reflectivity)
    dddraw(spectrum, M1_reflectivity)
    dddraw(spectrum, M2_reflectivity)

    #数据预处理
    print(reflectivity.shape)
    print('使用sg预处理')
    reflectivity = p.SG(reflectivity)
    print('使用标准化预处理')
    # reflectivity = p.area_normalize(reflectivity)
    # reflectivity = reflectivity.transpose()
    reflectivity = p.StandarScaler(reflectivity)
    reflectivity = reflectivity.transpose()

    x_data = np.array(reflectivity)
    y_data = np.array(target)

    # 数据划分
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(x_data, y_data)
    print('训练集形状:', X_train.shape, y_train.shape)
    print('训练集标签：',y_train)
    print('验证集形状:', X_val.shape, y_val.shape)
    print('测试集形状:', X_test.shape, y_test.shape)

    # 归一化处理
    # scaler = MinMaxScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_val_scaled = scaler.transform(X_val)
    # X_test_scaled = scaler.transform(X_test)
    # X_all = scaler.transform(x_data)

    X_train_Nom = torch.tensor(X_train, dtype=torch.float32)
    X_val_Nom = torch.tensor(X_val, dtype=torch.float32)
    # X_test_Nom = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.long)
    Y_val = torch.tensor(y_val, dtype=torch.long)
    # Y_test = torch.tensor(y_test, dtype=torch.long)

    #模型training过程
    net = SimplifiedCNN()
    lossFunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5,weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    train_loader = load_array((X_train_Nom, Y_train), batch_size, is_train=True)
    val_loader = load_array((X_val_Nom, Y_val), batch_size,is_train=False)

    best_acc = 0
    train_loss_values = []
    train_accuracies = [] #记录训练准确率
    train_preds=[]#用于保存每个epoch的预测结果
    train_labels=[]#用于保存每个epoch的标签
    val_accuracies = [] #记录验证准确率
    val_loss_values = []
    val_preds = []  # 用于存储所有预测结果
    for epoch in range(num_epochs):
        net.train()
        correct = 0
        val_correct = 0
        total = 0
        val_total = 0
        total_loss = 0
        val_total_loss =0
        # 清空存储的预测数据和标签
        train_preds.clear()
        train_labels.clear()
        for X_, y_ in train_loader:
            X_ = X_.unsqueeze(1)
            prey = net(X_)
            l = lossFunc(prey, y_)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # 计算训练准确率
            _, predicted = torch.max(prey.data, 1)
            total += y_.size(0)
            correct += (predicted == y_).sum().item()
            total_loss += l.item()
            # 存储当前批次的预测值和标签
            train_preds.append(prey.detach().numpy())  # detach() 是为了避免梯度计算
            train_labels.append(y_.detach().numpy())

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        avg_loss = total_loss / total
        train_loss_values.append(avg_loss)  # 记录当前损失

        net.eval()
        with torch.no_grad():
            for X_, Y_ in val_loader:
                val_pred = net(X_.unsqueeze(1))
                val_loss = lossFunc(val_pred, Y_)
                # val_accuracy = (torch.argmax(val_pred, dim=1) == Y_).float().mean().item() * 100
                #计算验证准确率
                _, val_predicted = torch.max(val_pred.data, 1)
                val_total += Y_.size(0)
                val_correct += (val_predicted == Y_).sum().item()
                val_total_loss += val_loss.item()
                val_preds.append(val_predicted.numpy())
            # 合并所有预测
            # val_pred = np.concatenate(val_preds)
            val_accuracy = 100 * val_correct/val_total
            val_accuracies.append(val_accuracy)
            val_loss = val_total_loss / val_total
            val_loss_values.append(val_loss)
        scheduler.step(val_loss)
        print(f'Epoch {epoch + 1}, Train Acc: {train_accuracy:.4f} Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            file_name = f"average_Mcell_modelCONV3_bt32_e100_loss_1003_SimplifiedCNN-Temporary_{formatted_time}.pth"
            file_path = os.path.join(folder_path, file_name)
            torch.save(net.state_dict(), file_path)
            print('更新了最佳模型！')

    val_pred_classes = np.concatenate(val_preds)
    val_true = Y_val
    print('模型训练结果输出')
    # val_pred_reshaped = val_pred.reshape(1, -1)
    print("真值为\n" + str(np.round(val_true, 2)))
    print("预测值为\n" + str(np.round(val_pred_classes, 2)))

    # 绘制损失图
    fig,ax = setfig(column=1).show()
    ax.plot(range(1, num_epochs + 1), train_loss_values, label='Train Loss', color='#2A347A')
    ax.plot(range(1, num_epochs + 1), val_loss_values, label='Validation Loss', color='#5F97D2')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_xlim(0,num_epochs)
    ax.set_ylabel('Loss',fontweight='bold')
    plt.title('Training and Validation Loss vs Epoch',fontweight='bold')
    ax2 = ax.twinx()
    ax2.plot(range(1, num_epochs + 1), train_accuracies, label='Train', color='#2A347A')
    ax2.plot(range(1, num_epochs + 1), val_accuracies, label='Validation',  color='#5F97D2')
    ax2.set_ylabel('Accuracy (%)',fontweight='bold')
    # ax.legend(loc='upper left')
    ax2.legend(loc='center right', bbox_to_anchor=(1, 0.7))
    plt.tight_layout()
    plt.savefig(r'E:\Study\yjs\y1\data\loss&accuracy3.16-1.svg', format='svg')
    plt.show()

if __name__ == "__main__":
    trainNIRCONV3()