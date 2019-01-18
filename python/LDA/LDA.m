[num, txt] = xlsread('D:\Users\xssong\study\machine-learing\matlab\WaterMelon_3.0.xlsx');

%提取有效数据
data = num(1:end, [1, 8, 9]);
label_txt = txt([2:end], 10);
label=ismember(label_txt,'是');

%整理所需数据
data = [data, label];
class1 = data(find(label==1), [2,3]);
class2 = data(find(label==0), [2,3]);

%核心代码
% 均值向量（中心向量）
mu1 = mean(class1);
mu2 = mean(class2);
% 协方差矩阵
s1 = cov(class1);
s2 = cov(class2);
% 类内散度矩阵
sw = s1 + s2;
% 类间散度矩阵
sb = (mu1 - mu2)' * (mu1 - mu2);

%取较大特征值对应的特征向量
[V, D] = eig(inv(sw) * sb);
w = V(:, 2);
pre_value1 = class1 * w;
pre_value2 = class2 * w;
pre_value = [pre_value1; pre_value2];

%使用学习到的模型对每个样本进行预测
offset = (mean(pre_value1) + mean(pre_value2)) / 2;
for i = 1 : length(pre_value)
    pre_value(i) = pre_value(i) - offset;
    %采用sigmod函数进行类别判断
    pre_label(i) = ~round(1 / (1 + exp(- pre_value(i))));   
end
data_out = [data, pre_value, pre_label'];
% xlswrite('D:\Users\xssong\study\machine-learing\matlab\LDA数据输出.xlsx', data_out);