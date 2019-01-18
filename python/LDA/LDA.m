[num, txt] = xlsread('D:\Users\xssong\study\machine-learing\matlab\WaterMelon_3.0.xlsx');

%��ȡ��Ч����
data = num(1:end, [1, 8, 9]);
label_txt = txt([2:end], 10);
label=ismember(label_txt,'��');

%������������
data = [data, label];
class1 = data(find(label==1), [2,3]);
class2 = data(find(label==0), [2,3]);

%���Ĵ���
% ��ֵ����������������
mu1 = mean(class1);
mu2 = mean(class2);
% Э�������
s1 = cov(class1);
s2 = cov(class2);
% ����ɢ�Ⱦ���
sw = s1 + s2;
% ���ɢ�Ⱦ���
sb = (mu1 - mu2)' * (mu1 - mu2);

%ȡ�ϴ�����ֵ��Ӧ����������
[V, D] = eig(inv(sw) * sb);
w = V(:, 2);
pre_value1 = class1 * w;
pre_value2 = class2 * w;
pre_value = [pre_value1; pre_value2];

%ʹ��ѧϰ����ģ�Ͷ�ÿ����������Ԥ��
offset = (mean(pre_value1) + mean(pre_value2)) / 2;
for i = 1 : length(pre_value)
    pre_value(i) = pre_value(i) - offset;
    %����sigmod������������ж�
    pre_label(i) = ~round(1 / (1 + exp(- pre_value(i))));   
end
data_out = [data, pre_value, pre_label'];
% xlswrite('D:\Users\xssong\study\machine-learing\matlab\LDA�������.xlsx', data_out);