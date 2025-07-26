function shap_values_py = calculate_shap(X_data,model)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here 
%SHAP
%X = data(:, 1:14); % 提取前 15 列
[rows, cols] = size(X_data);
%model = G_out_data.model_train; % 獲取訓練好的模型
predictions = predict_xgb(model, X_data);
%predictions = str2double(predictions);
% 计算SHAP值（简单近似）
shap_values = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        % 对于每个特征，计算其影响
        x_background = X_data(i, :);
        x_background(j) = 0; % 假设该特征为0
        pred_background = predict_xgb(model, x_background); % 预测背景
        % 计算SHAP值
        %pred_background = str2double(pred_background);
        shap_values(i, j) = predictions(i) - pred_background;
    end
end
shap_values_py = py.numpy.array(shap_values);
X_data_py = py.numpy.array(X_data);
% 在計算 SHAP 值後顯示消息
X_py_name = strcat('X_py_all.npy'); 
py.numpy.save(X_py_name, X_data_py); 
shap_values_name = strcat('shap_values_all.npy'); 
py.numpy.save(shap_values_name, shap_values_py);

disp('完成shap');

end