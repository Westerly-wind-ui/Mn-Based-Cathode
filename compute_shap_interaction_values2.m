function shap_interaction_values = compute_shap_interaction_values(Mdl, X)
    % 输入:
    % - Mdl: 已训练好的 TreeBagger 模型
    % - X: 预测数据（每行一个样本，每列一个特征）
    %
    % 输出:
    % - shap_interaction_values: 每个样本的 SHAP 交互值矩阵，形状为 (n_samples, n_features, n_features)

    [n_samples, n_features] = size(X);
    shap_interaction_values = zeros(n_samples, n_features, n_features);

    % 检查输入数据中是否存在 NaN 或 Inf
    if any(isnan(X), 'all') || any(isinf(X), 'all')
        disp('输入数据中存在 NaN 或 Inf');
        return;  % 如果输入数据不合法，直接返回
    end

    % 获取模型的预测均值 (全特征预测)
    full_prediction = predict(Mdl, X); % 获取所有特征的预测值

    % 如果是分类问题，处理预测输出为字符串的情况
    if iscell(full_prediction)
        full_prediction = str2double(full_prediction); % 将字符串转为数字
    end

    % 检查预测值是否存在 NaN
    if any(isnan(full_prediction))
        disp('预测结果中存在 NaN');
        return;
    end

    % 计算每个特征的单独贡献 (对角线元素)
    mean_X = mean(X, 1); % 计算每个特征的均值，避免重复计算

    for i = 1:n_features
        X_minus_i = X;
        X_minus_i(:, i) = mean_X(i); % 用平均值填补第 i 列
        pred_minus_i = predict(Mdl, X_minus_i);

        % 如果是分类问题，转换为数值类型
        if iscell(pred_minus_i)
            pred_minus_i = str2double(pred_minus_i);
        end

        % 检查预测值是否存在 NaN
        if any(isnan(pred_minus_i))
            disp('某次预测中出现了 NaN 值');
            return;
        end

        % 存储对角线元素
        shap_interaction_values(:, i, i) = (full_prediction - pred_minus_i) / 2;
    end

    % 计算交互贡献 (非对角线元素)
    for i = 1:n_features
        for j = i+1:n_features
            % 计算去除第 i 特征后的预测值
            X_minus_i = X;
            X_minus_i(:, i) = mean_X(i);
            pred_minus_i = predict(Mdl, X_minus_i);

            % 计算去除第 j 特征后的预测值
            X_minus_j = X;
            X_minus_j(:, j) = mean_X(j);
            pred_minus_j = predict(Mdl, X_minus_j);

            % 计算去除第 i 和第 j 特征后的预测值
            X_minus_ij = X;
            X_minus_ij(:, [i, j]) = repmat(mean_X([i, j]), size(X, 1), 1);
            
            % 检查 X_minus_ij 的大小是否正确
            disp(['X_minus_ij size: ', mat2str(size(X_minus_ij))]);

            pred_minus_ij = predict(Mdl, X_minus_ij);

            % 检查预测值是否存在 NaN
            if any(isnan(pred_minus_ij))
                disp('某次预测中出现了 NaN 值');
                return;
            end

            % 计算 SHAP 交互值
            shap_value = (pred_minus_ij - pred_minus_i - pred_minus_j + full_prediction) / 2;

            % 存储交互贡献并确保矩阵对称
            shap_interaction_values(:, i, j) = shap_value;
            shap_interaction_values(:, j, i) = shap_value;
        end
    end

    % 使用 Python 保存结果 (确保已经初始化 Python 环境)
    shap_values_name = 'shap_interaction_values_py_all.npy';
    try
        py.numpy.save(shap_values_name, shap_interaction_values);
        disp('SHAP interaction values have been saved to shap_interaction_values_py_all.npy');
    catch
        disp('无法保存为 numpy 文件。请检查 Python 环境设置。');
    end

    disp('完成 SHAP 交互值计算');
end
