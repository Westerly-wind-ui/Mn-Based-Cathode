function shap_interaction_values = compute_shap_interaction_values(Mdl, X)
    % 输入:
    % - Mdl: 已训练好的 TreeBagger 模型
    % - X: 预测数据（每行一个样本，每列一个特征）
    %
    % 输出:
    % - shap_interaction_values: 每个样本的 SHAP 交互值矩阵，形状为 (n_samples, n_features, n_features)

    [n_samples, n_features] = size(X);
    shap_interaction_values = zeros(n_samples, n_features, n_features);

    % 获取模型的预测均值 (全特征预测)
    full_prediction = predict_xgb(Mdl, X); % 所有特征的预测值

    % 对角线元素：特征的单独贡献
    for i = 1:n_features
        X_minus_i = X;
        X_minus_i(:, i) = mean(X(:, i)); % 用平均值填补第 i 列
        pred_minus_i = predict_xgb(Mdl, X_minus_i);

        % 存储对角线元素
        shap_interaction_values(:, i, i) = (full_prediction - pred_minus_i) / 2;
    end

    % 计算交互贡献
    for i = 1:n_features
        for j = i+1:n_features
            X_minus_i = X;
            X_minus_i(:, i) = mean(X(:, i));  % 填补第 i 个特征
            pred_minus_i = predict_xgb(Mdl, X_minus_i);

            X_minus_j = X;
            X_minus_j(:, j) = mean(X(:, j));  % 填补第 j 个特征
            pred_minus_j = predict_xgb(Mdl, X_minus_j);

            X_minus_ij = X;
            %X_minus_ij(:, [i, j]) = mean(X(:, [i, j]), 1);  % 同时填补 i 和 j
            X_minus_ij(:, [i, j]) = repmat(mean(X(:, [i, j]), 1), size(X, 1), 1);  % 同时填补 i 和 j

            pred_minus_ij = predict_xgb(Mdl, X_minus_ij);

            % 计算 SHAP 交互值并确保矩阵对称
            shap_value = (pred_minus_ij - pred_minus_i - pred_minus_j + full_prediction) / 2;
            shap_interaction_values(:, i, j) = shap_value;
            shap_interaction_values(:, j, i) = shap_value;
        end
    end
    shap_values_name = strcat('shap_interaction_values_py_all.npy'); 
    py.numpy.save(shap_values_name, shap_interaction_values);

    disp('完成shap interaction');
end
