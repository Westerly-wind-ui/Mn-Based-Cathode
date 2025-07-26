function syntheticData = SMOTE(X, numMinority, numNeighbors,get_muti)
    % X: 少数类样本特征矩阵
    % numMinority: 少数类样本数量
    % numNeighbors: 邻居数量
    
    % 计算 k 近邻
    [idx] = knnsearch(X, X, 'K', numNeighbors);
    
    % 生成合成样本
%     syntheticData =[];
    for i = 1:round(numMinority * get_muti/numNeighbors)
        for j = 1:numNeighbors
            if i<size(X,1)
            
            neighbor = X(idx(i, j), :);
            gap = neighbor - X(i, :);
            alpha = rand(); % 随机选择一个权重
            syntheticData((i - 1) * numNeighbors + j, :) = X(i, :) + alpha * gap;
            end
        end
    end
end