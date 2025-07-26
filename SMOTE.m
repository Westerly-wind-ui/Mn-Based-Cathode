function syntheticData = SMOTE(X, numMinority, numNeighbors,get_muti)
    % X: ������������������
    % numMinority: ��������������
    % numNeighbors: �ھ�����
    
    % ���� k ����
    [idx] = knnsearch(X, X, 'K', numNeighbors);
    
    % ���ɺϳ�����
%     syntheticData =[];
    for i = 1:round(numMinority * get_muti/numNeighbors)
        for j = 1:numNeighbors
            if i<size(X,1)
            
            neighbor = X(idx(i, j), :);
            gap = neighbor - X(i, :);
            alpha = rand(); % ���ѡ��һ��Ȩ��
            syntheticData((i - 1) * numNeighbors + j, :) = X(i, :) + alpha * gap;
            end
        end
    end
end