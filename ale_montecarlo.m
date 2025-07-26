clc;clear;close all;	
load('R_03_Apr_2025_15_07_22.mat')	
random_seed=G_out_data.random_seed ;  %界面设置的种子数 	
rng(random_seed)  %固定随机数种子 	
	
data_str="D:\matlab工具箱\代码\循环稳定性-最新\数据集-循环稳定性改.xlsx";  %读取数据的路径 	
dataO=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data1=dataO(:,2:end);test_data=table2cell(dataO(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;     %char类型	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;     %double类型	
      else	
        index_la(i)=0;     %其他类型	
     end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	
 %% 数值类型数据处理	
if length(index_double)>=1	
    data_numshuju=table2array(data1(:,index_double));	
    index_double1=index_double;	
	
    index_double1_index=1:size(data_numshuju,2);	
    data_NAN=(isnan(data_numshuju));    %找列的缺失值	
    num_NAN_ROW=sum(data_NAN);	
    index_NAN=num_NAN_ROW>round(0.2*size(data1,1));	
    index_double1(index_NAN==1)=[]; index_double1_index(index_NAN==1)=[];	
    data_numshuju1=data_numshuju(:,index_double1_index);	
    data_NAN1=(isnan(data_numshuju1));  %找行的缺失值	
    num_NAN__COL=sum(data_NAN1');	
    index_NAN1=num_NAN__COL>0;	
    index_double2_index=1:size(data_numshuju,1);	
    index_double2_index(index_NAN1==1)=[];	
    data_numshuju2=data_numshuju1(index_double2_index,:);	
    index_need_last=index_double1;	
 else	
    index_need_last=[];	
    data_numshuju2=[];	
end	
%% 文本类型数据处理	
	
data_shuju=[];	
 if length(index_char)>=1	
  for j=1:length(index_char)	
    data_get=table2array(data1(index_double2_index,index_char(j)));	
    data_label=unique(data_get);	
    if j==length(index_char)	
       data_label_str=data_label ;	
    end    	
	
     for NN=1:length(data_label)	
            idx = find(ismember(data_get,data_label{NN,1}));  	
            data_shuju(idx,j)=NN; 	
     end	
  end	
 end	
label_all_last=[index_char,index_need_last];	
[~,label_max]=max(label_all_last);	
 if(label_max==length(label_all_last))	
     str_label=0; %标记输出是否字符类型	
     data_all_last=[data_shuju,data_numshuju2];	
     label_all_last=[index_char,index_need_last];	
 else	
    str_label=1;	
    data_all_last=[data_numshuju2,data_shuju];	
    label_all_last=[index_need_last,index_char];     	
 end	
 data=data_all_last;	
 data_biao_all=data1.Properties.VariableNames;	
 for j=1:length(label_all_last)	
    data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
 end	
	
% 异常值检测	
data=data;	
	
%%  特征处理 特征选择或者降维	
	
 A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=G_out_data.select_feature_num1;   %特征选择的个数	
index_name=data_biao1;	
print_index_name=[]; 	
opts = statset('Display','iter');	
 [REF_tf,history] = sequentialfs(@REF_regress,A_data1(:,1:end-1),A_data1(:,end), 'Direction','backward','Options',opts);	
feature_need_last=find(REF_tf==1);  	
data_select=[A_data1(:,feature_need_last),A_data1(:,end)];	
	
	
for NN=1:length(feature_need_last) 	
   print_index_name{1,NN}=index_name{1,feature_need_last(NN)};	
end 	
disp('选择特征');disp(print_index_name)  	
	
	
	
%% 数据划分	
x_feature_label=data_select(:,1:end-1);    %x特征	
y_feature_label=data_select(:,end);          %y标签	
index_label1=1:(size(x_feature_label,1));	
index_label=G_out_data.spilt_label_data;  % 数据索引	
if isempty(index_label)	
     index_label=index_label1;	
end	
spilt_ri=G_out_data.spilt_rio;  %划分比例 训练集:验证集:测试集	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          %训练集个数	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
 %训练集，验证集，测试集	
train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
%Zscore 标准化	
%训练集	
x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
%验证集	
vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  %验证数据标准化	
%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 测试数据标准化  	
X_norm = [train_x_feature_label_norm;vaild_x_feature_label_norm;test_x_feature_label_norm];


%% 参数设置	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
attention_label=G_out_data.attention_label;   %注意力机制标签	
attention_head=G_out_data.attention_head;   %注意力机制设置	
	
%% 数据增强部分	
get_mutiple=G_out_data.get_mutiple;  %数据增加倍数	
methodchoose=1; 	
origin_data=[train_x_feature_label_norm,train_y_feature_label_norm;vaild_x_feature_label_norm,vaild_y_feature_label_norm]; 	
	
[SyntheticData,Synthetic_label,origin_data_label]=generate_regressdata(origin_data,methodchoose,get_mutiple);	
% 绘制生成后数据样本图	
figure_data_generate(origin_data,SyntheticData,origin_data_label,Synthetic_label)	
X_new_DATA=[origin_data;SyntheticData];             %生成的X特征数据	
Y_new_DATA=[origin_data_label;Synthetic_label];  %生成的Y标签数据	
	
syn_spilt=round(spilt_ri(1)/(spilt_ri(1)+spilt_ri(2))*length(Y_new_DATA));	
syn_index=randperm(length(Y_new_DATA));	
%以下将生成的数据随机分配到训练集和验证集中	
train_x_feature_label_norm=X_new_DATA(syn_index(1:syn_spilt),1:end-1);	
vaild_x_feature_label_norm=X_new_DATA(syn_index(syn_spilt+1:end),1:end-1);	
train_y_feature_label_norm=X_new_DATA(syn_index(1:syn_spilt),end);	
vaild_y_feature_label_norm=X_new_DATA(syn_index(syn_spilt+1:end),end);	
train_y_feature_label=train_y_feature_label_norm.*y_sig+y_mu;	
vaild_y_feature_label=vaild_y_feature_label_norm.*y_sig+y_mu;	
train_x_feature_label=train_x_feature_label_norm.*x_sig+x_mu;	
vaild_x_feature_label=vaild_x_feature_label_norm.*x_sig+x_mu;	
	
%数据生成输出数据	
train_x_feature_label_aug=(train_x_feature_label_norm.*x_sig)+x_mu;	
vaild_x_feature_label_aug=(vaild_x_feature_label_norm.*x_sig)+x_mu;	
%总体生成数据+原数据保存在以下的 augdata_all 数据里面	
augdata_all=[train_x_feature_label_aug,train_y_feature_label;vaild_x_feature_label_aug,vaild_y_feature_label;test_x_feature_label,test_y_feature_label];	
	
%% 算法处理块	
	
	
	
	
disp('XGBoost回归')	
t1=clock; 	
paramters.maxiter=50;        %最大迭代次数	
paramters.train_booster='gbtree';	
paramters.objective='reg:linear';	
paramters.depth_max=5;    %最大深度   	
paramters.learn_rate=0.1;   %学习率	
paramters.min_child=1;      %最小叶子	
paramters.subsample=0.95;  %采样	
paramters.colsample_bytree=1;	
paramters.num_parallel_tree=1; 	
	
	
[Mdl,fitness,Convergence_curve] = optimize_fitrtrXGB(train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm,num_pop,num_iter,method_mti);   	
y_train_predict_norm=predict_xgb(Mdl,train_x_feature_label_norm);  %训练集预测结果	
y_vaild_predict_norm=predict_xgb(Mdl,vaild_x_feature_label_norm);  %验证集预测结果	
y_test_predict_norm=predict_xgb(Mdl,test_x_feature_label_norm);  %测试集预测结果	
y_all_predict_norm=predict_xgb(Mdl,X_norm);
t2=clock;	
Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));       	
	

	
 y_train_predict=y_train_predict_norm*y_sig+y_mu;  %反标准化操作 	
 y_vaild_predict=y_vaild_predict_norm*y_sig+y_mu; 	
 y_test_predict=y_test_predict_norm*y_sig+y_mu; 
 y_all_predict_norm = y_all_predict_norm*y_sig+y_mu;
 train_y=train_y_feature_label; disp('***************************************************************************************************************')   	
 train_MAE=sum(abs(y_train_predict-train_y))/length(train_y) ; disp(['训练集平均绝对误差MAE：',num2str(train_MAE)])	
 train_MAPE=sum(abs((y_train_predict-train_y)./train_y))/length(train_y); disp(['训练集平均相对误差MAPE：',num2str(train_MAPE)])	
 train_MSE=(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方误差MSE：',num2str(train_MSE)]) 	
 train_RMSE=sqrt(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方根误差RMSE：',num2str(train_RMSE)]) 	
 train_R2= 1 - (norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp(['训练集R方系数R2：',num2str(train_R2)]) 	
 vaild_y=vaild_y_feature_label;disp('***************************************************************************************************************')	
 vaild_MAE=sum(abs(y_vaild_predict-vaild_y))/length(vaild_y) ; disp(['验证集平均绝对误差MAE：',num2str(vaild_MAE)])	
 vaild_MAPE=sum(abs((y_vaild_predict-vaild_y)./vaild_y))/length(vaild_y); disp(['验证集平均相对误差MAPE：',num2str(vaild_MAPE)])	
 vaild_MSE=(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方误差MSE：',num2str(vaild_MSE)])     	
 vaild_RMSE=sqrt(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方根误差RMSE：',num2str(vaild_RMSE)]) 	
 vaild_R2= 1 - (norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);    disp(['验证集R方系数R2:  ',num2str(vaild_R2)])			
 test_y=test_y_feature_label;disp('***************************************************************************************************************');   	
 test_MAE=sum(abs(y_test_predict-test_y))/length(test_y) ; disp(['测试集平均绝对误差MAE：',num2str(test_MAE)])        	
 test_MAPE=sum(abs((y_test_predict-test_y)./test_y))/length(test_y); disp(['测试集平均相对误差MAPE：',num2str(test_MAPE)])	
 test_MSE=(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方误差MSE：',num2str(test_MSE)]) 	
 test_RMSE=sqrt(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方根误差RMSE：',num2str(test_RMSE)]) 	
 test_R2= 1 - (norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   disp(['测试集R方系数R2：',num2str(test_R2)]) 	
 disp(['算法运行时间Time: ',num2str(Time)])	




%% 绘制一阶累积局部效应（ALE）图

% 参数设置
%% 绘制一阶累积局部效应（ALE）图

% 参数设置
num_bins = 10;      % 分箱数量
mc_samples = 50;    % 蒙特卡洛采样次数

% 遍历所有特征（最后一列通常为标签，所以 size(data_select,2)-1）
for feature_num = 1:size(data_select, 2)-1
    target_feature = data_select(:, feature_num);
    feature_name = print_index_name{feature_num}; % 获取特征名称

    % 分箱处理
    [~, edges] = histcounts(target_feature, num_bins);
    bin_centers = (edges(1:end-1) + edges(2:end)) / 2;

    % 初始化存储矩阵
    mc_delta = zeros(num_bins, mc_samples);  % 存储每个 bin 每次采样的预测差异

    % 对每次蒙特卡洛采样
    for mc = 1:mc_samples
        % 对每个 bin 进行采样
        for bin = 1:num_bins
            % 获取当前 bin 的边界
            lower_bound = edges(bin);
            upper_bound = edges(bin+1);
            
            % 创建扰动数据（使用标准化后的数据）
            modified_data = train_x_feature_label_norm;
            
            % 随机选择一个样本
            idx = randi(size(modified_data, 1));
            original_value = modified_data(idx, feature_num);
            
            % 生成扰动值（在 bin 范围内随机）
            perturb_value = (upper_bound - lower_bound) * rand() + lower_bound;
            perturb_value_norm = (perturb_value - x_mu(feature_num)) / x_sig(feature_num); % 标准化
            
            % 修改数据中对应的特征值
            modified_data(idx, feature_num) = perturb_value_norm;
            
            % 计算预测差异：先计算扰动后的预测，再恢复原始值计算基准预测
            orig_pred = predict_xgb(Mdl, modified_data(idx, :));
            modified_data(idx, feature_num) = original_value; % 恢复原始值
            base_pred = predict_xgb(Mdl, modified_data(idx, :));
            
            % 存储该次采样的预测差异
            mc_delta(bin, mc) = orig_pred - base_pred;
        end
    end

    % 计算平均 ALE 效应（对每个 bin 求均值）
    ale_effect = mean(mc_delta, 2);
    
    % 累积效应计算（基于平均效应）
    accumulated_effect = cumsum(ale_effect);
    
    % 计算蒙特卡洛采样下每次的累计效应曲线（按每个采样累加）
    mc_ales = cumsum(mc_delta, 1);  % 每列为一条曲线

    % 将数据转换为 Python 的 numpy 数组，并构造一个 Python 字典
    data_dict = py.dict(pyargs(...
        'bin_centers', py.numpy.array(bin_centers), ...
        'ale_effect', py.numpy.array(ale_effect), ...
        'accumulated_effect', py.numpy.array(accumulated_effect), ...
        'mc_ales', py.numpy.array(mc_ales), ...
        'train_feature', py.numpy.array(target_feature) ...
    ));

    % 保存为 npy 文件，文件名为 'ale_data_特征名.npy'
    data_struct_name = strcat('ale_data_', feature_name, '.npy');
    py.numpy.save(data_struct_name, data_dict);



   % 绘制ALE图
figure('Position', [100, 100, 800, 600])
plot(bin_centers, accumulated_effect, '-o', 'LineWidth', 2, 'MarkerSize', 8)
hold on
plot(edges, zeros(size(edges)), '--k') % 零基准线
title(sprintf('First-order ALE of feature ''%s''', feature_name))
xlabel(sprintf('Feature ''%s''', feature_name))
ylabel('Accumulated Local Effect')
grid on
set(gca, 'FontSize', 12)

% 设置x轴范围
xlim([edges(1), edges(end)])

% 添加分箱标记
for bin = 1:num_bins
    line([edges(bin) edges(bin)], [min(accumulated_effect)-0.1 max(accumulated_effect)+0.1],...
         'Color', [0.7 0.7 0.7], 'LineStyle', '--')
end

% 添加参数标注
text(edges(end-2), max(accumulated_effect)+0.1,...
     sprintf('Bins: %d - Monte-Carlo: %d', num_bins, mc_samples),...
     'HorizontalAlignment', 'right', 'FontSize', 10)

% 保存图像
save_path = sprintf('ALE_%s.png', feature_name);
saveas(gcf, save_path);
close(gcf); % 关闭当前图窗，释放内存
end