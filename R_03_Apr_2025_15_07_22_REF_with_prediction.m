clc;clear;close all;	
load('R_03_Apr_2025_15_07_22.mat')	
random_seed=G_out_data.random_seed ;  %界面设置的种子数 	
rng(random_seed)  %固定随机数种子 	
	
data_str="C:\Users\hongy\Desktop\dataset.xlsx";  %读取数据的路径 	
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
 select_feature_num=14;   %特征选择的个数	
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
	
	
	
 %% 另存Py类型
 y_all_predict = [y_train_predict; y_vaild_predict; y_test_predict];
 expected_value2 = mean(y_all_predict_norm);
 expected_value = mean(y_all_predict);
 y_all_pred_all_py = py.numpy.array(y_all_predict);
 y_all_pred_all2_py = py.numpy.array(y_all_predict_norm);
 py.numpy.save('y_all_predict_py.npy', y_all_pred_all2_py);
 %py.numpy.save('y_all_predict2_py.npy', y_all_pred_all_py);
 all_y_feature_label_py = py.numpy.array(y_feature_label);
 py.numpy.save('all_y_feature_label_py.npy',all_y_feature_label_py);
 expected_value_py = py.numpy.array(expected_value2);
 py.numpy.save('expected_value_all_py.npy',expected_value_py);
 
 X_all_norm = [train_x_feature_label_norm; vaild_x_feature_label_norm; test_x_feature_label_norm];
 %shap_values_test = calculate_shap(X_all_norm,Mdl);

 x_feature_label_norm_all=(x_feature_label-x_mu)./x_sig;
 shap_values_all = calculate_shap(x_feature_label_norm_all,Mdl);

 shap_values_all2 = compute_shap_interaction_values(Mdl,x_feature_label_norm_all);

 X_data_py = py.numpy.array(x_feature_label);
  % 在計算 SHAP 值後顯示消息
 X_py_name = strcat('X_py_all2.npy'); 
 py.numpy.save(X_py_name, X_data_py); 


 y_train_pred_all_py = py.numpy.array(y_train_predict); 
 py.numpy.save('y_train_pred_all_py.npy', y_train_pred_all_py);
 train_y_feature_label_py = py.numpy.array(train_y_feature_label);
 py.numpy.save('train_y_feature_label_py.npy',train_y_feature_label_py);
 
 y_vaild_pred_all_py = py.numpy.array(y_vaild_predict);
 py.numpy.save('y_vaild_pred_all_py.npy', y_vaild_pred_all_py);
 vaild_y_feature_label_py = py.numpy.array(vaild_y_feature_label);
 py.numpy.save('vaild_y_feature_label_py.npy',vaild_y_feature_label_py);

 y_test_pred_all_py = py.numpy.array(y_test_predict);
 py.numpy.save('y_test_pred_all_py.npy', y_test_pred_all_py);
 test_y_feature_label_py = py.numpy.array(test_y_feature_label);
 py.numpy.save('test_y_feature_label_py.npy',test_y_feature_label_py);	
	
	
	
	
	
	
	
	
%% K折验证	
x_feature_label_norm_all=(x_feature_label-x_mu)./x_sig;    %x特征	
y_feature_label_norm_all=(y_feature_label-y_mu)/y_sig;	
Kfold_num=G_out_data.Kfold_num;	
cv = cvpartition(size(x_feature_label_norm_all, 1), 'KFold', Kfold_num); % Split into K folds	
for k = 1:Kfold_num	
    trainingIdx = training(cv, k);	
    validationIdx = test(cv, k);	
     x_feature_label_norm_all_traink=x_feature_label_norm_all(trainingIdx,:);	
   y_feature_label_norm_all_traink=y_feature_label_norm_all(trainingIdx,:);	
	
   x_feature_label_norm_all_testk=x_feature_label_norm_all(validationIdx,:);	
   y_feature_label_norm_all_testk=y_feature_label_norm_all(validationIdx,:);	
	
   paramters.maxiter=Mdl.iters_optimal;  paramters.depth_max=Mdl.params.max_depth;  paramters.min_child=Mdl.params.min_child_weight;  %优化的参数	
   Mdlkf= train_xgb(x_feature_label_norm_all_traink,y_feature_label_norm_all_traink,paramters);	
   Mdl_kfold{1,k}=Mdlkf;	
   y_test_predict_norm_all_testk=predict_xgb(Mdlkf,x_feature_label_norm_all_testk);  %测试集预测结果	
   y_test_predict_all_testk=y_test_predict_norm_all_testk*y_sig+y_mu;	
   y_feature_label_all_testk=y_feature_label_norm_all_testk*y_sig+y_mu;	
   test_kfold=sum(abs(y_test_predict_all_testk-y_feature_label_all_testk))/length(y_feature_label_all_testk);% 采用的MAE	
   MAE_kfold(k)=test_kfold;	
	
	
	
end	
	
	
	
% k折验证结果绘图	
figure('color',[1 1 1]);	
	
color_set=[0.4902    0.7686    0.6510];	
plot(1:length(MAE_kfold),MAE_kfold,'--p','color',color_set,'Linewidth',1.3,'MarkerSize',6,'MarkerFaceColor',color_set,'MarkerFaceColor',[0.3,0.4,0.5]);	
grid on;	
box off;	
grid off;	
ylim([0.9*min(MAE_kfold),1.3*max(MAE_kfold)])	
xlabel('kfoldnum')	
ylabel('MAE')	
xticks(1:length(MAE_kfold))	
set(gca,'Xgrid','off');	
set(gca,'Linewidth',1);	
set(gca,'TickDir', 'out', 'TickLength', [.005 .005], 'XMinorTick', 'off', 'YMinorTick', 'off');	
yline(mean(MAE_kfold),'--')	
%小窗口柱状图的绘制	
axes('Position',[0.6,0.65,0.25,0.25],'box','on'); % 生成子图	
HBAR = bar(1:length(MAE_kfold),MAE_kfold,1,'EdgeColor','k');	
HBAR(1).FaceColor = color_set;	
xticks(1:length(MAE_kfold))	
xlabel('kfoldnum')	
ylabel('MAE')	
disp('****************************************************************************************')	
disp([num2str(Kfold_num),'折验证预测MAE平均绝对误差结果：'])  	
disp(MAE_kfold)  	
disp([num2str(Kfold_num),'折验证  ','MAE均值为： ' ,num2str(mean(MAE_kfold)),'     MAE标准差为： ' ,num2str(std(MAE_kfold))])  	
	
%% 绘图块	
color_list=G_out_data.color_list;   %颜色数据库	
rand_list1=G_out_data.rand_list1;   %颜色数据库	
Line_Width=G_out_data.Line_Width;   %线粗细	
makesize=G_out_data.makesize;   %标记大小	
yang_str2=G_out_data.yang_str2;   %符号库	
yang_str3=G_out_data.yang_str3;   %符号库	
kuang_width=G_out_data.kuang_width;   %画图展示数据	
show_num=G_out_data.show_num;   %测试集画图展示数据	
show_num1=G_out_data.show_num1;   %验证集画图展示数据	
show_num2=G_out_data.show_num2;   %训练集画图展示数据	
	
FontSize=G_out_data.FontSize;   %字体设置	
xlabel1=G_out_data.xlabel1;   %	
ylabel1=G_out_data.ylabel1;   %	
title1=G_out_data.title1;   %	
legend1=G_out_data.legend1;   %图例	
box1=G_out_data.box1;   %框	
le_kuang=G_out_data.le_kuang;   %图例框	
grid1=G_out_data.grid1;   %网格	
yang_fu3_ku=G_out_data.yang_fu3_ku;  %总体符号库	
color_index=G_out_data.color_index;	
yangsi_idnex=G_out_data.yangsi_idnex;  %总体样式库 	
  	
  	
	
XX=1:length(train_y_feature_label);	
index_show=1:show_num2;	
	
figure_density(yangsi_idnex(5),y_train_predict(index_show),train_y_feature_label(index_show),'测试集') 	
figure('Position',[200,200,600,350]); 	
plot(gca,XX(index_show),train_y_feature_label(index_show),yang_fu3_ku{1,yangsi_idnex(1)},'Color',color_list(yangsi_idnex(3),:),'LineWidth',Line_Width(1)) 	
hold (gca,'on') 	
%拟合值 	
plot(gca, XX(index_show),y_train_predict(index_show),yang_fu3_ku{1,yangsi_idnex(2)},'Color',color_list(yangsi_idnex(4),:),'LineWidth',Line_Width(1),'MarkerSize',makesize) 	
hold (gca,'on') 	
	
	
set(gca,'FontSize',FontSize,'LineWidth',kuang_width)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
title(gca,'训练集结果')	
legend(gca,legend1) 	
box(gca,box1)	
legend(gca,le_kuang) %图例框消失	
grid(gca,grid1)	
	
	
	
	
	
XX=1:length(vaild_y_feature_label);	
 index_show=1:show_num1;	
	
	
figure_density(yangsi_idnex(5),y_vaild_predict(index_show),vaild_y_feature_label(index_show),'验证集') 	
figure('Position',[200,200,600,350]); 	
plot(gca,XX(index_show),vaild_y_feature_label(index_show),yang_fu3_ku{1,yangsi_idnex(1)},'Color',color_list(yangsi_idnex(3),:),'LineWidth',Line_Width(1)) 	
hold (gca,'on') 	
%拟合值 	
plot(gca, XX(index_show),y_vaild_predict(index_show),yang_fu3_ku{1,yangsi_idnex(2)},'Color',color_list(yangsi_idnex(4),:),'LineWidth',Line_Width(1),'MarkerSize',makesize) 	
hold (gca,'on') 	
	
	
set(gca,'FontSize',FontSize,'LineWidth',kuang_width)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
title(gca,'验证集结果')	
legend(gca,legend1) 	
box(gca,box1)	
legend(gca,le_kuang) %图例框消失	
grid(gca,grid1)	
	
	
XX=1:length(test_y_feature_label);	
index_show=1:show_num;	
figure_density(yangsi_idnex(5),y_test_predict(index_show),test_y_feature_label(index_show),'测试集') 	
figure('Position',[200,200,600,350]); 	
plot(gca,XX(index_show),test_y_feature_label(index_show),yang_fu3_ku{1,yangsi_idnex(1)},'Color',color_list(yangsi_idnex(3),:),'LineWidth',Line_Width(1)) 	
hold (gca,'on') 	
%拟合值 	
plot(gca, XX(index_show),y_test_predict(index_show),yang_fu3_ku{1,yangsi_idnex(2)},'Color',color_list(yangsi_idnex(4),:),'LineWidth',Line_Width(1),'MarkerSize',makesize) 	
hold (gca,'on') 	
	
set(gca,'FontSize',FontSize,'LineWidth',kuang_width)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
title(gca,'测试集结果')	
legend(gca,legend1) 	
box(gca,box1)	
legend(gca,le_kuang) %图例框消失	
grid(gca,grid1)	
	
