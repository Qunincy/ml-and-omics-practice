# 一、数据导入
# 1. 加载所需包
library(tidyverse)     
library(caret)         
library(rpart)
library(rpart.plot)    
library(pROC)          

train_data <- read_csv("data/train.csv",locale = locale(encoding = "UTF-8"))
cat("=== train.csv数据集基本信息 ===\n")
dim(train_data) 
colSums(is.na(train_data))
str(train_data)

#二、数据预处理
# 1. 数据清洗与特征工程函数（仅依赖train_data自身统计量）
preprocess_train <- function(data) {
  processed <- data %>%
    select(-PassengerId, -Name, -Ticket, -Cabin) %>%
    mutate(
      Age = ifelse(is.na(Age), median(data$Age, na.rm = TRUE), Age),
      Embarked = ifelse(is.na(Embarked), 
                        names(sort(table(data$Embarked), decreasing = TRUE)[1]), 
                        Embarked)
    ) %>%
    mutate(
      Sex = as.factor(Sex),
      Embarked = as.factor(Embarked),
      Pclass = as.factor(Pclass),  
      Survived = factor(Survived, levels = c("0", "1"), labels = c("No", "Yes")) 
    ) %>%
    mutate(FamilySize = SibSp + Parch + 1) %>%  # +1包含乘客本人
    select(-SibSp, -Parch)
  
  return(processed)
}
train_processed <- preprocess_train(train_data)
cat("\n=== 预处理后数据缺失值统计 ===\n")
colSums(is.na(train_processed))
cat("\n=== 预处理后数据结构 ===\n")
str(train_processed)

#三、数据集拆分
set.seed(42)
## 分层采样拆分
train_index <- createDataPartition(
  y = train_processed$Survived,  
  p = 0.8,                     
  list = FALSE,       
  times = 1                     
)
new_train <- train_processed[train_index, ]  # 新训练集（约712条样本）
new_test <- train_processed[-train_index, ]  # 测试集（约179条样本）
## 验证分层效果
cat("\n=== 留出法拆分后存活比例验证 ===\n")
cat("原始训练集存活比例：", round(mean(train_processed$Survived == "Yes"), 3), "\n")
cat("新训练集存活比例：", round(mean(new_train$Survived == "Yes"), 3), "\n")
cat("测试集存活比例：", round(mean(new_test$Survived == "Yes"), 3), "\n")

#四、模型一：逻辑回归
lr_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
## 训练逻辑回归模型
lr_model <- train(
  Survived ~ .,
  data = new_train,
  method = "glm",
  trControl = lr_control,
  family = "binomial"
)
cat("=== 逻辑回归模型结果 ===\n")
print(lr_model)
#模型一评估
lr_pred <- predict(lr_model, newdata = new_test)
lr_pred_prob <- predict(lr_model, newdata = new_test, type = "prob")
## 混淆矩阵评估（准确率、灵敏度、特异度，无索引错误）
cat("=== 逻辑回归模型测试集评估 ===\n")
conf_mat <- confusionMatrix(lr_pred, new_test$Survived)
cat("测试集准确率：", round(conf_mat$overall["Accuracy"], 3), "\n")
by_class_metrics <- conf_mat$byClass
cat("灵敏度（召回率）：", round(by_class_metrics["Sensitivity"], 3), "\n")
cat("特异度：", round(by_class_metrics["Specificity"], 3), "\n")
## ROC曲线与AUC值
roc_obj <- roc(new_test$Survived, lr_pred_prob$Yes)
auc_value <- auc(roc_obj)
cat("测试集AUC值：", round(auc_value, 3), "\n")

#五、模型二：决策树
# 1. 定义交叉验证控制参数
dt_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
# 2. 定义决策树调优参数网格
dt_param <- expand.grid(cp = c(0.01,0.05,0.1))
# 3. 训练决策树模型
dt_model <- train(
  Survived ~ .,
  data = new_train,
  method = "rpart",
  trControl = dt_control,
  tuneGrid = dt_param,
  control = rpart.control(minsplit = 20),
  metric = "ROC"
)
# 4. 查看决策树模型结果
cat("=== 决策树模型结果 ===\n")
print(dt_model)
print(dt_model$bestTune)  # 最优cp
# 5. 决策树可视化（直观展示树结构）
rpart.plot(dt_model$finalModel, main = "泰坦尼克号生存预测决策树")

#模型二检验
dt_pred <- predict(dt_model, newdata = new_test)  # 预测生存类别（No/Yes）
dt_pred_prob <- predict(dt_model, newdata = new_test, type = "prob")  # 预测类别概率
## 混淆矩阵评估（准确率、灵敏度、特异度）
cat("=== 决策树模型测试集检验结果 ===\n")
conf_mat <- confusionMatrix(dt_pred, new_test$Survived)
cat("测试集准确率：", round(conf_mat$overall["Accuracy"], 3), "\n")
cat("灵敏度（召回率）：", round(conf_mat$byClass["Sensitivity"], 3), "\n")
cat("特异度：", round(conf_mat$byClass["Specificity"], 3), "\n")
## ROC曲线与AUC值（二分类核心评估指标）
roc_obj <- roc(new_test$Survived, dt_pred_prob$Yes)
auc_value <- auc(roc_obj)
cat("测试集AUC值：", round(auc_value, 3), "\n")

#六、模型对比
lr_acc=round(confusionMatrix(predict(lr_model,new_test),new_test$Survived)$overall["Accuracy"],3)
dt_acc=round(confusionMatrix(predict(dt_model,new_test),new_test$Survived)$overall["Accuracy"],3)
cat("最优模型：",if(lr_acc>dt_acc)"逻辑回归"else"决策树","(准确率：",max(lr_acc,dt_acc),")\n")
best_model <- if(lr_acc>dt_acc) lr_model else dt_model
