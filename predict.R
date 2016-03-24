library(data.table)
library(caret)
library(plyr)
library(tidyr)
library(dplyr)
library(xgboost)

# Read data

# data col 198, field contains '9999999999' - take care of other warnings
train <- fread('data/train.csv',integer64 = "character")
test <- fread('data/test.csv',colClasses=sapply(select(train, -TARGET), class))
train$TARGET <- factor(make.names(train$TARGET)) # values 1/0 will otherwise confuse R
originalTypes <- sapply(test, class)
testIDs <- test$ID

# Data analysis
symCols <- names(test)[!sapply(test, is.numeric)]
for (symcol in symCols) {
  levels <- unique(c(unique(train[[symcol]]),unique(test[[symcol]])))
  train[[symcol]] <- as.integer(factor(train[[symcol]],levels=levels))
  test[[symcol]] <- as.integer(factor(test[[symcol]],levels=levels))
}

# Feature engineering

rowWiseCount <- function(x, const) {
  return( sum(x == const) )
}
train$n0 <- apply(train, 1, FUN=rowWiseCount, 0)
test$n0 <- apply(test, 1, FUN=rowWiseCount, 0)

cat("Train size before variable selection:",dim(train),fill=T)

nzv <- colnames(train)[nearZeroVar(select(train, -TARGET))]
cat("Removed near-zero variables:", length(nzv), 
    "(of", length(names(train)), "):", nzv, fill=T)
train <- train[,!(names(train) %in% nzv), with=F]
test  <- test[,!(names(test) %in% nzv), with=F]

cat("Train size after near-zero variance removal:",dim(train),fill=T)

linearCombos <- colnames(train)[findLinearCombos(select(train, -TARGET))$remove]
cat("Removed linear combinations:", length(linearCombos), 
    "(of", length(names(train)), "):", linearCombos, fill=T)
train <- train[,!(names(train) %in% linearCombos),with=F]
test  <- test[,!(names(test) %in% linearCombos),with=F]

cat("Train size after linear combo removal:",dim(train),fill=T)

trainCor <- cor(select(train, -TARGET), method='spearman')
correlatedVars <- colnames(train)[findCorrelation(trainCor, cutoff = 0.95, verbose = F)]
cat("Removed highly correlated cols:", length(correlatedVars), 
    "(of", length(names(train)), ")", correlatedVars, fill=T)
train <- train[,!(names(train) %in% correlatedVars),with=F]
test  <- test[,!(names(test) %in% correlatedVars),with=F]

cat("Train size after highly correlated var removal:",dim(train),fill=T)

# Var importance (on a sample otherwise way too slow)
daSample <- sample_n(train, size=10000)
allAUC <- filterVarImp(x = select(daSample, -TARGET), y = daSample$TARGET)
varImp <- allAUC %>% select(X0) %>% dplyr::rename(AUC = X0) %>%
  mutate(field=rownames(allAUC)) %>% arrange(desc(AUC)) %>%
  mutate(type = originalTypes[field])
p <- ggplot(head(varImp,40), 
            aes(x=factor(field,levels=field), y=AUC, fill=type))+
  geom_bar(stat="identity")+
  coord_flip()+ggtitle("Univariate Var Importance")+xlab("Feature")+ylab("AUC")+
  scale_y_continuous(labels = scales::percent)
print(p)

# Keep only top-N most important variables
# topN <- c(head(varImp$field,10), "TARGET")
# train <- train[, names(train) %in% topN, with=F]
# test <- test[, names(test) %in% topN, with=F]

# Build model
crossValidation <- trainControl(
  # method = "none", # Fitting models without parameter tuning
  method = "cv", # "repeatedcv" and repeats
  number=5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  verbose=T)

xgbGrid <- expand.grid(nrounds = seq(10,500,by=50),#seq(500,1000,by=200),
                       eta = c(0.02, 0.05, 0.1, 0.2, 0.3), # 0.01
                       max_depth = seq(10),
                       gamma = 2, #c(2, 3, 4, 5), 
                       colsample_bytree = 0.85, 
                       min_child_weight = 5) 

model <- train(TARGET ~ ., 
               data = train, 
               method = "xgbTree"
               ,tuneGrid = xgbGrid
               ,metric = "ROC"
               ,maximize=T
               ,trControl = crossValidation
               ,verbose=T)

# Report on model
print(plot(varImp(model)))

# Report on CV tuning
print(model)
print(ggplot(model))
      
# Score test set
preds <- predict(model, test, type="prob")
submission <- data.frame(ID=testIDs, TARGET=preds[,2])
write.csv(submission, "submission.csv", row.names = F)
