#Set workking directory and load libraries and data
rm(list=ls())
setwd("C:/R/Projects/Text Classification - HealthCare")
library(dplyr)
library(plyr)
library(stringr)
Needed <- c("tm", "RColorBrewer", "ggplot2", "wordcloud", "doSNOW","slam","magrittr","e1071","xgboost",
          "Matrix","data.table","caret","cluster", "igraph", "fpc","mlr")

# install.packages(Needed, dependencies=TRUE)#Install Packages

lapply(Needed,require,character.only=TRUE) #Load Libraries
rm(Needed)

data <- fread("C:/R/Projects/Text Classification - HealthCare/TextClassification_Data.csv", sep=",", stringsAsFactors = FALSE)
class(data)
str(data)
setwd("C:/R/Driftter/Project")
############################# EDA(Exploratory Data Analysis) #############################
#Understanding of Data
head(data)
dim(data)
str(data)
summary(data)
# summarizeColumns(data) #Summarize 
colSums(is.na(data)) #Check missing values


# cat("The number of dupicated rows are", nrow(data)-nrow(unique(data)))
############Feature Engineering ########
# 1.UniVariate Analysis
str(data)
str(data$previous_appointment)
table(data$previous_appointment)


data$previous_appointment[data$previous_appointment %in% c("YES","yes","Yes")]="Yes"
data$previous_appointment[data$previous_appointment %in% c("No","NO","")]="No"
table(data$previous_appointment)

str(data)
#Another way of merging two variables
data$categories =  mapvalues(data$categories,from = c("asK_A_DOCTOR","mISCELLANEOUS","JUNK"),
                             to = c("ASK_A_DOCTOR","MISCELLANEOUS","MISCELLANEOUS"))
data$sub_categories = mapvalues(data$sub_categories, from = c("mEDICATION RELATED","JUNK"),
                                to = c("MEDICATION RELATED","OTHERS"))
# summarizeColumns(data) #Summarize values

table(data$categories)

##Just Analyzing data
data[data$sub_categories=="CHANGE OF PHARMACY","SUMMARY"]
data[data$sub_categories=="CHANGE OF HOSPITAL","SUMMARY"]
data[data$sub_categories=="CHANGE OF PROVIDER","SUMMARY"]
data[data$sub_categories=="QUERY ON CURRENT APPOINTMENT","SUMMARY"]
data[data$sub_categories=="OTHERS","SUMMARY"]
data[data$sub_categories=="JUNK","SUMMARY"]
data[data$sub_categories=="OTHERS" & data$SUMMARY=="Phone Note","fileid" ]

data$categories<- as.factor(data$categories)
data$sub_categories<- as.factor(data$sub_categories)
data$previous_appointment<- as.factor(data$previous_appointment)
data$fileid<- as.numeric(data$fileid)

table(data$sub_categories)

data$nchar <- as.numeric(nchar(data$DATA))
str(data)
data$nwords <- as.numeric(str_count(data$DATA, "\\S+"))

hist(data$nwords) # Visulaize words length

#Visualize words by their categories
ggplot(data, aes(nwords,fill=categories))+
  geom_histogram(binwidth = 6)

#Visualize words by their sub_categories
ggplot(data, aes(nwords,fill=sub_categories))+
  geom_histogram(binwidth = 6)



data$fileid<- NULL
data$nchar<- NULL
data$nwords<- NULL

# LabelCount Encoding for Categories and sub_categories


labelCountEncoding <- function(column){
  return(match(column,levels(column)[order(summary(column,maxsum=nlevels(column)))]))
}

##match(data$categories,levels(data$categories)[order(summary(data$categories, maxsum=nlevels(data$categories)))])##

data$categories <- labelCountEncoding(data$categories)
data$sub_categories <- labelCountEncoding(data$sub_categories)
data$previous_appointment <- labelCountEncoding(data$previous_appointment)


str(data)
names(data) ##Check Column names of data

table(data$sub_categories)
data$DATA[1]
data$DATA[7]

data1<- data$DATA #Backup of original data

###Convert in to Corpus
txt_corpus <- Corpus(VectorSource(data$DATA)) 
writeLines(as.character(txt_corpus[1])) #Analyze
writeLines(as.character(txt_corpus[3])) #Analyze


#Method1:-set parallel backend/ use CPU Cores
parallelStartSocket(cpus = detectCores())

####Method2:-Another way of using CPU Cores
#Time the code execution start
start.time<-Sys.time()
#Create a PARLLEL SOCKET Cluster
cl<-makeCluster(4, type="PSOCK")

for (j in seq(txt_corpus)) {
  txt_corpus[[j]] <- gsub("[a-z][\\\\]", " ", txt_corpus[[j]])
  txt_corpus[[j]] <- gsub("\\}", " ", txt_corpus[[j]])
  txt_corpus[[j]] <- gsub(";", " ", txt_corpus[[j]])
  txt_corpus[[j]] <- gsub("\\d+", " ", txt_corpus[[j]]) #Replace numbers with blank space
  txt_corpus[[j]] <- gsub("\\|", " ", txt_corpus[[j]])
  txt_corpus[[j]] <- gsub("xxxx-xxxx", " ", txt_corpus[[j]])
  txt_corpus[[j]] <- gsub("\\|", " ", txt_corpus[[j]])
  txt_corpus[[j]] <- gsub("\u2028", " ", txt_corpus[[j]])  # This is an ascii character that did not translate, so it had to be removed.
}


total.time<- Sys.time()- start.time
total.time


writeLines(as.character(txt_corpus[1])) #Analyze data
writeLines(as.character(txt_corpus[3]))
######Text cleaning######
txt <- tm_map(txt_corpus, content_transformer(tolower))
txt <- tm_map(txt, stripWhitespace)
txt <- tm_map(txt, removePunctuation)
txt <- tm_map(txt, removeWords, stopwords("english"))
txt <- tm_map(txt, stemDocument, language="english")
txt <- tm_map(txt, removeNumbers)

writeLines(as.character(txt[1])) #Analyze
writeLines(as.character(txt[3]))

## Remove additional words
txt <- tm_map(txt, removeWords, c("cf", "cb","f","xxxxxxx", "margbsxn","ftnbjfonttblf","margtsxn","phonexxxxxxx","sscharaux","pa","b", "fs", " par"))

writeLines(as.character(txt[1])) #Analyze
writeLines(as.character(txt[3]))

##Convert in to DTM
#use the tf-idf(term frequency-inverse document frequency) instead of the frequencies of the term as entries
# tf-idf measures the relative importance of a word to a document.

dtm <- DocumentTermMatrix(txt, control = list(weighting = weightTfIdf))
dtm
dtm_review <- removeSparseTerms(dtm, 0.80)
dtm
inspect(dtm_review)
data <- cbind(data, as.matrix(dtm_review))

########## Frequent Terms and Associations#########3
##Frequent terms
findFreqTerms(dtm_review, lowfreq=200)
frequent<-findFreqTerms(dtm_review, lowfreq=100)
findFreqTerms(dtm_review, lowfreq=10)

##################Xgboost Model for Categories #########
 # To sparse matrix
 varnames <- setdiff(colnames(data), c("ID","categories","sub_categories","SUMMARY","DATA"))
 varnames
 sapply(data[ID <"2015_5_6131_1001", varnames, with=FALSE],as.numeric) #Just for check
 str(sapply(data[ID <"2015_5_6131_1001", varnames, with=FALSE],as.numeric)) #Just analyzing
 
 
 str(data)
 train_sparse <- Matrix(as.matrix(sapply(data[ID <"2015_5_6131_1001", varnames, with=FALSE],as.numeric)), sparse=TRUE)
 str(train_sparse)
 test_sparse <- Matrix(as.matrix(sapply(data[ID >"2015_5_6131_1001", varnames, with=FALSE],as.numeric)), sparse=TRUE)
 str(test_sparse)
 
 y_train <- data[ID <"2015_5_6131_1001",categories]-1
 y_train
 unique(sort(y_train))
 test_ids <- data[ID >"2015_5_6131_1001",ID] 
 test_ids
 dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
 y_test<-data[ID >"2015_5_6131_1001",categories]-1
 dtest <- xgb.DMatrix(data=test_sparse , label=y_test)
 dtest
 nc<-length(unique(y_train))
 nc
 # Params for xgboost
 param <- list(booster = "gbtree",
               objective = "multi:softprob",
               eval_metric = "mlogloss",
               num_class = nc,
               min_child_weight = 1,
               subsample = .5,
               colsample_bytree = .5,
               missing=NA,
               seed=333
 )
 cvFoldsList <- createFolds(y_train, k=5, list=TRUE, returnTrain=FALSE)
 
 #Time the code execution start
 start.time<-Sys.time()
 #Create a PARLLEL SOCKET Cluster
 cl<-makeCluster(3, type="PSOCK")
 
 xgb_cv <- xgb.cv(data = dtrain,
                  params = param,
                  nrounds = 1000,
                  maximize = FALSE,
                  prediction = TRUE,
                  folds = cvFoldsList,
                  print.every.n = 5,
                  early.stop.round = 100
 )
 
 start.time_1<-Sys.time()
 
 xgb_model <- xgb.train(data = dtrain,
                        params = param,
                        watchlist = list(train = dtrain,test=dtest),
                        nrounds =  1000,
                        eta = 0.01,
                        verbose = 1,
                        gamma = 1,
                        max_depth = 6,
                        print.every.n = 5)
 
 stopCluster(cl)
 #Total time taken for cross validation and trian model
 Sys.time()- start.time #Time taken for CV
 Sys.time()- start.time_1 #Time taken to train Model

 

 
 # Training and Test error plot
 e<- data.frame(xgb_model$evaluation_log)
 plot(e$iter,e$train_mlogloss , col="blue")
 lines(e$iter,e$test_mlogloss , col="red")
 min(e$test_mlogloss)
 # e[e$test_mlogloss==1.03606,]
 
 #Use nrounds where error curve has minimum (50 in this case)
 plot(log(xgb_cv$evaluation_log$test_mlogloss_mean), type = 'l')
 

 #Feature imortance
 imp<- xgb.importance(colnames(dtrain),model=xgb_model)
 print(imp)
 xgb.plot.importance(imp[1:20,])

 #Predict the test values 
 p<- predict(xgb_model,newdata=dtest)
 
 head(p)
 
 
 pred<- matrix(p,nrow=nc,ncol=length(p)/nc) %>%
   t() %>%
   data.frame() %>%
   mutate(label=y_test,max_prob=max.col(.,"last")-1)

  pred$label
 
 #Confusion Matrix
 table(Prediction=pred$max_prob,Actual=pred$label)
 
 
 xgb.save(xgb_model, "xgboost.model") #Save Model
 saved_model<-xgb.load("xgboost.model") # load model to R
 p1 <- predict(saved_model, dtest)
 
 pred1<- matrix(p1,nrow=nc,ncol=length(p1)/nc) %>%
   t() %>%
   data.frame() %>%
   mutate(label=y_test,max_prob=max.col(.,"last")-1)
 
 
 # And now the test to check if loaded model is behaving properly
 print(paste("sum(abs(pred1-pred))=", round(sum(abs(pred1-pred)),digits = 0)))
 
 
 
 #######Xgboost Model for Subcategories 
 

label<- pred$label

cat_label<-as.data.frame(cbind(test_ids,label))
 
y_train1<- data[ID <"2015_5_6131_1001",sub_categories]-1
y_train1

dtrain1 <- xgb.DMatrix(data=train_sparse, label=y_train1)
y_test1<-data[ID >"2015_5_6131_1001",sub_categories]-1
dtest1 <- xgb.DMatrix(data=test_sparse , label=y_test1)
dtest1

nc1<-length(unique(y_train1))
nc1
 
 
# Params for xgboost
param1 <- list(booster = "gbtree",
              objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = nc1,
              min_child_weight = 1,
              subsample = .5,
              colsample_bytree = .5,
              missing=NA,
              seed=333
)
cvFoldsList1 <- createFolds(y_train, k=20, list=TRUE, returnTrain=FALSE)

#Time the code execution start
start.time<-Sys.time()
#Create a PARLLEL SOCKET Cluster
cl<-makeCluster(3, type="PSOCK")

xgb_cv1 <- xgb.cv(data = dtrain1,
                 params = param1,
                 nrounds = 1000,
                 maximize = FALSE,
                 prediction = TRUE,
                 folds = cvFoldsList,
                 print.every.n = 5,
                 early.stop.round = 100
)

start.time_1<-Sys.time()

xgb_model1 <- xgb.train(data = dtrain1,
                       params = param1,
                       watchlist = list(train = dtrain1,test=dtest1),
                       nrounds =  1000,
                       eta = 0.1,
                       verbose = 1,
                       gamma = 1,
                       max_depth = 5,
                       print.every.n = 5)

stopCluster(cl)
#Total time taken for cross validation and trian model
Sys.time()- start.time #Time taken for CV
Sys.time()- start.time_1 #Time taken to train Model

# Training and Test error plot
e1<- data.frame(xgb_model1$evaluation_log)
plot(e$iter,e$train_mlogloss , col="blue")
lines(e$iter,e$test_mlogloss , col="red")
min(e1$test_mlogloss)
# e1[e1$test_mlogloss==1.038322,]
 

#Feature imortance
imp<- xgb.importance(colnames(dtrain1),model=xgb_model1)
# print(imp)
xgb.plot.importance(imp[1:20,])

#Predict Test Values
p2<- predict(xgb_model1,newdata=dtest1)

head(p2)

pred2<- matrix(p2,nrow=nc1,ncol=length(p2)/nc1) %>%
  t() %>%
  data.frame() %>%
  mutate(label=y_test1,max_prob=max.col(.,"last")-1)

pred2$label

#confusion Matrix
table(Prediction=pred2$max_prob,Actual=pred2$label)

xgb.save(xgb_model1, "xgboost.model_1") #Save Model
saved_model_1<-xgb.load("xgboost.model_1") # load model to R

#predict using the saved model
p3 <- predict(saved_model_1, dtest1)

pred3<- matrix(p3,nrow=nc1,ncol=length(p3)/nc1) %>%
  t() %>%
  data.frame() %>%
  mutate(label=y_test1,max_prob=max.col(.,"last")-1)




# And now the test
print(paste("sum(abs(pred3-pred2))=", round(sum(abs(pred3-pred2)),digits=0)))


label1<- pred2$label
label1

subcat_label<-as.data.frame(cbind(test_ids,label1))


final_result<-merge(subcat_label,cat_label,by="test_ids")
final_result<-final_result[,c(1,3,2)]

names(final_result)<- c("test_ids","Category","Sub_category")
class(final_result)
head(final_result)
#Write result in to CSV file
write.csv( final_result,"final_result_Xgboost.csv", quote=FALSE, row.names=FALSE)


 
