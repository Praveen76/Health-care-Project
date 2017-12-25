#Set workking directory and load libraries and data
rm(list=ls())
setwd("C:/R/Projects/Text Classification - HealthCare")

Needed <- c("tm", "RColorBrewer", "ggplot2", "wordcloud", "doSNOW","slam","magrittr","e1071","xgboost",
       "parallel","parallelMap","plyr","stringr","dplyr","stringr", "syuzhet","Matrix","data.table","caret","cluster", "igraph", "fpc","mlr")

# install.packages(Needed, dependencies=TRUE)#Install Packages

lapply(Needed,require,character.only=TRUE) #Load Libraries
rm(Needed)

data <- fread("C:/R/Projects/Text Classification - HealthCare/TextClassification_Data.csv", sep=",", stringsAsFactors = FALSE)
class(data)
str(data)

############################# EDA(Exploratory Data Analysis) #############################
#Understanding of Data
head(data)
dim(data)
str(data)
colSums(is.na(data)) #Check missing values

cat("The number of dupicated rows are", nrow(data)-nrow(unique(data)))
############Feature Engineering ########
# 1.UniVariate Analysis
str(data)
str(data$previous_appointment)
table(data$previous_appointment)

data$previous_appointment[data$previous_appointment %in% c("YES","yes","Yes")]="Yes"
data$previous_appointment[data$previous_appointment %in% c("No","NO","")]="No"
table(data$previous_appointment)
str(data)
#Just another way of merging two variables
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
data$ID <- as.factor(data$ID)
str(data)

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

data1<- data$DATA #Backup of original data

###Convert in to Corpus
txt_corpus <- Corpus(VectorSource(data$DATA)) 
writeLines(as.character(txt_corpus[1])) #Analyze
writeLines(as.character(txt_corpus[3])) #Analyze


#Time the code execution start
start.time<-Sys.time()
#Create a cluster to work on 4 logical cores
cl<- makeCluster(3,type="SOCK") #Socket cluster creates mutltiple instances of R-Studio
registerDoSNOW(cl)

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

stopCluster(cl)
#Total time of execution on workstation
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

# ########## Frequent Terms and Associations#########3
##Frequent terms
findFreqTerms(dtm_review, lowfreq=200)
frequent<-findFreqTerms(dtm_review, lowfreq=100)
findFreqTerms(dtm_review, lowfreq=10)

Data2<-data  #Backup the data
data$ID<- as.character(data$ID)
str(data)

##################Xgboost Model for Categories #########
# To sparse matrix
varnames <- setdiff(colnames(data), c("ID","sub_categories","categories","DATA","SUMMARY"))
varnames

str(data)

cd_train<- as.data.frame(sapply(data[ID <"2015_5_6131_1001", varnames, with=FALSE],as.numeric))
class(cd_train)

train_categories<-data$categories[data$ID <"2015_5_6131_1001"]
cd_train<- cbind(cd_train,train_categories)
head(cd_train)

cd_test<-as.data.frame(sapply(data[ID >"2015_5_6131_1001", varnames, with=FALSE],as.numeric))
class(cd_test)

test_categories<-data$categories[data$ID >"2015_5_6131_1001"]
cd_test<- cbind(cd_test,test_categories)

ID<- data$ID[data$ID >"2015_5_6131_1001"]
head(ID)

#Create Task
train_task<- makeClassifTask(data = cd_train,target = "train_categories")#, cols = NULL)
head(train_task)
test_task<- makeClassifTask(data = cd_test,target = "test_categories")


#do one hot encoding`
train_task <- createDummyFeatures(obj = train_task)
str(train_task)
train_task$task.desc$class.levels
test_task <- createDummyFeatures(obj = test_task )
test_task$task.desc$class.levels

#Make Learner with inititial parameters
Xg_set<-makeLearner("classif.xgboost",predict.type = "response")
Xg_set$par.vals <- list(Objective="multi:softprob",eval.metric="mlogloss",nrounds=100, eta=0.1)

#define Parameters for tuning
xg_ps<- makeParamSet(
  makeIntegerParam("nrounds",lower = 200, upper = 500),
  makeIntegerParam("max_depth",lower = 3,upper = 5),
  makeNumericParam("gamma",lower=0.8,upper= 1),
  makeNumericParam("min_child_weight",lower = 1,upper = 3),
  makeNumericParam("subsample",lower = .5,upper=1),
  makeNumericParam("colsample_bytree",lower = .5 ,upper = 1)
)

#Define search function
ran_control<- makeTuneControlRandom(maxit = 3)

#3 fold CV
set_cv<- makeResampleDesc('CV',stratify = T,iters=3L) #With stratify=T, we'll ensure that distribution of target class is maintained in the resampled data sets.

#set parallel backend
# parallelStartSocket(cpus = detectCores())

# #Time the code execution start
start.time<-Sys.time()
# #Create a cluster to work on 3 logical cores
cl<- makeCluster(4,type="SOCK") #Socket cluster creates mutltiple instances of R-Studio
registerDoSNOW(cl)

#Tune Parameter
mytune<- tuneParams(learner = Xg_set,task=train_task,resampling = set_cv,measures = acc,par.set = xg_ps,control=ran_control,show.info = T)

mytune$y
mytune$x

stopCluster(cl)
#Total time of execution on workstation
total.time<- Sys.time()- start.time
total.time

#Set Parameters
Xg_New<- setHyperPars(learner = Xg_set , par.vals = mytune$x)

#Train Model
Xg_Model <- train(Xg_New , train_task)


saveRDS(Xg_Model, "Xg_Model.rds")##Save Model
mod <- readRDS("Xg_Model.rds") ##Load Model

#Test_Model
result<- predict(mod,test_task)

categories<- result$data$response
head(categories)
table(categories)

#Confusion Matrix
table(cd_test$test_categories,result$data$response)
Accuracy=(2764+2470+576+2010+3367)/(2764+2470+576+2010+3367+613+57+677+148+465+81+417+539+316+285+381+82+1109+613+114+400+379+669+26+596)*100

# ((2764+2470+576+2010+3367)/(2764+2470+576+2010+3367+613+57+677+148+465+81+417+539+316+285+381+82+1109+613+114+400+379+669+26+596))

################Xgboost Model for Sub_Categories#####

cd_train_1<- as.data.frame(sapply(data[ID <"2015_5_6131_1001", varnames, with=FALSE],as.numeric))
class(cd_train_1)

train_subcategories<-data$sub_categories[data$ID <"2015_5_6131_1001"]
cd_train_1<- cbind(cd_train_1,train_subcategories)

cd_test_1<-as.data.frame(sapply(data[ID >"2015_5_6131_1001", varnames, with=FALSE],as.numeric))
class(cd_test_1)
test_Subcategories<-data$sub_categories[data$ID >"2015_5_6131_1001"]
cd_test_1<- cbind(cd_test_1,test_Subcategories)

#Create Task
train_task<- makeClassifTask(data = cd_train_1,target = "train_subcategories")
test_task<- makeClassifTask(data = cd_test_1,target = "test_Subcategories")


#do one hot encoding
train_task <- createDummyFeatures(obj = train_task)
str(train_task)

test_task <- createDummyFeatures (obj = test_task)


#Make Learner with inititial parameters
Xg_set_1<-makeLearner("classif.xgboost",predict.type = "response")
Xg_set_1$par.vals <- list(Objective="multi:softprob",eval.metric="mlogloss",nrounds=100, eta=0.1)

#define Parameters for tuning
xg_ps_1<- makeParamSet(
  makeIntegerParam("nrounds",lower = 200, upper = 500),
  makeIntegerParam("max_depth",lower = 3,upper = 5),
  makeNumericParam("gamma",lower=0.8,upper= 1),
  makeNumericParam("min_child_weight",lower = 1,upper = 3),
  makeNumericParam("subsample",lower = .5,upper=1),
  makeNumericParam("colsample_bytree",lower = .5 ,upper = 1)
)

#Define search function
ran_control_1<- makeTuneControlRandom(maxit = 3)

#3 fold CV
set_cv_1<- makeResampleDesc('CV',stratify = T,iters=3L) #With stratify=T, we'll ensure that distribution of target class is maintained in the resampled data sets.

# #Time the code execution start
start.time<-Sys.time()
# #Create a cluster to work on 3 logical cores
cl<- makeCluster(4,type="SOCK") #Socket cluster creates mutltiple instances of R-Studio
registerDoSNOW(cl)

#Tune Parameter
mytune_1<- tuneParams(learner = Xg_set_1,task=train_task,resampling = set_cv_1,measures = acc,par.set = xg_ps_1,control=ran_control_1,show.info = T)

mytune_1$y
mytune_1$x

stopCluster(cl)
#Total time of execution on workstation
total.time<- Sys.time()- start.time
total.time

#Set Parameters
Xg_New_1<- setHyperPars(learner = Xg_set_1 , par.vals = mytune_1$x)

#Train Model
Xg_Model_1 <- train(Xg_New_1 ,train_task)

saveRDS(Xg_Model_1, "Xg_Model_1.rds")##Save Model
mod1 <- readRDS("Xg_Model_1.rds") ##Load Model

#Test_Model
result1<- predict(mod1,test_task)
sub_categories<- result1$data$response

#Confusion Matrix
table_subcategories<-table(cd_test_1$test_Subcategories,result1$data$response)
write.csv(table_subcategories,"table_subcategories_Xgboost_tuning.csv") #So we can see confusion matrix easily



Final_result<-as.data.frame(cbind(ID,cbind(result$data$response,result1$data$response)))
names(Final_result) <-c("ID","categories","sub_categories")
head(Final_result)
Final_result$categories = mapvalues(Final_result$categories , from = c("1" ,"2","3","4","5"),
                                to = c("APPOINTMENTS","ASK_A_DOCTOR","LAB","MISCELLANEOUS","PRESCRIPTION"))

Final_result$sub_categories = mapvalues(Final_result$sub_categories , from = c("1" ,"2","3","4","5","6","7",
                                        "8","9","10","11","12","13","14","15","16","17","18","19","20"),
to = c("CANCELLATION","CHANGE OF HOSPITAL","CHANGE OF PHARMACY ","CHANGE OF PROVIDER","FOLLOW UP ON PREVIOUS REQUEST",
"LAB RESULTS","MEDICATION RELATED","NEW APPOINTMENT","OTHERS","PRIOR AUTHORIZATION","PROVIDER","QUERIES FROM INSURANCE FIRM",
"QUERIES FROM PHARMACY","QUERY ON CURRENT APPOINTMENT","REFILL","RESCHEDULING ","RUNNING LATE TO APPOINTMENT",
"SHARING OF HEALTH RECORDS (FAX, E-MAIL, ETC.)","SHARING OF LAB RECORDS (FAX, E-MAIL, ETC.)","SYMPTOMS"))

write.csv(Final_result,"Final_result_XgBoost with Hyper Parameter tuning.csv")


