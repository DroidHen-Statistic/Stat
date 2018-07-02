library(RMySQL)
library(party)
library(dplyr)
library(survival)

con <- dbConnect(MySQL(), host = "218.108.40.13", dbname = "wja", username = "wja", password = "wja")
features <- c("churn", "lifetime", "average_bet", "spin_per_active_day", "purchase_times", "average_login_interval","average_spin_interval", "average_bonus_win", "average_day_active_time", "bonus_ratio", "free_spin_ratio","active_ratio", "coin", "active_days")
feature_str <- paste(features, collapse = ', ')
print(feature_str)
sql <- paste("select", feature_str, "from slot_churn_profile", sep = " ")
result <- dbGetQuery(con, sql)

set.seed(1234)  
ind <- sample(2, nrow(result), replace=TRUE, prob=c(0.7, 0.3))
#print(ind)
train <- result[ind==1,]
test <- result[ind==2,]


result <- mutate(result, purchase = purchase_times > 0)

train <- result[ind==1,]
test <- result[ind==2,]


pay_user <- filter(result, purchase_times > 0)
no_pay_user <- filter(result, purchase_times == 0)
#print(nrow(pay_user))

my_fit <- survfit(Surv(lifetime, churn)~purchase, data = result)
print(my_fit)
#plot(my_fit, col = c("red"), main="Kaplan-Meier estimate with 95% confidence bounds", xlab="time", ylab="survival function", lwd = 2)
plot(my_fit, col = c("green", "red"), main="Kaplan-Meier estimate with 95% confidence bounds", xlab="time", ylab="survival function", lwd = 3)
legend("topright", legend = c("not_pay", "pay"), col = c("green", "red"), lty = 1, lwd = 3)
my_tree <- ctree(Surv(lifetime, churn) ~ ., data = train)
my_forest <- cforest(Surv(lifetime, churn)~., data = train)

plot(my_tree, mar = c(0.1,0.1,0.1,0.1), fin = c(20, 20))
pre <- predict(my_forest, newdata = test, type = "response")
#pre_response <- predict(my_forest, newdata = train, type = "prob")
#print(pre_response)
plot(test$lifetime, pre, col = "blue", main="Survival forest", xlab="lifetime", ylab="predict")

print(pre)
dbDisconnect(con)