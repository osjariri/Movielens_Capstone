
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
#if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")
#tinytex:::install_prebuilt()
#tinytex::install_tinytex()

if(!require(tidyr)) install.packages("tidyr")
library(dplyr)
library(tidyverse)
library(tidyr)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Introduction
# In this project we will build recommendation system for rating movies, we will use ratings given by users for movies, and recommend movies to users as per their ratings to other movies; it will predict the movies rating for a specific user, and movies with high rating will be recommended to the user. Our goal in this project is to minimise the RMSE value to be less than 0.8649.

# In this project we will use movielens dataset, we will use the 10M version of the datset. each row in the dataset represents one rating given by one user to one movie, and has the columns 'Userid', "Movieid", "Title", "Rating", "TimeStamp" and "Genres".
# The relation between users and movies is many to many, implies that one user can rate many movies and one movie can be rated by many users.  

# We divided the dataset to training and validation data, and start to evaluate the RMSE using the below models  
# - Using Movies effect model  
# - Using Movies and users effect model  
# - Using Regularized movie effect model  
# - Using Regularized movie effect and user effect model  


# edx Dataset


head(edx)


movies_Number <- nrow(distinct(edx,edx$movieId))

users_number <- nrow(distinct(edx,edx$userId))



m_u_counts <- data_frame( Data = "Movies", Count = as.character(movies_Number))
m_u_counts <- bind_rows(m_u_counts, data.frame( Data = "Users", Count = as.character(users_number)))
m_u_counts %>% knitr::kable()




edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30) + 
  scale_x_log10() + 
  ggtitle("Movies")


edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30) + 
  scale_x_log10() + 
  ggtitle("Users")



edx %>% 
  ggplot(aes(x=rating)) + 
  geom_histogram(bins=30) +
  ggtitle("Ratings")




RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

mu_hat <- mean(edx$rating)
mu_hat

naive_RMSE <- RMSE(validation$rating, mu_hat)
naive_RMSE


testRMSE <- RMSE(validation$rating, 2.5)
testRMSE


rmse_results <- data_frame(method =  "Just the Average", RMSE = naive_RMSE)

paste('Using the "Average" effect model we have achived a RMSE of value',naive_RMSE)


#Average Rating with Movie effect Model

mu <- mean(edx$rating)
movie_avgs <- edx %>%
  group_by(movieId) %>% 
  summarise(b_i= mean(rating-mu))



movie_avgs %>% ggplot(aes(b_i)) +
  geom_histogram(bins=10)




predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

movie_effect_RMSE <- RMSE(predicted_ratings, validation$rating)




rmse_results <- bind_rows(rmse_results, data.frame(method= "Movie Effect Model", RMSE= movie_effect_RMSE))


paste('Using the "movie" effect model we have achived a RMSE of value',movie_effect_RMSE)


#Average Rating with Movie and User effect Model 

edx %>% group_by(userId) %>% select(userId,rating) %>% count(userId) %>% filter(n>100) %>%
  left_join(edx,by='userId') %>% summarize(avg=mean(rating)) %>% ggplot(aes(avg)) + 
  geom_histogram(bins=30) +
  ggtitle("Ratings")


users_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>% 
  summarise(b_u= mean(rating - mu - b_i))

users_avgs %>% ggplot(aes(b_u)) +
  geom_histogram(bins=10)




predicted_ratings <-  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(users_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred



movie_user_effect_RMSE <- RMSE(predicted_ratings, validation$rating)




rmse_results <- bind_rows(rmse_results, data.frame(method= "Movie + User Effect Model", RMSE= movie_user_effect_RMSE))

paste('Using the "movie + User" effect model we have achived a RMSE of value',movie_user_effect_RMSE)



# Regularization  


edx %>%
  left_join(movie_avgs, by='movieId') %>% mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual ))) %>% select(title,residual) %>% slice(1:10) %>% knitr::kable()


movie_titles <- edx %>% select(movieId, title) %>% distinct()
movie_avgs %>% left_join(movie_titles,by='movieId') %>% arrange(desc(b_i)) %>%select(title, b_i) %>%
  slice(1:10) %>% knitr::kable()


movie_avgs %>% left_join(movie_titles,by='movieId') %>% arrange(b_i) %>%select(title, b_i) %>%
  slice(1:10) %>% knitr::kable()


edx %>% count(movieId) %>% left_join(movie_avgs) %>% left_join(movie_titles,by='movieId') %>%
  arrange(desc(b_i)) %>% select(title, b_i, n) %>% slice(1:10) %>% knitr::kable()


edx %>% count(movieId) %>% left_join(movie_avgs) %>% left_join(movie_titles,by='movieId') %>%
  arrange(b_i) %>% select(title, b_i, n) %>% slice(1:10) %>% knitr::kable()



# Regulization with movie effect


# preparing the training data and test data the will use cross validation to determine the best lambda.

set.seed(1, sample.kind="Rounding")

test_index_Reg <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_Reg <- edx[-test_index_Reg,]
temp_Reg <- edx[test_index_Reg,]


test_Reg <- temp_Reg %>% 
  semi_join(edx_Reg, by = "movieId") 


removed_Reg <- anti_join(temp_Reg, test_Reg)
edx_reg <- rbind(edx_Reg, removed_Reg)

rm(test_index_Reg, temp_Reg, removed_Reg)


RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

lambdas <- seq(0, 5, 0.25)

rmses <- sapply(lambdas,function(l){
  
  
  mu <- mean(edx_Reg$rating)
  
  
  b_i <- edx_Reg %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  
  predicted_ratings <- 
    test_Reg %>% 
    left_join(b_i, by = "movieId") %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  
  return(RMSE(predicted_ratings, test_Reg$rating))
})

plot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]
paste('Optimal RMSE of',min(rmses),'is achieved with Lambda',lambda)

# Now we will use the lambda to calculate the RMSE using the **Validation** data

l <- lambda


mu <- mean(edx$rating)

reg_movie_avgs <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

predicted_ratings <- 
  validation %>% 
  left_join(reg_movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  .$pred #validation


finalRMSE_movie = RMSE(predicted_ratings, validation$rating)



edx %>% count(movieId) %>% left_join(reg_movie_avgs) %>% left_join(movie_titles,by='movieId') %>%
  arrange(desc(b_i)) %>% select(title, b_i, n) %>% slice(1:10) %>% knitr::kable()


rmse_results <- bind_rows(rmse_results, data.frame(method= "Regulized Movie Effect Model", RMSE= finalRMSE_movie))




# Regulization with movie + user effect


# preparing the training data and test data the will use cross validation to determine the best lambda.

set.seed(1, sample.kind="Rounding")

test_index_Reg <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_Reg <- edx[-test_index_Reg,]
temp_Reg <- edx[test_index_Reg,]


test_Reg <- temp_Reg %>% 
  semi_join(edx_Reg, by = "movieId") %>%
  semi_join(edx_Reg, by = "userId")



removed_Reg <- anti_join(temp_Reg, test_Reg)
edx_reg <- rbind(edx_Reg, removed_Reg)

rm(test_index_Reg, temp_Reg, removed_Reg)

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

lambdas <- seq(0, 5, 0.25)

rmses <- sapply(lambdas,function(l){
  
  
  mu <- mean(edx_Reg$rating)
  
  
  reg_movie_avgs <- edx_Reg %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  
  reg_user_avgs <- edx_Reg %>% 
    left_join(reg_movie_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  
  predicted_ratings <- 
    test_Reg %>% 
    left_join(reg_movie_avgs, by = "movieId") %>%
    left_join(reg_user_avgs, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, test_Reg$rating))
})

plot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]
paste('Optimal RMSE of',min(rmses),'is achieved with Lambda',lambda)

# Now we will use the lambda to calculate the RMSE using the Validation data

pred_y_lse <- sapply(lambda,function(l){
  
  
  mu <- mean(edx$rating)
  
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred #validation
  
  return(predicted_ratings)
  
})
finalRMSE = RMSE(pred_y_lse, validation$rating)

rmse_results <- bind_rows(rmse_results, data.frame(method= "Regulized Movie + User Effect Model", RMSE= finalRMSE))




rmse_results %>% knitr::kable()
