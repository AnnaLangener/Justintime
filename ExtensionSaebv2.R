######### Load Packages ############
####################################

library(randomForest)
library(R.matlab)
library(foreach)
library(doParallel)
library(dplyr)
library(lme4)
library(tseries)
library(caret)
library(pROC)
library(boot)
library(plotly)



#set.seed(1234)  # Setting the seed for random number generation
##################################################
########### Set Simualtion Parameters ############
##################################################
# Basic Parameters from Study
n_features <- 10 # FIXED IN SAEB
n_samples <- 90  # Timepoints/ RECORDS (fixed in saeb) Number of samples per subject
n_subjects <- 150 # SUBJECTS (VARIED 4 TO 32) # Needs to be an even number

# Generating Outcome
overall_prob_outcome <- 0.1  # Overall probability of 1 (for the entire dataset) (e.g., tracking depression 0.146)
sqrt(overall_prob_outcome*(1-overall_prob_outcome))
sd_outcome <- 0.25 # Controls variability BETWEEN different subject (e.g., tracking depression 0.21)
# within_variability_outcome has to be: sd_outcome < sqrt(overall_prob_outcome*(1-overall_prob_outcome))
# (variance < mean(1-mean)

time_effect = FALSE

# Generating Features
A <- 0.05 # Relationship between features and outcome
feature_std <- 0.1 # population level feature generating process
B <- 0.7  # Cross-subject variability ("random effect") (added per participants for all timepoints)
# Add varying B (B[i], C[i] should work)
C <- 0.1 # Within-subject variability (added within participant for each timepoint)

##################################################
################# Data Simulation ################
##################################################
source("Simulation_Functions.R")

features_sample <- create_data(n_features,n_samples,n_subjects,A,feature_std,B,C,overall_prob_outcome,sd_outcome,time_effect)
features_sample_Astd <- features_sample[[2]]
features_sample_centered <- features_sample[[3]]
features_sample <- features_sample[[1]]

#features_sample_rep <- create_data(n_features,n_samples,n_subjects = n_subjects_rep,n_test,n_train,A,feature_std,B,C,overall_prob_outcome,sd_outcome,time_effect)
#features_sample <- read.csv("/Users/f007qrc/Library/CloudStorage/GoogleDrive-anna.m.langener@dartmouth.edu/My Drive/Darmouth Drive/ML CrossValidation Project/feature_sample.csv")

sim <- run_simulation(features_sample,"row-wise",1, testsize = 0.3)
#write.csv(features_sample,"/Users/f007qrc/Library/CloudStorage/GoogleDrive-anna.m.langener@dartmouth.edu/My Drive/Darmouth Drive/ML CrossValidation Project/feature_sample.csv" )

sim <- run_simulation_centering(features_sample,"row-wise",1,testsize = 0.3)
sim <- run_simulation_slidingwindow(features_sample,1,windowsize = 14)

features_sample_centered %>% 
  ggplot(aes(x=V1, fill=as.factor(y))) +
  geom_histogram(alpha=0.6, position = 'identity') +
  scale_fill_manual(values=c("#FC9D9A","#83AF9B")) +
  theme_minimal() +
  labs(fill="")
mean(features_sample_centered$V1[features_sample_centered$y == 1])
mean(features_sample_centered$V1[features_sample_centered$y == 0])

features_sample %>% 
  ggplot(aes(x=V1, fill=as.factor(y))) +
  geom_histogram(alpha=0.6, position = 'identity') +
  scale_fill_manual(values=c("#FC9D9A","#83AF9B")) +
  theme_minimal() +
  labs(fill="")

mean(features_sample$V1[features_sample$y == 1])
mean(features_sample$V1[features_sample$y == 0])


####################################################
############ Test different parameters #############
####################################################

n_features <- 10 # FIXED IN SAEB
n_samples <- 90  # Timepoints/ RECORDS (fixed in saeb) Number of samples per subject
n_subjects <- 150 # SUBJECTS (VARIED 4 TO 32) # Needs to be an even number
time_effect = FALSE

A <- 0.9 # Relationship between features and outcome try 0.15 again
feature_std <- 0.1 # population level feature generating process

# Initialize parameters
overall_prob_outcome <- c(0.1,0.15,0.2,0.25,0.3,0.7,0.75,0.85,0.9)
sd_outcome <- seq(0.15, 0.29, by = 0.1)            
B <- seq(0.1, 1, by = 0.05)       
C <-  0.1   


# Create a grid of all parameter combinations
param_grid <- expand.grid(overall_prob_outcome = overall_prob_outcome,
                          sd_outcome = sd_outcome,
                          B = B)

# Set up parallel backend
n_cores <- parallel::detectCores() - 1  # Use all but two core
cl <- makeCluster(n_cores)
registerDoParallel(cl)
old <- Sys.time() # get start time

# Parallel loop with foreach
result <- foreach(i = 1:nrow(param_grid), .combine = rbind, .packages = c("dplyr", "lme4", "pROC", "randomForest"), .errorhandling = "remove") %dopar% {
  # Extract parameter combination for this iteration
  params <- param_grid[i, ]
  overall_prob <- params$overall_prob_outcome
  sd <- params$sd_outcome
  B_value <- params$B
  C_value <- C
  
  # Generate data for this iteration
  features_sample <- create_data(n_features, n_samples, n_subjects, A, feature_std, B_value, C_value, overall_prob, sd, time_effect)
  features_sample_Astd <- features_sample[[2]]
  features_sample <- features_sample[[1]]
  
  ### Descriptives ####
  test <- features_sample %>% group_by(subject) %>% summarise(prob = mean(y))
  
  # ICC outcome
  model <- lmer(y ~ 1 + (1 | subject), data = features_sample)
  var_random <- as.data.frame(VarCorr(model))$vcov[1]  # Random intercept variance
  var_residual <- attr(VarCorr(model), "sc")^2         # Residual variance
  icc = var_random / (var_random + var_residual)
  
  # ICC predictor
  model_pred <- lmer(V1 ~ 1 + ( 1  | subject), data = features_sample) # In our example we simulated the same relationships for all features
  var_random <- as.data.frame(VarCorr(model_pred))$vcov[1]  
  var_residual <- attr(VarCorr(model_pred), "sc")^2        
  icc_pred <- var_random / (var_random + var_residual)
  
  sim <- run_simulation(features_sample, "row-wise", 1, testsize = 0.3)
  sim3 <- run_simulation_centering(features_sample,"row-wise",1,testsize = 0.3)
  
  # Create results
  result_row <- data.frame(
    mean = mean(features_sample$y),
    sd = sd(test$prob),
    icc = icc,
    icc_pred = icc_pred,
    auc_value_base = sim$auc_value_base,
    auc = mean(sim$auc_value),
    auc_individual = sim$overall_summary$mean,
    total_n = sim$overall_summary$total_n,
    auc_c = mean(sim3$auc_value),
    auc_c_individual = sim3$overall_summary$mean,
    overall_prob_outcome = overall_prob,
    sd_outcome = sd,
    A = A,
    B = B_value,
    C = C_value,
    sd_intercept = as.data.frame(VarCorr(model_pred))$sdcor[1], # B recovered
    sd_residual = as.data.frame(VarCorr(model_pred))$sdcor[2] # C recovered
  )
  
  # Save the result to a file in real time
  write.table(result_row, file = "/Users/f007qrc/Library/CloudStorage/GoogleDrive-anna.m.langener@dartmouth.edu/My Drive/Darmouth Drive/ML CrossValidation Project/simulation_results_test_new.csv", append = TRUE, sep = ",", col.names = !file.exists("/Users/f007qrc/Library/CloudStorage/GoogleDrive-anna.m.langener@dartmouth.edu/My Drive/Darmouth Drive/ML CrossValidation Project/simulation_results_test_new.csv"), row.names = FALSE)
  
  # Return the result 
  result_row
}


# Stop the cluster
stopCluster(cl)

# print elapsed time
Sys.time() - old # calculate difference


####### Overall Performance ##########
library(patchwork)

p1 <- ggplot(result_0, aes(x=icc, y=auc_c_individual, color = icc_pred)) +
  geom_point(alpha=0.5, size = 3) +
  #scale_size(range = c(.001, 6), name="") +
  theme_minimal() +
  scale_colour_gradientn(colours = c("#2A363B","#83AF9B","#C8C8A9","#F9CDAD","#FC9D9A","#FE4365")) +
  ylab("AUC") +
  xlab("ICC outcome") +
  guides(col = guide_colourbar()) +
  ggtitle(paste("A = ", result_0$A)) +
  # ggtitle(paste("A = ", A, "AUC_c min =",round(min(result$auc_c),2), "AUC_c max = ", round(max(result$auc_c),2),"AUC min =",round(min(result$auc),2), "AUC max = ", round(max(result$auc),2))) +
  geom_hline(yintercept = 0.5) +
  ylim(range= c(0.5,0.9))

p3 <- ggplot(result_0.1, aes(x=icc, y=auc_c_individual, color = icc_pred)) +
  geom_point(alpha=0.5, size = 3) +
  #scale_size(range = c(.001, 6), name="") +
  theme_minimal() +
  scale_colour_gradientn(colours = c("#2A363B","#83AF9B","#C8C8A9","#F9CDAD","#FC9D9A","#FE4365")) +
  ylab("AUC") +
  xlab("ICC outcome") +
  guides(col = guide_colourbar()) +
  ggtitle(paste("A = ", result_0.1$A)) +
  # ggtitle(paste("A = ", A, "AUC_c min =",round(min(result$auc_c),2), "AUC_c max = ", round(max(result$auc_c),2),"AUC min =",round(min(result$auc),2), "AUC max = ", round(max(result$auc),2))) +
  geom_hline(yintercept = 0.5) +
  ylim(range= c(0.5,0.9))

combined_plot <- (p1 / p3) +
  plot_annotation(
    title = "Overall Performance (not centered)",
  ) 

########## Within-Person Perfromance ##########
p1 <-  ggplot(result, aes(x=icc, y=auc_c_individual-auc_individual, color = icc_pred)) +
  geom_point(alpha=0.5, size = 2) +
  #scale_size(range = c(.001, 6), name="") +
  theme_minimal() +
  scale_colour_gradientn(colours = c("#2A363B","#83AF9B","#C8C8A9","#F9CDAD","#FC9D9A","#FE4365")) +
  ylab("AUC centered - AUC") +
  xlab("ICC outcome") +
  guides(col = guide_colourbar()) +
  ggtitle(paste(
    "Within person results:",
    "A = ", A,
    "\nAUC centered mean = ", round(mean(result$auc_c_individual), 2),",",
    "AUC not centered mean = ", round(mean(result$auc_individual), 2)
  )) +
  geom_hline(yintercept = 0) 


p2 <- ggplot(result, aes(x=icc, y=auc_individual, color = icc_pred)) +
  geom_point(alpha=0.5, size = 2) +
  #scale_size(range = c(.001, 6), name="") +
  theme_minimal() +
  scale_colour_gradientn(colours = c("#2A363B","#83AF9B","#C8C8A9","#F9CDAD","#FC9D9A","#FE4365")) +
  ylab("AUC") +
  xlab("ICC outcome") +
  guides(col = guide_colourbar()) +
  geom_hline(yintercept = 0.5) +
  ylim(range= c(0.5,0.9))

p3 <- ggplot(result, aes(x=icc, y=auc_c_individual, color = icc_pred)) +
  geom_point(alpha=0.5, size = 2) +
  #scale_size(range = c(.001, 6), name="") +
  theme_minimal() +
  scale_colour_gradientn(colours = c("#2A363B","#83AF9B","#C8C8A9","#F9CDAD","#FC9D9A","#FE4365")) +
  ylab("AUC") +
  xlab("ICC outcome") +
  guides(col = guide_colourbar()) +
  geom_hline(yintercept = 0.5) +
  ylim(range= c(0.5,0.9))
  

combined_plot <- (p1 / p2/ p3) +
  plot_annotation(
    title = "Within-Person Performance",
  ) 



############ Relationship AUC & AUC individual ###########
p1 <- ggplot(result, aes(x=auc_c, y=auc_c_individual, color = icc_pred)) +
  geom_point(alpha=0.5, size = 3) +
  #scale_size(range = c(.001, 6), name="") +
  theme_minimal() +
  scale_colour_gradientn(colours = c("#2A363B","#83AF9B","#C8C8A9","#F9CDAD","#FC9D9A","#FE4365")) +
  ylab("AUC Individual centered") +
  xlab("AUC centered") +
  guides(col = guide_colourbar(title = "ICC predictor")) +
  ggtitle("A = 0.05") +
 # ggtitle(paste("A = ", A, "AUC_c min =",round(min(result$auc_c),2), "AUC_c max = ", round(max(result$auc_c),2),"AUC min =",round(min(result$auc),2), "AUC max = ", round(max(result$auc),2))) +
  geom_hline(yintercept = 0.5)
  
p2 <- ggplot(result, aes(x=auc, y=auc_individual, color = icc_pred)) +
  geom_point(alpha=0.5, size = 3) +
  #scale_size(range = c(.001, 6), name="") +
  theme_minimal() +
  scale_colour_gradientn(colours = c("#2A363B","#83AF9B","#C8C8A9","#F9CDAD","#FC9D9A","#FE4365")) +
  ylab("AUC Individual") +
  xlab("AUC") +
  guides(col = guide_colourbar(title = "ICC predictor")) +
  ggtitle("A = 0.05") +
# ggtitle(paste("A = ", A, "AUC_c min =",round(min(result$auc_c),2), "AUC_c max = ", round(max(result$auc_c),2),"AUC min =",round(min(result$auc),2), "AUC max = ", round(max(result$auc),2))) +
  geom_hline(yintercept = 0.5) +
  ylim(range= c(0.5,0.9))



combined_plot <- (p2 / p1) +
  plot_annotation(
    title = "Relationship Overall AUC and within person AUC",
  ) 



plot_ly(result, 
        x = ~icc_pred, 
        y = ~icc, 
        z = ~auc, 
        type = 'scatter3d', 
        mode = 'markers',
        marker = list(size = 3),
        text = ~paste('ICC Pred:', icc_pred, '<br>ICC:', icc, '<br>AUC:', auc, '<br>SD:', sd),  # Custom hover text
        hoverinfo = 'text') 



######## Run Diagnostics ############
#####################################

### To be PROVIDED by the USER ###

#features_sample <- read data in
feature_names <- colnames(features_sample)[1:n_features] # name of columns that contain features
outcome_variable <- "y" # name of outcome variable
id_variable <- "subject" # name of id/ subject variable
time_variable <- "time" # name of time variable
# add feature_std and check again if this is really needed
# testsize can be added
# number of ML repitions can be added

### Basic Descriptives (extract prevalence and SD) ###
overall_prob_outcome = mean(features_sample$y)
grouped <- features_sample %>% group_by(subject) %>% summarise(prob = mean(y))
sd_outcome = sd(grouped$prob)

n_subjects  <- nrow(unique(features_sample[id_variable]))  
n_samples <- nrow(unique(features_sample[time_variable]))  

# - To do: add plot from below

### ICC Predictor (extract B, C) ###
feature_names <- colnames(features_sample)[1:n_features]
id_variable <- "subject"
time_variable <- "time"

predictors_des <- data.frame(variable = rep(NA,n_features), icc = rep(NA,n_features), B = rep(NA,n_features), C = rep(NA,n_features))

for(i in 1:n_features){
  predictors_des$variable[i] <- feature_names[i]
  
  model_pred <- lmer(as.formula(paste0(feature_names[i], "~ 1 + ( 1  |", id_variable,")")), data = features_sample) 
  var_random <- as.data.frame(VarCorr(model_pred))$vcov[1]  
  var_residual <- attr(VarCorr(model_pred), "sc")^2     
  
  predictors_des$icc[i] <- var_random / (var_random + var_residual)
  predictors_des$B[i] <- as.data.frame(VarCorr(model_pred))$sdcor[1]
  predictors_des$C[i] <- as.data.frame(VarCorr(model_pred))$sdcor[2]
}


### Run Simulation (to get "baseline comparisons) ###


A <- 0 # Relationship between features and outcome
feature_std <- 0.1 # CHECK AGAIN population level feature generating process
B <- predictors_des$B  # Cross-subject variability ("random effect") (added per participants for all timepoints)
# Add varying B (B[i], C[i] should work)
C <- predictors_des$C # Within-subject variability (added within participant for each timepoint)



features_sample <- create_data(n_features,n_samples,n_subjects,A,feature_std,B,C,overall_prob_outcome,sd_outcome,time_effect)
sim <- run_simulation(features_sample,"row-wise",1, testsize = 0.3)
sim <- run_simulation(features_sample,"subject-wise",1, testsize = 0.3)

#### ICC Outcome (maybe delete?)
model <- lmer(y ~ 1 + (1 | subject), data = features_sample)
var_random <- as.data.frame(VarCorr(model))$vcov[1]  # Random intercept variance
var_residual <- attr(VarCorr(model), "sc")^2         # Residual variance
icc <- var_random / (var_random + var_residual)


# person center predictor variables to get model to generalize
# To enter condition in which model is unable to do it (person centering) > information gets thrown away
# Based on within person process that generalzie 

# 1. Person centered
# 2. Custom loss (keras, R)


# run same plot with some relationship

# difference in AUC plot (AUC RF, AUC Baseline)

# Low ICC, check true signal to noise, check how to quantify, start with low ICC
# within-person: signal, between-person: noise

# add some individual loss function > and check how it performs
####################################################


#### Data Visualization Outome #####
stat <- features_sample %>%
  group_by(subject) %>%
  summarise(stationary_p = adf.test(y)$p.value)

# Merge the p-values with the original dataset for plotting
data <- features_sample %>%
  left_join(stat, by = "subject")

# Plot the data and add p-values as annotations (TODO remove NA pvalue)
p2 <- ggplot(data, aes(x = time, y = y)) +
  geom_point(color = "#83AF9B") +                       
  facet_wrap(~subject) +               
  labs(x = "Time", y = "Outcome") +
  theme_minimal() +
  guides(fill = "none") 

p2

#geom_text(data = stat,               # Add p-values
#                  aes(x = Inf, y = Inf, label = paste0("p = ", round(stationary_p, 3))),
#          inherit.aes = FALSE,       # Do not inherit x and y aesthetics
#          hjust = 1.1, vjust = 1.8) +


#########################################################
##########################################################
# Other code

# Code to extract individual AUC from sim study
test <- sim$ind[[1]] %>%
  group_by(subject) %>%
  filter(length(unique(true)) > 1) %>%
  filter(sum(true == 1) > 0, sum(true == 0) > 0) %>%
  summarise(
    auc_val = auc(roc(as.numeric(as.character(true)), as.numeric(as.character(pred)), quiet = TRUE))[1],
    .groups = "drop" # Ungroup after summarizing
  )




#model <- glmer(y ~ 0 + V1 + V2 + (V1 + V2| subject), data = features_sample,family =  binomial(link = "logit"))
#summary(model) 
#anova(model)
#coef(summary(model))

#model <- lm(y ~ 0 + v1, data = features_sample)
#coef(summary(model))



# Stationarity Binary
stat <- data %>%
  group_by(subject) %>%
  summarise(stationary_p = summary(lm(y ~ time))$coefficients[2, 4])

# test time effect (randomness needs to be deleted)
test = data %>% group_by(time) %>% summarise(n = sum(y))

p1 <- ggplot(test, aes(x = time, y = n)) +
  geom_line() +
  labs(x = "Time", y = "Count") +  # Add axis labels (optional, based on context)
  theme_minimal()

#hist(data$y)

################ Empirical Example ###############
exp <- read.csv('/Users/f007qrc/Documents/Unishare_Saved/Tracking Depression Study/ESM_data_rollingrmssd.csv')
binarize_rmssd <- function(df) {
  # Calculate mean and standard deviation for each participant within the normalized day range (-4 to 24)
  stats <- df %>%
    filter(normalized_day >= -4 & normalized_day <= 90) %>%
    group_by(uid) %>%
    summarise(
      std = sd(rolling_rmssd, na.rm = TRUE),
      mean = mean(rolling_rmssd, na.rm = TRUE),
    )
  
  df <- df %>%
    left_join(stats, by = "uid")
  
  df["treshhold"] = df["mean"] + df["std"]
  df["rmmssd_binary"] = ifelse(df["rolling_rmssd"] > df["mean"] + df["std"], 1,0)
  
  return(df)
}


#exp <- binarize_rmssd(exp)
rmssd = as.numeric(unlist(na.omit(exp["rolling_rmssd"])))

exp$rmssd_binary = ifelse(exp$rolling_rmssd > mean(rmssd)+  sd(rmssd), 1,0)

mean(exp$rmssd_binary, na.rm = TRUE)
test <- exp %>% group_by(uid) %>% summarise(prob = mean(rmssd_binary, na.rm = TRUE))
sd(test$prob, na.rm = TRUE)

# Intraclass correlation
model <- lmer(rmssd_binary ~ 1 + (1 | uid), data = exp)
var_random <- as.data.frame(VarCorr(model))$vcov[1]  # Random intercept variance
var_residual <- attr(VarCorr(model), "sc")^2         # Residual variance
icc <- var_random / (var_random + var_residual)

icc

