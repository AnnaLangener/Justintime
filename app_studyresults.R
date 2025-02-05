library(shiny)
library(bslib)

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

# Source external functions
source("Simulation_Functions.R")
source("Simulation_UploadData.R")
library(shiny)
library(bslib)
library(ggplot2)
library(dplyr)
library(lme4)
library(tseries)
library(shinycssloaders)
library(DT)


# Define UI for application
ui <- fluidPage(
  theme = bs_theme(version = 5, bootswatch = "journal"),
  
  titlePanel("Just in time?"),
  
  tabsetPanel(
    # Original Simulation Tab
    tabPanel("Run your own simulation",
             sidebarLayout(
               sidebarPanel(
                 h5("Basic Parameters"),
                 numericInput("n_features_sim", "Number of Features:", value = 10, min = 1, step = 1),
                 numericInput("n_samples_sim", "Number of Samples (timepoints per subject):", value = 90, min = 1, step = 1),
                 numericInput("n_subjects", "Number of Subjects:", value = 150, min = 2, step = 2),
                 
                 h5("Outcome Parameters"),
                 numericInput("overall_prob_outcome", "Overall Probability of Outcome:", value = 0.1, min = 0, max = 1, step = 0.01),
                 numericInput("sd_outcome", "Standard Deviation of Outcome (Between Subjects):", value = 0.1, step = 0.01),
                 
                 h5("Feature Generation"),
                 numericInput("A", "Relationship Between Features and Outcome (A):", value = 0.05, step = 0.01),
                 numericInput("feature_std", "Population-level Feature Variability (Feature Std):", value = 0.1, step = 0.01),
                 numericInput("B", "Cross-Subject Variability (B):", value = 0.7, step = 0.01),
                 numericInput("C", "Within-Subject Variability (C):", value = 0.1, step = 0.01),
                 
                 h5("Simulation Parameters"),
                 selectInput("split_method", "Data Split Method:", choices = c("row-wise", "subject-wise", "moving-window"), selected = "row-wise"),
                 numericInput("test_size", "Test Set Size (Proportion):", value = 0.3, min = 0.1, max = 0.9, step = 0.1),
                 numericInput("windowsize_sim", "Window size (moving-window only, timepoints):", value = 14, min = 1, max = 100, step = 1),
  
                 
                 numericInput("replications", "Number of Replications:", value = 1, min = 1, step = 1),
                 
                 actionButton("generate_data", "Generate Data"),
                 actionButton("run_sim", "Run Simulation")
               ),
               
               mainPanel(
                 
                 accordion( 
                   accordion_panel( 
                     title = "Visualization", 
                     plotOutput("simulation_plot",width = "600px", height = "650px")
                     
                   ),
                   accordion_panel( 
                     title = "Example Features", 
                     tableOutput("generated_features")
                     
                   ),
                   accordion_panel( 
                     title = "Simulation Results", 
                     tableOutput("simulation_results"),
                     tableOutput("simulation_results_centered"),
                
                   )  
                 ),
             
            
               )
             )
    ),

    # Data Upload Tab
    tabPanel("Upload your data",
             sidebarLayout(
               sidebarPanel(
                 h3("Upload Dataset"),
                 fileInput("data_file", "Choose CSV File", accept = c(".csv")),
                 h5("Select Variables"),
                 selectInput("outcome_variable", choices = character(0), label = "Outcome Variable Name:"),
                 selectInput("id_variable", choices = character(0), label = "ID/Subject Variable Name:"),
                 selectInput("time_variable", choices = character(0), label =  "Time Variable Name:"),
                 selectInput("n_features_upload", choices = character(0), label =  "Predictors:",multiple = TRUE),
                 h5("Choose Cross-Validation Parameters"),
                 selectInput("split_method_own", "Data Split Method:", choices = c("row-wise", "subject-wise", "moving-window"), selected = "row-wise"),
                 numericInput("test_size_upload", "Test Set Size (Proportion):", value = 0.3, min = 0.1, max = 0.9, step = 0.1),
                 numericInput("windowsize", "Windowsize (moving-window only, timepoints):", value = 14, min = 1, max = 100, step = 1),
                 numericInput("reps_upload", "Number of Replications:", value = 1, min = 1, step = 1),
                 actionButton("analyze_data", "Analyze Data"),
                 actionButton("run_baseline", "Baseline Comparison"),
                 actionButton("run_sim_upload", "Run Model")
               ),
               
               mainPanel(
                 accordion( 
                   accordion_panel( 
                     title = "1. Define Use Case, Employment Strategy, and Cross-Validation Approach", 
                     textOutput("output_text_1"),
                     tags$br(),
                     uiOutput("cv_image")  
                   ),  
                   
                   accordion_panel( 
                     title = "2. Evaluate Feasibility in (Clinical) Practice", 
                     h5("General Information and Descriptives"),
                     textOutput("output_text_2"),
                     tags$br(),
                     textOutput("methods_text"),
                     tags$br(),
                     tableOutput("descriptives"),
                    
                     
                     h5("Visualization and Variability"),
                     textOutput("dynamic_title_text"),
                    tags$br(),
                     plotOutput("upload_plot",width = "600px", height = "650px"),
                   )
                  ,
                  accordion_panel( 
                    title = "3. Select Meaningful Baselines for Performance Comparison", 
                    h5("Baseline Model 1: Random-Intercept Only"),
                    withSpinner(textOutput("simulation_results_text1")),
                    tags$br(),
                    tableOutput("simulation_results_upload1"),
                    h5("Baseline Model 2: Shuffled Outcome Variable"),
                    h6("Variance Explained by Between Person Differences"),
                    "In addition to a random intercept model, it’s useful to include a baseline model with predictors that have no relationship to the outcome.",
                    tags$br(),
                    "This helps determine if the predictors are merely distinguishing between individuals or if they capture a relationship to the outcome. If a significant portion of the variance in both the outcome and predictors is explained by between-person differences, the model may merely differentiate between individuals. Below, you can see the percentage of variance in the outcome (ICC outcome) and predictors explained by between-person differences.",
                    textOutput("methods_text_icc"),
                    tags$br(),
                    div(DT::dataTableOutput("icc_table"), style = "font-size: 75%; width: 75%"),
                    
                    tags$br(),
                    h6("Model Performance"),
                    "To assess the impact of the variance explained by the predictor variables, it’s helpful to include, in addition to a random intercept model, a baseline model with predictors that have no true relationship to the outcome. We achieve this by shuffling the outcome variable within each person and also shuffling the subject identifiers. This preserves the outcome distribution but removes any true relationship with the predictors. We then run a random forest model. Please note that both baseline models should perform worse than your actual model to ensure that the predictors are truly contributing to the outcome.",
                    tags$br(),
                    withSpinner(textOutput("simulation_results_text1_baseline")),
                    tags$br(),
                    tableOutput("simulation_results_upload1_baseline"),
                  ),
                  accordion_panel( 
                    title = "4. Include Within-Person Evaluation for Tracking and Consider Centering", 
                    h5("General Information"),
                    "When tracking individuals over time and predicting within-person differences, model performance should be evaluated both across the full dataset and within each individual. We’ve shown that centering predictors can improve accuracy and reduce bias, so we present results for both centered and uncentered predictors. We ran a random forest model",
                    tags$br(),
                    h5("Simulation Results (not centered)"),
                    withSpinner(textOutput("model_results_text")),
                    tags$br(),
                    tableOutput("simulation_results_upload"),
                    h5("Simulation Results (centered)"),    
                    withSpinner(textOutput("model_results_text_centered")),
                    tags$br(),
                    tableOutput("simulation_results_upload_centered"),
                  ),
                   ),  
          
                 
               )
             )
    ),
    
    tabPanel("Explore study results",
             sidebarLayout(
               sidebarPanel(
                 h4("Filter data?"),
                 checkboxInput("filter_values", "Filter Data"),
                 h5("Paramters:"),
                 selectInput("A_viz", "A:", c(0,0.05,0.2,0.8), 0.05),
                 sliderInput("mean_viz", "Overall Probability Outcome:", min = 0.1, max = 0.9, value = c(0.1, 0.9)),
                 sliderInput("sd_viz", "Standard Deviation of Outcome:", min = 0.05, max = 0.25, value = c(0.05, 0.25)),
                 sliderInput("icc_viz", "ICC:", min = 0, max = 0.85, value = c(0, 0.85)),
                 sliderInput("icc_pred_viz", "Predictors ICC:", min = 0, max = 1, value = c(0, 1)),
                 sliderInput("auc_value_base_viz", "AUC Value Base:", min = 0.4, max = 1, value = c(0.4, 1)),
                 sliderInput("auc_value_viz", "AUC Value:", min = 0.4, max = 1, value = c(0.4, 1)),
                 sliderInput("auc_individual_mean_viz", "AUC Within-Person:", min = 0.4, max = 1, value = c(0.4, 1)),
                 sliderInput("auc_c_mean_viz", "AUC Centered:", min = 0.4, max = 1, value = c(0.4, 1)),
                 sliderInput("auc_c_individual_mean_viz", "AUC Centered Within-Person:",min = 0, max = 1, value = c(0, 1)),
                 sliderInput("total_n_viz", "Total N:", min = 0, max = 150, value = c(0, 150)),
                 sliderInput("sd_intercept_viz", "SD Intercept:", min = 0, max = 1.2, value = c(0, 1.2)), # Just add 4 different bins
                 sliderInput("sd_residual_viz", "SD Residual:", min = 0, max = 0.5, value = c(0.1, 0.5)) # Just add 4 different bins
                
               ),
               mainPanel(
                 fluidRow(
                   # Create columns to position select inputs next to each other
                   column(4,  # 4/12 width for the first select input
                          selectInput("x_var", "X-Axis Variable:", 
                                      choices = c("icc", "auc", "auc_c", "sd_residual"), 
                                      selected = "icc")
                   ),
                   column(4,  # 4/12 width for the second select input
                          selectInput("y_var", "Y-Axis Variable:", 
                                      choices = c("auc", "auc_c", "auc_individual", "auc_c_individual"), 
                                      selected = "auc")
                   ),
                   column(4,  # 4/12 width for the third select input
                          selectInput("color_var", "Color Variable:", 
                                      choices = c("sd_residual", "icc_pred", "sd_intercept", "sd_outcome"), 
                                      selected = "icc_pred")
                   )
                 ),
                 
                 h4("Visualization"),
                 
                 plotOutput("plot1"),
                 plotOutput("plot3")
                 
               )
             )
    ),selected="Upload your data"
    
    
  )
)

server <- function(input, output, session) {
  # Original Simulation Logic
  generated_features <- eventReactive(input$generate_data, {
    # Simulate some data (you'll need to implement create_data)
    create_data(
      n_features = input$n_features_sim,
      n_samples = input$n_samples_sim,
      n_subjects = input$n_subjects,
      A = input$A,
      feature_std = input$feature_std,
      B = input$B,
      C = input$C,
      overall_prob_outcome = input$overall_prob_outcome,
      sd_outcome = input$sd_outcome,
      time_effect = input$time_effect
    )
  })
  
  output$generated_features <- renderTable({
    req(generated_features())
    head(generated_features())
  })
  
  output$simulation_plot <- renderPlot({
    req(generated_features())
    features_sample <- generated_features()
    
    grouped <- features_sample %>% group_by(subject) %>% summarise(prob = mean(y))
    sd_outcome <- sd(grouped$prob)
    mean_y = mean(features_sample$y)
    
    ggplot(features_sample, aes(x = time, y = y)) +
      geom_point(color = "#83AF9B") +
      facet_wrap(~subject) +
      labs(x = "Time", y = "Outcome") +
      ggtitle(paste("Mean prevalence:",round(mean_y,2), "sd:",round(sd_outcome,2))) +
      theme_minimal() +
      xlab("") +
      theme(axis.text=element_text(size=0),
            axis.title=element_text(size=14,face="bold"),
            title =element_text(size=14,face="bold") ) 
    
  },height = 650, width = 600)
  
  simulation_results <- eventReactive(input$run_sim, {
    req(generated_features())
    
    if(input$split_method == "row-wise" |input$split_method == "subject-wise" ){
      return(run_simulation(
        features_sample = generated_features(),
        cv = input$split_method,
        n_bootstrap = input$replications,
        testsize = input$test_size,
        n_features = input$n_features_sim
      ))
    }
   
    else if(input$split_method == "moving-window" ){
      return(run_simulation_slidingwindow(
        features_sample = generated_features(),
        n_bootstrap = input$replications,
        windowsize = input$windowsize_sim,
        n_features = input$n_features_sim
      ))
    } else {
      return(data.frame()) 
    }
  })
  
  simulation_results_centered <- eventReactive(input$run_sim, {
    req(generated_features())
    if(input$split_method == "row-wise"){
      return(run_simulation_centering(
          features_sample = generated_features(),
          cv = input$split_method,
          n_bootstrap = input$replications,
          testsize = input$test_size
        ))
    }
    
    else if(input$split_method == "moving-window" ){
      return(run_simulation_slidingwindow_centering(
        features_sample = generated_features(),
        n_bootstrap = input$replications,
        windowsize = input$windowsize_sim,
        n_features = input$n_features_sim
      ))
    } else {
      return(data.frame())
    }
  })
  
  
  output$simulation_results <- renderTable({
    req(simulation_results())  # Ensure reactive value exists
    result <- simulation_results()
    
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
    return(result)
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  
  output$simulation_results_centered <- renderTable({
    req(simulation_results_centered())  # Ensure reactive value exists
    result_cen <- simulation_results_centered()
    
    print(result_cen)
    print(is.null(result_cen)) 
    
    if (is.null(result_cen) || nrow(result_cen) == 0) {
      return(data.frame(Message = "No centered results available"))  # Display a message instead of an empty table
    }
    
    return(result_cen)
  }, include.rownames = TRUE, include.colnames = FALSE)
  
  #########################################################
  ######################## Tab 2 ##########################
  #########################################################
  
  ######### First Section ##########
  
  output$output_text_1 <- renderText({
    switch(input$split_method_own,
           "row-wise" = "Choosing a cross-validation strategy that aligns with your specific use case and study objectives is important for ensuring meaningful results. You selected a 'row-wise' cross-validation split. This means that participants are present in both the training and testing sets. 
         This split is useful when the goal is to track individuals over time, as it allows for capturing within-person variability and between-person variability. It is particularly relevant when the model will be applied to the same group of individuals.",
           
           "subject-wise" = "Choosing a cross-validation strategy that aligns with your specific use case and study objectives is important for ensuring meaningful results. You selected a 'subject-wise' cross-validation split. In this approach, the data is split such that each subject is either in the training or testing set, but not both. 
         This split is useful when the aim is to assess the model’s ability to generalize to new, unseen subjects. It ensures that the training and testing sets are independent, making it a better option for evaluating performance in cases where the model needs to generalize across different individuals or populations.",
           
           "moving-window" = "Choosing a cross-validation strategy that aligns with your specific use case and study objectives is important for ensuring meaningful results. You selected a 'moving-window' cross-validation split. This method involves using a rolling window of data for training and testing. 
         In this split, a fixed-size training set is used, and as the model moves forward in time, the training set is updated, and a new testing set is created. This approach is useful for time-series data or when you want to simulate the model’s performance in a dynamic setting, where the training data gradually changes over time as new information becomes available.
           Importantly, this split assumes that the model will be updated over time.")
  })
  
  output$output_text_2 <- renderText({
    switch(input$split_method_own,
           "row-wise" = "The next step is to assess whether the chosen use case scenario and cross-validation strategy are appropriate for your objectives. 
         You have selected a 'row-wise' cross-validation strategy, which is particularly useful if your goal is to capture within-person variability. 
         Below, we have visualized the data and calculated some basic descriptives. Consider whether the outcome shows sufficient variability for your intended use case. If the variability is low, it may be worth shifting to a task that focuses on distinguishing between individuals.",
           
           "subject-wise" = "The next step is to evaluate whether the chosen use case scenario and cross-validation strategy are suitable for your objectives. 
         You have chosen a 'subject-wise' cross-validation strategy, which is particularly useful if your goal is to differentiate between individuals. 
         TO DO: ADD MORE INFORMATION ABOUT THIS STRATEGY.",
           
           "moving-window" = "The next step is to assess whether the chosen use case scenario and cross-validation strategy are appropriate for your objectives. 
         You have selected a 'moving-window' cross-validation strategy, which is particularly useful for capturing within-person variability over time. 
         Below, we have visualized the data and calculated some basic descriptives. Consider whether the outcome has sufficient variability for your intended use case. If variability is low, consider shifting to a task focused on distinguishing between individuals.")
  })
  
  output$cv_image <- renderUI({
    req(input$split_method_own)  # Ensure cv_choice is selected before rendering
    
    # Define image path based on input
    img_src <- switch(input$split_method_own,
                      "moving-window" = "moving-window.png",
                      "row-wise" = "Record-wise.png",
                      "subject-wise" = "Subject-wise.png")
    tags$img(src = img_src, width = "40%", height = "40%")
  })
  
  ############ Data Upload Logic ##############
  # 1. Create a reactive expression that reads in the CSV file once.
  uploaded_data <- reactive({
    req(input$data_file)
    read.csv(input$data_file$datapath, stringsAsFactors = FALSE)
  })
  
  # 2. Update all select inputs based on the new dataset.
  observeEvent(uploaded_data(), {
    df <- uploaded_data()
    cols <- colnames(df)
    
    # Update each select input with the column names.
    updateSelectInput(session, inputId = "outcome_variable", choices = cols, selected = cols[13])
    updateSelectInput(session, inputId = "time_variable", choices = cols, selected = cols[12])
    updateSelectInput(session, inputId = "id_variable", choices = cols, selected = cols[11])
    
    updateSelectInput(session, inputId = "n_features_upload", choices = cols)
  })
  
  # # 3. Process the uploaded data when the "analyze_data" button is clicked.
  analyzed_data <- eventReactive(input$analyze_data, {
    req(uploaded_data())
    data <- uploaded_data()

    data  # Return the (possibly shuffled) dataset.


  })
  
  # 4. Render Descriptive Statistics.
  # Reactive model to be used in both the table and text
  model_data <- reactive({
    req(analyzed_data())
    data <- na.omit(analyzed_data())
    
    overall_prob_outcome <- mean(data[[input$outcome_variable]], na.rm = TRUE)
    
    grouped <- data %>% 
      group_by_at(input$id_variable) %>% 
      summarise(prob = mean(!!rlang::sym(input$outcome_variable)), .groups = "drop")
    
    sd_outcome <- sd(grouped$prob, na.rm = TRUE)
    
    model <- lmer(as.formula(paste0(input$outcome_variable, " ~ 1 + (1|", input$id_variable, ")")), data = data)
    var_random   <- as.data.frame(VarCorr(model))$vcov[1]
    var_residual <- attr(VarCorr(model), "sc")^2
    icc_data <- var_random / (var_random + var_residual)
    
    list(
      overall_prob_outcome = overall_prob_outcome,
      sd_outcome = sd_outcome,
      num_subjects = length(unique(data[[input$id_variable]])),
      num_samples = length(unique(data[[input$time_variable]])),
      icc_data = icc_data
    )
  })
  
  # Render the table
  output$descriptives <- renderTable({
    model_result <- model_data()
    
    data.frame(
      "Overall Outcome Probability" = round(model_result$overall_prob_outcome, 3),
      "Outcome SD"                  = round(model_result$sd_outcome, 3),
      "Number of Subjects"          = model_result$num_subjects,
      "Number of Samples per Subject" = model_result$num_samples
    )
  })
  
  # Render the descriptive text
  output$methods_text <- renderText({
    model_result <- model_data()
    
    paste0(
      "A total of ", model_result$num_subjects, " unique subjects are included in the analysis. ",
      "Each subject has a maximum of ", model_result$num_samples, " repeated measurements. ",
      "The mean outcome probability is ", round(model_result$overall_prob_outcome, 3), 
      " (SD = ", round(model_result$sd_outcome, 3), "). ",
      "The percentage of variation in the outcome that is explained by differences between people is ", 
      round(model_result$icc_data, 3), "."
    )
  })
  
  output$methods_text_icc <- renderText({
    model_result <- model_data()
    
    paste0(
      "The percentage of variation in the outcome that is explained by differences between people is ", 
      round(model_result$icc_data, 3), "."
    )
  })
  

  
  
  # 5. Render the ICC Table.
  output$icc_table <- renderDT({
    req(analyzed_data())
    data <- na.omit(analyzed_data())
    
    # Select features based on the input.
    feature_names <- colnames(data)[colnames(data) %in% input$n_features_upload]
    
    icc_data <- data.frame(variable = feature_names, icc = NA)
    
    for (i in seq_along(feature_names)) {
      model <- lmer(as.formula(paste0(feature_names[i], " ~ 1 + (1|", input$id_variable, ")")), data = data)
      var_random   <- as.data.frame(VarCorr(model))$vcov[1]
      var_residual <- attr(VarCorr(model), "sc")^2
      icc_data$icc[i] <- var_random / (var_random + var_residual)
     # icc_data$B[i]   <- as.data.frame(VarCorr(model))$sdcor[1]
      #icc_data$C[i]   <- as.data.frame(VarCorr(model))$sdcor[2]
    }
    
    icc_data
  },options = list(pageLength = 5))
  
  # 8. Render a Plot of the Uploaded Data.
  output$upload_plot <- renderPlot({
    req(analyzed_data())
    data <- analyzed_data()
    
    # Rename columns for easier access
    colnames(data)[colnames(data) == input$id_variable]    <- "subject"
    colnames(data)[colnames(data) == input$time_variable]  <- "time"
    colnames(data)[colnames(data) == input$outcome_variable] <- "y"
    data <- data[!is.na(data$y),]
    
    subject_stats <- data %>%
      group_by(subject) %>%
      summarize(mean_y = mean(y, na.rm = TRUE),
                has_y1 = any(y == 1)) %>%  # Identify if subject has any y = 1
      arrange(mean_y)  
    
    subject_stats$subject <- factor(subject_stats$subject, levels = subject_stats$subject)
    
    num_subjects_with_y1 <- sum(subject_stats$has_y1)
    
    data <- data %>%
      mutate(subject = factor(subject, levels = subject_stats$subject))
    
    data <- left_join(data, subject_stats, by = "subject")
    
    # Create the plot with dynamic title
    ggplot(data, aes(x = time, y = y)) +
      # Add background shading for subjects without y = 1
      geom_rect(
        data = data %>% filter(!has_y1),
        aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf),
        fill = "#E5E4E2", alpha = 0.05, inherit.aes = T
      ) +
      geom_point(color = "#FC9D9A", size = 0.5) +                       
      facet_wrap(~subject) +               
      labs(
        x = "Time", 
        y = "Outcome", 
        title = "Within-person variability over time."
      ) +
      scale_y_continuous(breaks = c(0, 1)) +  # Ensure only 0 and 1 appear on the y-axis
      theme_minimal() +
      theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
      guides(fill = "none")
  }, height = 650, width = 600)
  
  # Render the dynamic title as extra text
  # Render the dynamic title as extra text
  output$dynamic_title_text <- renderText({
    req(analyzed_data())
    data <- analyzed_data()
    
    # Rename columns for easier access
    colnames(data)[colnames(data) == input$id_variable]    <- "subject"
    colnames(data)[colnames(data) == input$time_variable]  <- "time"
    colnames(data)[colnames(data) == input$outcome_variable] <- "y"
    data <- data[!is.na(data$y),]
    
    # Summarize data to find variability per subject
    subject_stats <- data %>%
      group_by(subject) %>%
      summarize(mean_y = mean(y, na.rm = TRUE),
                has_y1 = any(y == 1)) %>%  # Identify if subject has any y = 1
      arrange(mean_y)
    
    subject_stats$subject <- factor(subject_stats$subject, levels = subject_stats$subject)
    
    # Number of subjects who have at least one y = 1
    num_subjects_with_y1 <- sum(subject_stats$has_y1)
    total_subjects <- nrow(subject_stats)
    
    # Calculate the percentage of subjects with variability
    variability_percentage <- 100 * num_subjects_with_y1 / total_subjects
    
    # Base dynamic title text about variability
    base_text <- paste0(
      "The plot below shows within-person variability over time. In your data, ",
      num_subjects_with_y1, " participants (", round(variability_percentage, 1), "%) show variability in the outcome variable."
    )
    
    # Determine the variability level text
    if (variability_percentage > 99) {
      variability_text <- "This means that all subjects experience variability in the outcome. This suggests that your outcome likely has sufficient variability for modeling."
    } else if (variability_percentage > 50) {
      variability_text <- "Thus, many subjects experience some variability in the outcome. Consider whether this level of variability aligns with your study's needs."
    } else {
      variability_text <- "Hence, most subjects have no variability in the outcome. Consider whether this level of variability aligns with your study's needs."
    }
    
    # Add dynamic text depending on cross-validation strategy
    cv_strategy_text <- switch(input$split_method_own,
                               "row-wise" = "As you have chosen a 'row-wise' cross-validation strategy, it is important to capture within-person variability. This means that there must be sufficient variability in the outcome.",
                               "subject-wise" = "You chose a 'subject-wise' cross-validation strategy, which is ideal for differentiating between subjects. The level of variability may influence how well this strategy generalizes to new subjects but is generally less important.",
                               "moving-window" = "With the 'moving-window' cross-validation strategy, it is important to capture within-person variability. This means that there must be sufficient variability in the outcome. Additionally, time-series aspects of the data are also important. The variability over time will affect the performance of this strategy in capturing temporal trends.",
                               "Unknown strategy selected.")  # Default case if no valid strategy is selected
    
    # Combine all the text into the final dynamic title text
    dynamic_title_text <- paste(base_text, variability_text, cv_strategy_text)
    
    dynamic_title_text
  })
  
  
  
  ################## Run actual model centered
  simulation_results_upload <- eventReactive(input$run_sim_upload, {
    req(analyzed_data())
    data <- na.omit(analyzed_data())
    
    # Rename columns as needed.
    colnames(data)[colnames(data) == input$id_variable]    <- "subject"
    colnames(data)[colnames(data) == input$time_variable]  <- "time"
    colnames(data)[colnames(data) == input$outcome_variable] <- "y"
    
    # Convert subject identifiers to numeric indices if needed.
    data$subject <- sapply(data$subject, match, unique(unlist(data$subject)))
    
    
    if(input$split_method_own == "row-wise" |input$split_method_own == "subject-wise" ){
      return(run_simulation_own(
        features_sample = data,
        cv              = input$split_method_own,
        n_bootstrap     = input$reps_upload,
        testsize        = input$test_size_upload,
        n_features      = input$n_features_upload
      ))
    }
    
   else if(input$split_method_own == "moving-window" ){
      return(run_simulation_slidingwindow_own(
        features_sample = data,
        n_bootstrap     = input$reps_upload,
        windowsize      = input$windowsize,
        n_features      = input$n_features_upload
      ))
    } else {
      return(data.frame())
    }
  })
  
  output$simulation_results_upload <- renderTable({
    req(simulation_results_upload())  # Ensure reactive value exists
    result <- simulation_results_upload()
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
    return(result)
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  
  output$simulation_results_upload1 <- renderTable({
    req(simulation_results_upload())  # Ensure reactive value exists
    result <- simulation_results_upload()
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
    return(result[c("AUC random intercept only:","Accuracy random intercept only:"),])
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  output$simulation_results_text1 <- renderText({
    req(simulation_results_upload())  # Ensure reactive value exists
    result <- simulation_results_upload()
    
    # Round the AUC and Accuracy values
    auc_value <- round(result[c("AUC random intercept only:"),], 2)
    accuracy_value <- round(result[c("Accuracy random intercept only:"),], 2)
    
    if (input$split_method == "row-wise") {
      if (auc_value > 0.8) {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        Thus, the baseline model already demonstrates strong performance. 
        Given this, it may be necessary to reconsider whether the selected use-case scenario and cross-validation strategy are truly meaningful in practice. 
        The machine learning model must exceed these values to be considered clinically relevant.")
        )
      } else {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        Therefore, the machine learning model needs to exceed these values to be clinically relevant.")
        )
      }
      
    } else if (input$split_method == "subject-wise") {
      return(paste0("TO DO: ADD BASELINE?"))
      
    } else if (input$split_method == "moving-window") {
      if (auc_value > 0.8) {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        Thus, the baseline model already demonstrates strong performance. 
        Given this, it may be necessary to reconsider whether the selected use-case scenario and cross-validation strategy are truly meaningful in practice. 
        The machine learning model must exceed these values to be considered clinically relevant. TO DO: ADD WITHIN-PERSON")
        )
      } else {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        Therefore, the machine learning model needs to exceed these values to be clinically relevant. TO DO: ADD WITHIN-PERSON")
        )
      }
    }
  })
  
  
  
  
  simulation_results_upload <- eventReactive(input$run_sim_upload, {
    req(analyzed_data())
    data <- na.omit(analyzed_data())
    
    # Rename columns as needed.
    colnames(data)[colnames(data) == input$id_variable]    <- "subject"
    colnames(data)[colnames(data) == input$time_variable]  <- "time"
    colnames(data)[colnames(data) == input$outcome_variable] <- "y"
    
    # Convert subject identifiers to numeric indices if needed.
    data$subject <- sapply(data$subject, match, unique(unlist(data$subject)))
    
    
    if(input$split_method_own == "row-wise" |input$split_method_own == "subject-wise" ){
      return(run_simulation_own(
        features_sample = data,
        cv              = input$split_method_own,
        n_bootstrap     = input$reps_upload,
        testsize        = input$test_size_upload,
        n_features      = input$n_features_upload
      ))
    }
    
   else if(input$split_method_own == "moving-window" ){
      return(run_simulation_slidingwindow_own(
        features_sample = data,
        n_bootstrap     = input$reps_upload,
        windowsize      = input$windowsize,
        n_features      = input$n_features_upload
      ))
    } else {
      return(data.frame())
    }
  })
  
  output$simulation_results_upload <- renderTable({
    req(simulation_results_upload())  # Ensure reactive value exists
    result <- simulation_results_upload()
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
    return(result)
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  
  output$simulation_results_upload1 <- renderTable({
    req(simulation_results_upload())  # Ensure reactive value exists
    result <- simulation_results_upload()
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
    return(result[c("AUC random intercept only:","Accuracy random intercept only:"),])
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  output$simulation_results_text1 <- renderText({
    req(simulation_results_upload())  # Ensure reactive value exists
    result <- simulation_results_upload()
    
    # Round the AUC and Accuracy values
    auc_value <- round(result[c("AUC random intercept only:"),], 2)
    accuracy_value <- round(result[c("Accuracy random intercept only:"),], 2)
    
    if (input$split_method == "row-wise") {
      if (auc_value > 0.8) {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        Thus, the baseline model already demonstrates strong performance. 
        Given this, it may be necessary to reconsider whether the selected use-case scenario and cross-validation strategy are truly meaningful in practice. 
        The machine learning model must exceed these values to be considered clinically relevant.")
        )
      } else {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        Therefore, the machine learning model needs to exceed these values to be clinically relevant.")
        )
      }
      
    } else if (input$split_method == "subject-wise") {
      return(paste0("TO DO: ADD BASELINE?"))
      
    } else if (input$split_method == "moving-window") {
      if (auc_value > 0.8) {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        Thus, the baseline model already demonstrates strong performance. 
        Given this, it may be necessary to reconsider whether the selected use-case scenario and cross-validation strategy are truly meaningful in practice. 
        The machine learning model must exceed these values to be considered clinically relevant. TO DO: ADD WITHIN-PERSON")
        )
      } else {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        Therefore, the machine learning model needs to exceed these values to be clinically relevant. TO DO: ADD WITHIN-PERSON")
        )
      }
    }
  })
  
  #### Baseline 2 (SHUFFLED!!!!)
  simulation_baseline_upload <- eventReactive(input$run_baseline, {
    req(analyzed_data())
    data <- na.omit(analyzed_data())
    data <- shuffle_data(data, subject_var = input$id_variable, outcome_var = input$outcome_variable)
    
    # Rename columns as needed.
    colnames(data)[colnames(data) == input$id_variable]    <- "subject"
    colnames(data)[colnames(data) == input$time_variable]  <- "time"
    colnames(data)[colnames(data) == input$outcome_variable] <- "y"
    
    # Convert subject identifiers to numeric indices if needed.
    data$subject <- sapply(data$subject, match, unique(unlist(data$subject)))
    
    
    if(input$split_method_own == "row-wise" |input$split_method_own == "subject-wise" ){
      return(run_simulation_own(
        features_sample = data,
        cv              = input$split_method_own,
        n_bootstrap     = input$reps_upload,
        testsize        = input$test_size_upload,
        n_features      = input$n_features_upload
      ))
    }
    
    else if(input$split_method_own == "moving-window" ){
      return(run_simulation_slidingwindow_own(
        features_sample = data,
        n_bootstrap     = input$reps_upload,
        windowsize      = input$windowsize,
        n_features      = input$n_features_upload
      ))
    } else {
      return(data.frame())
    }
  })
  
  output$simulation_baseline_upload <- renderTable({
    req(simulation_baseline_upload())  # Ensure reactive value exists
    result <- simulation_baseline_upload()
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
    return(result)
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  
  output$simulation_results_upload1_baseline <- renderTable({
    req(simulation_baseline_upload())  # Ensure reactive value exists
    result <- simulation_baseline_upload()
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
    return(result[c("AUC:","Accuracy:"),])
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  output$simulation_results_text1_baseline <- renderText({
    req(simulation_baseline_upload())  # Ensure reactive value exists
    result <- simulation_baseline_upload()
    
    # Round the AUC and Accuracy values
    auc_value <- round(result[c("AUC:"),], 2)
    accuracy_value <- round(result[c("Accuracy:"),], 2)
    
    if (input$split_method == "row-wise") {
      if (auc_value > 0.8) {
        return(paste0("
        The baseline model, which removes the relationship between predictor and outcome variables, has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value," 
       This indicates strong performance, with the variance in the outcome primarily explained by differences between individuals. In other words, the model is capable of distinguishing between people based on the included predictors.
Given this strong baseline performance, it may be necessary to reconsider whether the selected use-case scenario and cross-validation strategy are truly meaningful in practice. For the machine learning model that includes the relationship between outcome and predictors to be considered clinically relevant, it must exceed these baseline values.")
        )
      } else {
        return(paste0("The baseline model, which removes the relationship between predictor and outcome variables, has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value," . 
        The machine learning model needs to exceed these values to be clinically relevant.")
        )
      }
      
    } else if (input$split_method == "subject-wise") {
      return(paste0("TO DO: ADD BASELINE?"))
      
    } else if (input$split_method == "moving-window") {
      if (auc_value > 0.8) {
        return(paste0("
        The baseline model, which removes the relationship between predictor and outcome variables, has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value," 
       This indicates strong performance, with the variance in the outcome primarily explained by differences between individuals. In other words, the model is capable of distinguishing between people based on the included predictors.
Given this strong baseline performance, it may be necessary to reconsider whether the selected use-case scenario and cross-validation strategy are truly meaningful in practice. For the machine learning model that includes the relationship between outcome and predictors to be considered clinically relevant, it must exceed these baseline values.
               TO DO: ADD WITHIN-PERSON")
        )
      } else {
        return(paste0("The baseline model, which removes the relationship between predictor and outcome variables, has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value," . 
        The machine learning model needs to exceed these values to be clinically relevant. TO DO: ADD WITHIN-PERSON")
        )
      }
    }
  })
  
  
  
  
  ####################
  
  # Run actual model centered
  simulation_results_upload_centered <- eventReactive(input$run_sim_upload, {
    req(analyzed_data())
    data <- na.omit(analyzed_data())
    
    # Rename columns as needed.
    colnames(data)[colnames(data) == input$id_variable]    <- "subject"
    colnames(data)[colnames(data) == input$time_variable]  <- "time"
    colnames(data)[colnames(data) == input$outcome_variable] <- "y"
    
    # Convert subject identifiers to numeric indices if needed.
    data$subject <- sapply(data$subject, match, unique(unlist(data$subject)))
    
    
    if(input$split_method_own == "row-wise"){
      return(run_simulation_centering_own(
        features_sample = data,
        cv              = input$split_method_own,
        n_bootstrap     = input$reps_upload,
        testsize        = input$test_size_upload,
        n_features      = input$n_features_upload
      ))
    }
    
    else if(input$split_method_own == "moving-window" ){
      return(run_simulation_slidingwindow_own_centering(
        features_sample = data,
        n_bootstrap     = input$reps_upload,
        windowsize      = input$windowsize,
        n_features      = input$n_features_upload
      ))
    } else {
      return(data.frame())
    }
  })
  
  output$simulation_results_upload_centered <- renderTable({
    req(simulation_results_upload_centered())  # Ensure reactive value exists
    result <- simulation_results_upload_centered()
    
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
    return(result)
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  
  output$model_results_text <- renderText({
    req(simulation_results_upload())  # Ensure reactive value exists
    result <- simulation_results_upload()
    
    # Round the relevant metrics for better readability
    auc_value <- round(result[c("AUC:"),], 2)
    accuracy_value <- round(result[c("Accuracy:"),], 2)
    
    mean_auc_within <- round(result[c("Mean AUC within-person:"),], 2)
    sd_auc_within <- round(result[c("SD AUC within-person:"),], 2)
    perc_auc_above_05 <- round(result[c("% of AUC > 0.5 within-person:"),], 1)
    n_included_within <- result[c("N included within-person:"),]
    
    # Define the base result text
    base_text <- paste0(
      "The machine learning model that used non-centered predictors achieved an overall AUC of ", auc_value, 
      " and an accuracy of ", accuracy_value, ". "
    )
    
    # Add cross-validation strategy-specific information
    if (input$split_method_own == "row-wise") {
      within_person_text <- paste0(
        "Since you selected a 'row-wise' cross-validation strategy, within-person performance is an important aspect to consider. ",
        "For within-person performance, the model achieved a mean AUC of ", mean_auc_within, 
        " (SD: ", sd_auc_within, "), with ", perc_auc_above_05, "% of participants having an AUC above 0.5. ",
        "A total of ", n_included_within, " participants were included in this analysis."
      )
    } else if (input$split_method_own == "subject-wise") {
      within_person_text <- paste0(
        "Also for a 'subject-wise' cross-validation strategy, within-person performance may be an important aspect to consider. ",
        "For within-person performance, the model achieved a mean AUC of ", mean_auc_within, 
        " (SD: ", sd_auc_within, "), with ", perc_auc_above_05, "% of participants having an AUC above 0.5. ",
        "A total of ", n_included_within, " participants were included in this analysis."
      )
    } else if (input$split_method_own == "moving-window") {
      within_person_text <- paste0(
        "For within-person performance, the model achieved a mean AUC of ", mean_auc_within, 
        " (SD: ", sd_auc_within, "), with ", perc_auc_above_05, "% of participants having an AUC above 0.5. ",
        "A total of ", n_included_within, " participants were included in this analysis."
      )
    }
    # Combine and return the complete text
    return(paste0(base_text, within_person_text))
  })
  
  output$model_results_text_centered <- renderText({
    req(simulation_results_upload_centered())  # Ensure reactive value exists
    result <- simulation_results_upload_centered()
    
    # Round the relevant metrics for better readability
    auc_value <- round(result[c("AUC:"),], 2)
    accuracy_value <- round(result[c("Accuracy:"),], 2)
    
    mean_auc_within <- round(result[c("Mean AUC within-person:"),], 2)
    sd_auc_within <- round(result[c("SD AUC within-person:"),], 2)
    perc_auc_above_05 <- round(result[c("% of AUC > 0.5 within-person:"),], 1)
    n_included_within <- result[c("N included within-person:"),]
    
    # Add cross-validation strategy-specific information
    if (input$split_method_own == "row-wise") {
      within_person_text <- paste0(
        "The machine learning model that used centered predictors achieved an overall AUC of ", auc_value, 
        " and an accuracy of ", accuracy_value, ". Since you selected a 'row-wise' cross-validation strategy, within-person variability is an important aspect to consider. ",
        "For within-person performance, the model achieved a mean AUC of ", mean_auc_within, 
        " (SD: ", sd_auc_within, "), with ", perc_auc_above_05, "% of participants having an AUC above 0.5. ",
        "A total of ", n_included_within, " participants were included in this analysis."
      )
    } else if (input$split_method_own == "subject-wise") {
      within_person_text <- paste0(
        "Since you selected a 'subject-wise' cross-validation strategy, centered predicots are not relevant in this context."
      )
    } else if (input$split_method_own == "moving-window") {
      within_person_text <- paste0(
        "The machine learning model that used centered predictors achieved an overall AUC of ", auc_value, 
        " and an accuracy of ", accuracy_value, ". For within-person performance, the model achieved a mean AUC of ", mean_auc_within, 
        " (SD: ", sd_auc_within, "), with ", perc_auc_above_05, "% of participants having an AUC above 0.5. ",
        "A total of ", n_included_within, " participants were included in this analysis."
      )
    }
    # Combine and return the complete text
    return(paste0(within_person_text))
  })

  
  
  ###################################################
  ############# Simulation Results Viz ##############
  ###################################################
  
  
  data_res <- read.csv("simulation_results.csv")

  
  # Simulation Results
  output$plot1 <- renderPlot({
    
    
    if (input$filter_values) {
      data_res <- subset(data_res, 
                         A == input$A_viz &
                         icc >= input$icc_viz[1] & icc <= input$icc_viz[2] &
                         sd_residual >= input$sd_residual_viz[1] & sd_residual <= input$sd_residual_viz[2] &
                         mean >= input$mean_viz[1] & mean <= input$mean_viz[2] &
                         sd_outcome >= input$sd_viz[1] & sd_outcome <= input$sd_viz[2] &
                         icc_pred >= input$icc_pred_viz[1] & icc_pred <= input$icc_pred_viz[2] &
                         auc_value_base >= input$auc_value_base_viz[1] & auc_value_base <= input$auc_value_base_viz[2] &
                         auc >= input$auc_value_viz[1] & auc <= input$auc_value_viz[2] &
                         auc_individual >= input$auc_individual_mean_viz[1] & auc_individual <= input$auc_individual_mean_viz[2] &
                         total_n >= input$total_n_viz[1] & total_n <= input$total_n_viz[2] &
                         auc_c >= input$auc_c_mean_viz[1] & auc_c <= input$auc_c_mean_viz[2] &
                         auc_c_individual >= input$auc_c_individual_mean_viz[1] & auc_c_individual <= input$auc_c_individual_mean_viz[2] &
                         sd_intercept >= input$sd_intercept_viz[1] & sd_intercept <= input$sd_intercept_viz[2]
        )
      }
    
    
    # Example visualization
    ggplot(data_res, aes_string(x = input$x_var, y = input$y_var, color = input$color_var)) +
      geom_point(alpha = 0.5, size = 3) +
      theme_minimal() +
      scale_colour_gradientn(colours = c("#2A363B", "#83AF9B", "#C8C8A9", "#F9CDAD", "#FC9D9A", "#FE4365")) +
      ylab(input$y_var) +
      xlab(input$x_var) +
      guides(col = guide_colourbar()) +
      ggtitle("") +
      geom_hline(yintercept = 0.5)
    
   
  })
  
  # Simulation Results
  output$plot3 <- renderPlot({
    # Retrieve data
    if (input$filter_values) {
      data_res <- subset(data_res, 
                     A == input$A_viz &
                       icc >= input$icc_viz[1] & icc <= input$icc_viz[2] &
                       sd_residual >= input$sd_residual_viz[1] & sd_residual <= input$sd_residual_viz[2] &
                       mean >= input$mean_viz[1] & mean <= input$mean_viz[2] &
                       sd_outcome >= input$sd_viz[1] & sd_outcome <= input$sd_viz[2] &
                       icc_pred >= input$icc_pred_viz[1] & icc_pred <= input$icc_pred_viz[2] &
                       auc_value_base >= input$auc_value_base_viz[1] & auc_value_base <= input$auc_value_base_viz[2] &
                       auc >= input$auc_value_viz[1] & auc <= input$auc_value_viz[2] &
                       auc_individual >= input$auc_individual_mean_viz[1] & auc_individual <= input$auc_individual_mean_viz[2] &
                       total_n >= input$total_n_viz[1] & total_n <= input$total_n_viz[2] &
                       auc_c >= input$auc_c_mean_viz[1] & auc_c <= input$auc_c_mean_viz[2] &
                       auc_c_individual >= input$auc_c_individual_mean_viz[1] & auc_c_individual <= input$auc_c_individual_mean_viz[2] &
                       sd_intercept >= input$sd_intercept_viz[1] & sd_intercept <= input$sd_intercept_viz[2]
      )
    }

    data_res$auc_diff <- data_res$auc_c_individual - data_res$auc_individual
    
    # Example visualization
    ggplot(data_res, aes_string(x = input$x_var, y = "auc_diff", color = input$color_var)) +
      geom_point(alpha = 0.5, size = 3) +
      theme_minimal() +
      scale_colour_gradientn(colours = c("#2A363B", "#83AF9B", "#C8C8A9", "#F9CDAD", "#FC9D9A", "#FE4365")) +
      ylab("AUC Difference (Within Person)") +
      xlab(input$x_var) +
      guides(col = guide_colourbar()) +
      ggtitle("") +
      geom_hline(yintercept = 0)
    
  })
  
  
}


# Run the application
shinyApp(ui = ui, server = server)
