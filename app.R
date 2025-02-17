renv::init()
if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
#renv::restore()

library(shiny)
library(bslib)
library(randomForest)
library(dplyr)
library(lme4)
library(caret)
library(pROC)
library(boot)
library(ggplot2)
library(shinycssloaders)
library(DT)
library(shinyjs)


source("Simulation_Functions.R")
source("Simulation_UploadData.R")

ui <- fluidPage(
  useShinyjs(), 
  theme = bs_theme(version = 5, bootswatch = "journal"),
  
  titlePanel("Just in Time or Just a Guess? Validating Prediction Models Based on Longitudinal Data"),
  
  tabsetPanel(
    tabPanel("General information",
             fluidRow(
               column(4,
                      # Placeholder for an image
                      wellPanel(
                        h4("How to Use the App"),
                        p("In the ", 
                          tags$strong(tags$span(style = "color: #83AF9B;", "'Upload your own data'")), 
                          " tab, researchers can upload their own data and examine common pitfalls that might occur in their evaluation strategy."
                        ),
                        p("In the ", 
                          tags$strong(tags$span(style = "color:#83AF9B;", "'Create simulated data'")), 
                          " tab, researchers can simulate datasets with specific parameters. These simulated datasets can be downloaded and later uploaded in the 'Upload your own data' tab. This provides an opportunity to check for potential issues before actual data collection, and also allows replication of the simulation."
                        )
                      )
               ),
               column(8,
                      h4("Welcome"),
                      p("This Shiny app accompanies the paper 'Just in Time or Just a Guess? Addressing Challenges in Validating Prediction Models Based on Longitudinal Data (in prep, Langener & Jacobson, 2025)'. The goal of this app is to guide you in developing and evaluating prediction models using longitudinal data, particularly through train/test splits or cross-validation methods."),
                      h4("How can this tool help you?"),
                      p("It is designed to help you check if your chosen cross-validation strategy is aligned with your research goals. You can either upload your own data or simulate new data. For a detailed explanation of common pitfalls and the simulation setup and parameters, make sure to check out our paper."),
                      
                      p("We hope this tool is helpful in advancing your research on longitudinal data analysis and prediction models."),
                      h4("Any questions?"),
                      p("Feel free to explore the app and reach out for any questions or feedback ",
                        a(href = "mailto:anna.m.langener@dartmouth.edu", 
                          icon("envelope"), "anna.m.langener@dartmouth.edu")
                       
                      ),
                      br(),
                      accordion(
                        accordion_panel(
                          title = "Information on Input Fields",
                          p(style = "font-size: 14px; color: grey;", 
                            "Note: This information will also appear dynamically in the 'Create simulated data' and 'Upload your own data' tabs in most browsers."
                          ),
                          tableOutput("static_info_table")  # Table with static information
                        )
                      )
               ),
             )
    ),
    tabPanel("Create simulated data",
             sidebarLayout(
               sidebarPanel(
                   h5("Basic Parameters"),
                   
                   tooltip(numericInput("n_features_sim", "Number of Features:", value = 10, min = 1, step = 1),  
                           "Number of variables/features/ predictors included in the analysis. These features are used to predict the outcome.", 
                           placement = "right"),
                   
                   tooltip(numericInput("n_samples_sim", "Number of Samples (timepoints per subject):", value = 60, min = 1, step = 1),
                           "Indicates how often each subject is sampled. A higher value represents more timepoints per participant.", 
                           placement = "right"),
                   
                   tooltip(numericInput("n_subjects", "Number of Subjects:", value = 90, min = 2, step = 2),
                           "How many participants are included in the analysis.", 
                           placement = "right"),
                   
                   h5("Outcome Parameters"),
                   
                   tooltip(numericInput("overall_prob_outcome", "Overall Probability of Outcome:", value = 0.1, min = 0, max = 1, step = 0.01),
                           "Overall probability of the outcome being present ('1') across participants and time points. A higher mean results in a greater likelihood of the outcome being present.", 
                           placement = "right"),
                   
                   tooltip(numericInput("sd_outcome", "Standard Deviation of Outcome (Between Subjects):", value = 0.1, step = 0.01),
                           "Controls the variability of the outcome probability across participants. A larger standard deviation means more variability between participants.", 
                           placement = "right"),
                   
                   h5("Variable/Feature Generation"),
                   
                   tooltip(numericInput("A", "Relationship Between Features and Outcome (A):", value = 0.05, step = 0.01),
                           "Magnitude of the outcome effect, where 'A' reflects the strength of the relationship between features and outcome at timepoints with the effect and '-A' reflects the strength of the relationship between features and outcome at timepoints with no effect.", 
                           placement = "right"),
                   
                   tooltip(numericInput("feature_std", "Population-level Feature Variability (Feature Noise):", value = 0.1, step = 0.01),
                           "The noise/variability introduced at the population level for features. A larger value reflects more variability.", 
                           placement = "right"),
                   
                   tooltip(numericInput("B", "Between-subject variability (random effects, B):", value = 0.7, step = 0.01),
                           "A higher value increases the variability of predictors between subjects, meaning that differences in the predictors are primarily due to individual participant characteristics.", 
                           placement = "right"),
                   
                   tooltip(numericInput("C", "Within-Subject Variability (C):", value = 0.1, step = 0.01),
                           "Reflects variability within subjects across timepoints and features.", 
                           placement = "right"),
                   
                   h5("Simulation Parameters"),
                   
                   tooltip(selectInput("split_method", "Data Split Method:", choices = c("record-wise", "subject-wise", "moving-window"), selected = "record-wise"),
                           "Choose how the data will be split: 'record-wise' for splitting by individual data points, 'subject-wise' for splitting by participants, or 'moving-window' for a sliding time window.", 
                           placement = "right"),
                   
                   tooltip(numericInput("test_size", "Test Set Size (Proportion):", value = 0.3, min = 0.1, max = 0.9, step = 0.1),
                           "The proportion of the dataset allocated to the test set. Only relevant for 'record-wise' and 'subject-wise' data split methods.", 
                           placement = "right"),
                   
                   tooltip(numericInput("windowsize_sim", "Window size (moving-window only, timepoints):", value = 14, min = 1, max = 100, step = 1),
                           "For the moving-window method, this defines how many timepoints are included in each sliding window.", 
                           placement = "right"),
                   
                   tooltip(numericInput("replications", "Number of Replications:", value = 1, min = 1, step = 1),
                           "DELETE? The number of times the simulation should be repeated. This can be used to increase the robustness of the results.",
                           placement = "right"),
                   
                   actionButton("generate_data", "Generate Data"),
                   actionButton("run_sim", "Run Model")
                 ),
               
               mainPanel(
                 h5("Download Simulated Data"),
                 p("Note: After generating the data you can use this data as an example in the 'Upload your own data' tab. For more information on how the data is simulated, see 'Just in Time or Just a Guess? Addressing Challenges in Validating Prediction Models Based on Longitudinal Data (in prep, Langener & Jacobson, 2025)'."),
                 downloadButton("download_sim", "Download Simulated Data"),
                 tags$br(),
                 tags$br(),
                 
                 accordion(
                   accordion_panel(
                     title = "Preview of Simulated Dataset",
                     p("Below is a preview of your simulated dataset. Variables starting with 'V' represent the simulated predictor variables. 'Subject' indicates the subject identifier, 'time' represents the time point, 'Y' is the binary outcome, and 'A' denotes the relationship to the outcome."),
                     
                     tableOutput("generated_features")
                   ),
                   
                   accordion_panel(
                     title = "Preview Simulation Results",
                     h5("Info"),
                     p("You can download this data and upload it under the 'Upload your own data' tab for further exploration, visualization, and analysis. For a quick preview of model performance, you can find the results below. We ran a simple Random Forest model using your selected parameters, executing it twice—once with non-centered predictors and once with centered predictors."),
                     
                     h5("Simulation Results (Not Centered)"),
                     withSpinner(tableOutput("simulation_results")),
                     
                     h5("Simulation Results (Centered)"),
                     withSpinner(tableOutput("simulation_results_centered"))
                
                   )  
                 ),
             
            
               )
             )
    ),

    # Data Upload Tab
    tabPanel("Upload your own data",
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
                 tooltip(selectInput("split_method_own", "Data Split Method:", choices = c("record-wise", "subject-wise", "moving-window"), selected = "record-wise"),
                         "Choose how the data will be split: 'record-wise' for splitting by individual data points, 'subject-wise' for splitting by participants, or 'moving-window' for a sliding time window.", 
                         placement = "right"),
                 tooltip(numericInput("test_size_upload", "Test Set Size (Proportion):", value = 0.3, min = 0.1, max = 0.9, step = 0.1),
                         "The proportion of the dataset allocated to the test set. Only relevant for 'record-wise' and 'subject-wise' data split methods.", 
                         placement = "right"),
                 
                 tooltip(numericInput("windowsize", "Window size (moving-window only, timepoints):", value = 14, min = 1, max = 100, step = 1),
                         "For the moving-window method, this defines how many timepoints are included in each sliding window.", 
                         placement = "right"),
                 
                 tooltip(numericInput("reps_upload", "Number of Replications:", value = 1, min = 1, step = 1),
                         "DELETE? The number of times the simulation should be repeated. This can be used to increase the robustness of the results.",
                         placement = "right"),
                 actionButton("run_sim_upload", "Run Diagnostics")
               ),
               
               mainPanel(
                 accordion( 
                   accordion_panel( 
                     title = "1. Define Use Case and Select Matching Cross-Validation Approach", 
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
                    tags$br(),
                    h5("Baseline Model 2: Shuffled Outcome Variable"),
                    h6("Variance Explained by Between Person Differences"),
                    textOutput("methods_text_icc"),
                    tags$br(),
                    div(DT::dataTableOutput("icc_table"), style = "font-size: 75%; width: 75%"),
                    
                    tags$br(),
                    h6("Model Performance"),
                    withSpinner(textOutput("simulation_results_text1_baseline")),
                    tags$br(),
                    tableOutput("simulation_results_upload1_baseline"),
                  ),
                  accordion_panel( 
                    title = "4. Include Within-Person Evaluation for Tracking and Consider Centering", 
                    h5("General Information"),
                    "When tracking individuals over time and predicting within-person differences, model performance should be evaluated both across the full dataset and within each individual. In our paper we have shown that centering predictors can improve accuracy and reduce bias, so we present results for both centered and uncentered predictors. In both cases we ran a simple Random Forest model.",
                    tags$br(),
                    tags$br(),
                    h5("Simulation Results (not centered)"),
                    withSpinner(textOutput("model_results_text")),
                    tags$br(),
                    tableOutput("simulation_results_upload"),
                    tags$br(),
                    h5("Simulation Results (centered)"),    
                    withSpinner(textOutput("model_results_text_centered")),
                    tags$br(),
                    tableOutput("simulation_results_upload_centered"),
                  ),
                   ),  
          
                 
               )
             )
    ),selected="General information"
  )
)

server <- function(input, output, session) {
  # Static Information Table for the first tab
  output$static_info_table <- renderTable({
    data.frame(
      "Field" = c(
        "Number of Features",
        "Number of Samples (timepoints per subject)",
        "Number of Subjects",
        "Overall Probability of Outcome",
        "Standard Deviation of Outcome (Between Subjects)",
        "Relationship Between Features and Outcome (A)",
        "Population-level Feature Variability (Feature Noise)",
        "Between-subject variability (random effects, B)",
        "Within-Subject Variability (C)",
        "Data Split Method",
        "Test Set Size (Proportion)",
        "Window Size (Moving-Window Only)",
        "Number of Replications"
      ),
      "Description" = c(
        "Number of variables/features/predictors included in the analysis. These features are used to predict the outcome.",
        "Indicates how often each subject is sampled. A higher value represents more timepoints per participant.",
        "How many participants are included in the analysis.",
        "Overall probability of the outcome being present ('1') across participants and time points. A higher mean results in a greater likelihood of the outcome being present.",
        "Controls the variability of the outcome probability across participants. A larger standard deviation means more variability between participants.",
        "Magnitude of the outcome effect, where 'A' reflects the strength of the relationship between features and outcome at timepoints with the effect and '-A' reflects the strength of the relationship between features and outcome at timepoints with no effect.",
        "The noise/variability introduced at the population level for features. A larger value reflects more variability.",
        "A higher value increases the variability of predictors between subjects, meaning that differences in the predictors are primarily due to individual participant characteristics.",
        "Reflects variability within subjects across timepoints and features.",
        "Choose how the data will be split: 'record-wise' for splitting by individual data points, 'subject-wise' for splitting by participants, or 'moving-window' for a sliding time window.",
        "The proportion of the dataset allocated to the test set. Only relevant for 'record-wise' and 'subject-wise' data split methods.",
        "For the moving-window method, this defines how many timepoints are included in each sliding window.",
        "The number of times the simulation should be repeated. This can be used to increase the robustness of the results."
      ),
      stringsAsFactors = FALSE
    )
  })
  
  #########################################################
  ######################## Data Sim #######################
  #########################################################
  observeEvent(input$info_n_features, {
    showModal(modalDialog(
      title = "Number of Features",
      "Specify the number of features (variables) in the dataset. This will determine how many different types of data are generated for each subject.",
      easyClose = TRUE,
      footer = NULL
    ))
  })
  
  observe( {
    req(input$overall_prob_outcome, input$sd_outcome) 
    max_sd <- sqrt(input$overall_prob_outcome * (1 - input$overall_prob_outcome))
    
    if (input$sd_outcome > max_sd) {
      showModal(modalDialog(
        title = "Warning",
        paste("Given your selected outcome probability, the standard deviation should be smaller than",round(max_sd,2)),
        easyClose = TRUE,
        footer = NULL
      ))
    }
  })
  


  generated_features <- eventReactive(input$generate_data, {
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
  
  
  output$download_sim <- downloadHandler(
    filename = function() {
      paste("simulated_data.csv", sep = "")  # File name for download
    },
    content = function(file) {
      req(generated_features())  # Ensure data is available before downloading
      data <- generated_features()  # Retrieve generated data
      write.csv(data, file, row.names = FALSE)  # Write data to CSV
    }
  )
  
  
  simulation_results <- eventReactive(input$run_sim, {
    req(generated_features())
  
    if(input$split_method == "record-wise" |input$split_method == "subject-wise" ){
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
    if(input$split_method == "record-wise"){
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
    print(mem_used())})
  
  
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
    
    if (is.null(result_cen) || nrow(result_cen) == 0) {
      return(data.frame(Message = "No centered results available"))  # Display a message instead of an empty table
    }
    
    return(result_cen)
  }, include.rownames = TRUE, include.colnames = FALSE)
  
  #########################################################
  ######################## Tab 2 ##########################
  #########################################################
  
  ############ Data Upload/ Settings ##############
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
  
  ######### First Section ##########
  
  output$output_text_1 <- renderText({
    switch(input$split_method_own,
           "record-wise" = "Choosing a cross-validation strategy that aligns with your specific use case and study objectives is important for ensuring meaningful results. You selected a 'record-wise' cross-validation split. This means that participants are present in both the training and testing sets. 
         This split is useful when the goal is to track individuals over time, as it allows for capturing within-person variability and between-person variability. It is particularly relevant when the model will be applied to the same group of individuals.",
           
           "subject-wise" = "Choosing a cross-validation strategy that aligns with your specific use case and study objectives is important for ensuring meaningful results. You selected a 'subject-wise' cross-validation split. In this approach, the data is split such that each subject is either in the training or testing set, but not both. 
         This split is useful when the aim is to assess the model’s ability to generalize to new, unseen subjects. It ensures that the training and testing sets are independent, making it a better option for evaluating performance in cases where the model needs to generalize across different individuals or populations.",
           
           "moving-window" = "Choosing a cross-validation strategy that aligns with your specific use case and study objectives is important for ensuring meaningful results. You selected a 'moving-window' cross-validation split. This method involves using a moving window of data for training and testing. 
         In this split, a fixed-size training set is used, and as the model moves forward in time, the training set is updated, and a new testing set is created. This approach is useful for time-series data or when you want to simulate the model’s performance in a dynamic setting, where the training data gradually changes over time as new information becomes available.
           Importantly, this split assumes that the model will be updated over time.")
  })
  
  output$cv_image <- renderUI({
    req(input$split_method_own)  # Ensure cv_choice is selected before rendering
    
    # Define image path based on input
    img_src <- switch(input$split_method_own,
                      "moving-window" = "moving-window.png",
                      "record-wise" = "Record-wise.png",
                      "subject-wise" = "Subject-wise.png")
    tags$img(src = img_src, width = "40%", height = "40%")
  })
  
  observeEvent(input$run_sim_upload, {
    # Check if the 'n_features_upload' is empty
    if (length(input$n_features_upload) == 0) {
      # Display a warning message
      showModal(modalDialog(
        title = "Warning",
        "Please select at least one predictor before running diagnostics.",
        easyClose = TRUE,
        footer = NULL
      ))
    } else {
  
  # # 3. Process the uploaded data when the "analyze_data" button is clicked.
  analyzed_data <- eventReactive(input$run_sim_upload, {
    req(uploaded_data())
    data <- uploaded_data()
    
    data  # Return the (possibly shuffled) dataset.
    
    
  })
  
  
  ######### Second Section ##########
  
  output$output_text_2 <- renderText({
    switch(input$split_method_own,
           "record-wise" = "The next step is to assess whether the chosen use case scenario and cross-validation strategy are appropriate for your objectives. 
         You have selected a 'record-wise' cross-validation strategy, which is particularly useful if your goal is to capture within-person variability. 
         Below, we have visualized the data and calculated some basic descriptives. Consider whether the outcome shows sufficient variability for your intended use case. If the variability is low, it may be worth shifting to a task that focuses on distinguishing between individuals.",
           
           "subject-wise" = "The next step is to evaluate whether the chosen use case scenario and cross-validation strategy are suitable for your objectives. 
         You have chosen a 'subject-wise' cross-validation strategy, which is particularly useful if your goal is to differentiate between individuals.",
           
           "moving-window" = "The next step is to assess whether the chosen use case scenario and cross-validation strategy are appropriate for your objectives. 
         You have selected a 'moving-window' cross-validation strategy, which is particularly useful for capturing within-person variability over time. 
         Below, we have visualized the data and calculated some basic descriptives. Consider whether the outcome has sufficient variability for your intended use case. If variability is low, consider shifting to a task focused on distinguishing between individuals.")
  })
  
  
  ### Descriptive
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
  
  # Descriptive Table
  output$descriptives <- renderTable({
    model_result <- model_data()
    
    data.frame(
      "Overall Outcome Probability" = round(model_result$overall_prob_outcome, 3),
      "Outcome SD"                  = round(model_result$sd_outcome, 3),
      "Number of Subjects"          = model_result$num_subjects,
      "Number of Samples per Subject" = model_result$num_samples,
      check.names = FALSE
    )
  })
  
  # Descriptive text
  output$methods_text <- renderText({
    model_result <- model_data()
    
    paste0(
      "A total of ", model_result$num_subjects, " unique subjects are included in the analysis. ",
      "Each subject has a maximum of ", model_result$num_samples, " repeated measurements. ",
      "The mean outcome probability is ", round(model_result$overall_prob_outcome, 3), 
      " (SD = ", round(model_result$sd_outcome, 3), ")."
    )
  })
  
  
  
  
  output$methods_text_icc <- renderText({
    model_result <- model_data()
    
    if(input$split_method_own == "record-wise" | input$split_method_own == "moving-window"){
    paste0(
      "In addition to a random intercept model, it is useful to include a baseline model with predictors that have no relationship to the outcome. This helps determine if the predictors are merely distinguishing between individuals or if they capture a relationship to the outcome. If a significant portion of the variance in both the outcome and predictors is explained by between-person differences, the model may merely differentiate between individuals. The percentage of variation in the outcome that is explained by differences between people is ", 
      round(model_result$icc_data, 3), " In the table below, you can find the percentage of variance in the predictors explained by between-person differences."
    )}else(
      paste0("You have chosen a subject-wise split. Since subject-wise validation separates participants between training and testing sets, this section is not relevant for your research.")
    )
  })
  
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
                               "record-wise" = "As you have chosen a 'record-wise' cross-validation strategy, it is important to capture within-person variability. This means that there must be sufficient variability in the outcome.",
                               "subject-wise" = "You chose a 'subject-wise' cross-validation strategy, which is ideal for differentiating between subjects. The level of variability may influence how well this strategy generalizes to new subjects but is generally less important.",
                               "moving-window" = "With the 'moving-window' cross-validation strategy, it is important to capture within-person variability. This means that there must be sufficient variability in the outcome. Additionally, time-series aspects of the data are also important. The variability over time will affect the performance of this strategy in capturing temporal trends.",
                               "Unknown strategy selected.")  # Default case if no valid strategy is selected
    
    # Combine all the text into the final dynamic title text
    dynamic_title_text <- paste(base_text, variability_text, cv_strategy_text)
    
    dynamic_title_text
  })

  
  ######### Third Section ##########
  #  ICC Table.
  output$icc_table <- DT::renderDT({
    if(input$split_method_own == "record-wise" | input$split_method_own == "moving-window"){
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
    }
  },options = list(pageLength = 5), server = FALSE)
  

  
  ################## Run Models
  
  ######### Text and output baseline 1 ##########
  ## Actual model
  
  simulation_results_upload <- eventReactive(input$run_sim_upload, {
    req(analyzed_data())
    data <- na.omit(analyzed_data())
    
    # Rename columns as needed.
    colnames(data)[colnames(data) == input$id_variable]    <- "subject"
    colnames(data)[colnames(data) == input$time_variable]  <- "time"
    colnames(data)[colnames(data) == input$outcome_variable] <- "y"
    
    # Convert subject identifiers to numeric indices if needed.
    data$subject <- sapply(data$subject, match, unique(unlist(data$subject)))
    
    
    if(input$split_method_own == "record-wise" |input$split_method_own == "subject-wise" ){
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
  
  
  ##### BASELINE 1
  
  output$simulation_results_upload1 <- renderTable({
    req(simulation_results_upload())  # Ensure reactive value exists
    result <- NULL
    
    if (input$split_method_own == "record-wise") {
    result <- simulation_results_upload()
    return(result[c("AUC random intercept only:","Accuracy random intercept only:"),])
    }
    
    if (input$split_method_own == "moving-window") {
      result <- simulation_results_upload()
      result_ind <- result[[2]]
      result <- result[[1]]

      auc_within <- result_ind[[1]] %>%
        group_by(subject) %>%
        filter(length(unique(true)) > 1) %>%
        filter(sum(true == 1) > 0, sum(true == 0) > 0) %>%
        summarise(
          auc_val = auc(roc(as.numeric(as.character(true)), as.numeric(as.character(pred)), quiet = TRUE))[1],
          .groups = "drop" 
        )
      
      mean_auc_within <- mean(auc_within$auc_val, na.rm = TRUE)
      new_row <- data.frame(V1 = mean_auc_within, row.names = "Mean AUC within-person random intercept only:")
      result <- rbind(result, new_row)
      selected_rows <- c("AUC random intercept only:", "Accuracy random intercept only:", "Mean AUC within-person random intercept only:")
      result <- result[rownames(result) %in% selected_rows, , drop = FALSE]
      return(result)
    }
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
   
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  output$simulation_results_text1 <- renderText({
    req(simulation_results_upload())  # Ensure reactive value exists
    if (input$split_method_own == "moving-window") {
      result <- simulation_results_upload()
      result_ind <- result[[2]]
      result <- result[[1]]
      
      auc_within <- result_ind[[1]] %>%
        group_by(subject) %>%
        filter(length(unique(true)) > 1) %>%
        filter(sum(true == 1) > 0, sum(true == 0) > 0) %>%
        summarise(
          auc_val = auc(roc(as.numeric(as.character(true)), as.numeric(as.character(pred)), quiet = TRUE))[1],
          .groups = "drop" 
        )
      
      mean_auc_within <- round(mean(auc_within$auc_val, na.rm = TRUE),2)
      auc_value <- round(result[c("AUC random intercept only:"),], 2)
      accuracy_value <- round(result[c("Accuracy random intercept only:"),], 2)
      
    }else{
    result <- simulation_results_upload()
    auc_value <- round(result[c("AUC random intercept only:"),], 2)
    accuracy_value <- round(result[c("Accuracy random intercept only:"),], 2)
    }

    if (input$split_method_own == "record-wise") {
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
      
    } else if (input$split_method_own == "subject-wise") {
      return(paste0("You have chosen a subject-wise split. Since subject-wise validation separates participants between training and testing sets, there is no participant-specific majority class in the training data, making a direct baseline comparison based on participants' majority class infeasible."))
      
    } else if (input$split_method_own == "moving-window") {
      if (auc_value > 0.8) {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        The mean within-person AUC is ", mean_auc_within,
        ". Thus, the baseline model already demonstrates overall strong performance. 
        Given this, it may be necessary to reconsider whether the selected use-case scenario and cross-validation strategy are truly meaningful in practice. 
        The machine learning model must exceed these values to be considered clinically relevant.")
        )
      } else {
        return(paste0("The model's performance needs to be compared with a baseline model to accurately assess its effectiveness and avoid misleading conclusions. 
        For the model to be clinically useful, it must outperform a simple baseline model. 
        In this case, a simple random intercept-only baseline model (which includes no predictors) has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value, ". 
        The mean within-person AUC is ", mean_auc_within,
        "Therefore, the machine learning model needs to exceed these values to be clinically relevant.")
        )
      }
    }
  })
  
  #### Baseline 2 (SHUFFLED!!!!)
  simulation_baseline_upload <- eventReactive(input$run_sim_upload, {
    req(analyzed_data())
    data <- na.omit(analyzed_data())
    data <- shuffle_data(data, subject_var = input$id_variable, outcome_var = input$outcome_variable)
    
    # Rename columns as needed.
    colnames(data)[colnames(data) == input$id_variable]    <- "subject"
    colnames(data)[colnames(data) == input$time_variable]  <- "time"
    colnames(data)[colnames(data) == input$outcome_variable] <- "y"
    
    # Convert subject identifiers to numeric indices if needed.
    #data$subject <- sapply(data$subject, match, unique(unlist(data$subject)))
    
    
    if(input$split_method_own == "record-wise" |input$split_method_own == "subject-wise" ){
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
  
  
  output$simulation_results_upload1_baseline <- renderTable({
    req(simulation_baseline_upload())  # Ensure reactive value exists
    result <- NULL
    if(input$split_method_own == "record-wise" ){
    result <- simulation_baseline_upload()
    return(result[c("AUC:","Accuracy:"),])
    }else if (input$split_method_own == "moving-window") {
      result <- simulation_baseline_upload()
      result <- result[[1]]
      return(result[c("AUC:","Accuracy:","Mean AUC within-person:"),])
    }
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  output$simulation_results_text1_baseline <- renderText({
    req(simulation_baseline_upload())  # Ensure reactive value exists
    if (input$split_method_own == "moving-window") {
      result <- simulation_baseline_upload()
      result <- result[[1]]
    }else{
      result <- simulation_baseline_upload()
    }
    
    # Round the AUC and Accuracy values
    auc_value <- round(result[c("AUC:"),], 2)
    accuracy_value <- round(result[c("Accuracy:"),], 2)
    mean_auc_within <- round(result[c("Mean AUC within-person:"),], 2)
    
    if (input$split_method_own == "record-wise") {
      if (auc_value > 0.8) {
        return(paste0("
        To assess the impact of the variance explained by the predictor variables, it’s helpful to include, in addition to a random intercept model, a baseline model with predictors that have no true relationship to the outcome. We achieve this by shuffling the outcome variable within each person and also shuffling the subject identifiers. This preserves the outcome distribution but removes any true relationship with the predictors. We then run a random forest model. Please note that both baseline models should perform worse than your actual model to ensure that the predictors are truly contributing to the outcome. \nThe baseline model, which removes the relationship between predictor and outcome variables, has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value," 
       This indicates strong performance, with the variance in the outcome primarily explained by differences between individuals. In other words, the model is capable of distinguishing between people based on the included predictors.
Given this strong baseline performance, it may be necessary to reconsider whether the selected use-case scenario and cross-validation strategy are truly meaningful in practice. For the machine learning model that includes the relationship between outcome and predictors to be considered clinically relevant, it must exceed these baseline values.")
        )
      } else {
        return(paste0("To assess the impact of the variance explained by the predictor variables, it’s helpful to include, in addition to a random intercept model, a baseline model with predictors that have no true relationship to the outcome. We achieve this by shuffling the outcome variable within each person and also shuffling the subject identifiers. This preserves the outcome distribution but removes any true relationship with the predictors. We then run a random forest model. Please note that both baseline models should perform worse than your actual model to ensure that the predictors are truly contributing to the outcome.\nThe baseline model, which removes the relationship between predictor and outcome variables, has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value,".")
        )
      }
      
    } else if (input$split_method_own == "moving-window") {
      if (auc_value > 0.8) {
        return(paste0("
        To assess the impact of the variance explained by the predictor variables, it’s helpful to include, in addition to a random intercept model, a baseline model with predictors that have no true relationship to the outcome. We achieve this by shuffling the outcome variable within each person and also shuffling the subject identifiers. This preserves the outcome distribution but removes any true relationship with the predictors. We then run a random forest model. Please note that both baseline models should perform worse than your actual model to ensure that the predictors are truly contributing to the outcome.\nThe baseline model, which removes the relationship between predictor and outcome variables, has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value," 
       The mean within-person performance is ", mean_auc_within,
       ". This indicates strong overall performance, with the variance in the outcome primarily explained by differences between individuals. In other words, the model is capable of distinguishing between people based on the included predictors.
Given this strong baseline performance, it may be necessary to reconsider whether the selected use-case scenario and cross-validation strategy are truly meaningful in practice. For the machine learning model that includes the relationship between outcome and predictors to be considered clinically relevant, it must exceed these baseline values.")
        )
      } else {
        return(paste0("To assess the impact of the variance explained by the predictor variables, it’s helpful to include, in addition to a random intercept model, a baseline model with predictors that have no true relationship to the outcome. We achieve this by shuffling the outcome variable within each person and also shuffling the subject identifiers. This preserves the outcome distribution but removes any true relationship with the predictors. We then run a random forest model. Please note that both baseline models should perform worse than your actual model to ensure that the predictors are truly contributing to the outcome.\nThe baseline model, which removes the relationship between predictor and outcome variables, has an overall AUC of ", auc_value, " and an overall accuracy of ", accuracy_value,".The mean within-person performance is ", mean_auc_within,".")
        )
      }
    }
  })
  
  
  
  #################### Section 4 ################
  ####### Table 1 (not centered) ############
  output$simulation_results_upload <- renderTable({
    req(simulation_results_upload())  # Ensure reactive value exists
    if (input$split_method_own == "moving-window") {
      result <- simulation_results_upload()
      result <- result[[1]]
    }else{
      result <- simulation_results_upload()
    }
    if (is.null(result) || nrow(result) == 0) {
      return(data.frame(Message = "No results available"))  # Display a message instead of an empty table
    }
    
    return(result)
  }, include.rownames = TRUE, include.colnames = FALSE) 
  
  
  
  ######### Model Centered ##########
  simulation_results_upload_centered <- eventReactive(input$run_sim_upload, {
    req(analyzed_data())
    data <- na.omit(analyzed_data())
    
    # Rename columns as needed.
    colnames(data)[colnames(data) == input$id_variable]    <- "subject"
    colnames(data)[colnames(data) == input$time_variable]  <- "time"
    colnames(data)[colnames(data) == input$outcome_variable] <- "y"
    
    # Convert subject identifiers to numeric indices if needed.
    data$subject <- sapply(data$subject, match, unique(unlist(data$subject)))
    
    
    if(input$split_method_own == "record-wise"){
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
  
  
  ########## True Result Model text ############
  output$model_results_text <- renderText({
    req(simulation_results_upload())  # Ensure reactive value exists
    if (input$split_method_own == "moving-window") {
      result <- simulation_results_upload()
      result <- result[[1]]
    }else{
      result <- simulation_results_upload()
    }
    
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
    if (input$split_method_own == "record-wise") {
      within_person_text <- paste0(
        "Since you selected a 'record-wise' cross-validation strategy, within-person performance is an important aspect to consider. ",
        "For within-person performance, the model achieved a mean AUC of ", mean_auc_within, 
        " (SD: ", sd_auc_within, "), with ", perc_auc_above_05, "% of participants having an AUC above 0.5. ",
        "A total of ", n_included_within, " participants were included in the within-person performance analysis (the other participants did not experience any variability in the test split)."
      )
    } else if (input$split_method_own == "subject-wise") {
      within_person_text <- paste0(
        "Also for a 'subject-wise' cross-validation strategy, within-person performance may be an important aspect to consider. ",
        "For within-person performance, the model achieved a mean AUC of ", mean_auc_within, 
        " (SD: ", sd_auc_within, "), with ", perc_auc_above_05, "% of participants having an AUC above 0.5. ",
        "A total of ", n_included_within, " participants were included in the within-person performance analysis (the other participants did not experience any variability in the test split)."
      )
    } else if (input$split_method_own == "moving-window") {
      within_person_text <- paste0(
        "For within-person performance, the model achieved a mean AUC of ", mean_auc_within, 
        " (SD: ", sd_auc_within, "), with ", perc_auc_above_05, "% of participants having an AUC above 0.5. ",
        "A total of ", n_included_within, " participants were included in the within-person performance analysis (the other participants did not experience any variability in the test split)."
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
    if (input$split_method_own == "record-wise") {
      within_person_text <- paste0(
        "The machine learning model that used centered predictors achieved an overall AUC of ", auc_value, 
        " and an accuracy of ", accuracy_value, ". Since you selected a 'record-wise' cross-validation strategy, within-person variability is an important aspect to consider. ",
        "For within-person performance, the model achieved a mean AUC of ", mean_auc_within, 
        " (SD: ", sd_auc_within, "), with ", perc_auc_above_05, "% of participants having an AUC above 0.5. ",
        "A total of ", n_included_within, " participants were included in the within-person performance analysis (the other participants did not experience any variability in the test split)."
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
        "A total of ", n_included_within, " participants were included in the within-person performance analysis (the other participants did not experience any variability in the test split)."
      )
    }
    # Combine and return the complete text
    return(paste0(within_person_text))
  })
    }
  })
}
  
shinyApp(ui = ui, server = server)

