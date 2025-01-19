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
library(shiny)
library(bslib)
library(ggplot2)
library(dplyr)
library(lme4)
library(tseries)

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
                 numericInput("n_features", "Number of Features:", value = 10, min = 1, step = 1),
                 numericInput("n_samples", "Number of Samples (timepoints per subject):", value = 90, min = 1, step = 1),
                 numericInput("n_subjects", "Number of Subjects:", value = 150, min = 2, step = 2),
                 
                 h5("Outcome Parameters"),
                 numericInput("overall_prob_outcome", "Overall Probability of Outcome:", value = 0.1, min = 0, max = 1, step = 0.01),
                 numericInput("sd_outcome", "Standard Deviation of Outcome (Between Subjects):", value = 0.25, step = 0.01),
                 checkboxInput("time_effect", "Include Time Effect", value = FALSE),
                 
                 h5("Feature Generation"),
                 numericInput("A", "Relationship Between Features and Outcome (A):", value = 0.05, step = 0.01),
                 numericInput("feature_std", "Population-level Feature Variability (Feature Std):", value = 0.1, step = 0.01),
                 numericInput("B", "Cross-Subject Variability (B):", value = 0.7, step = 0.01),
                 numericInput("C", "Within-Subject Variability (C):", value = 0.1, step = 0.01),
                 
                 h5("Simulation Parameters"),
                 numericInput("test_size", "Test Set Size (Proportion):", value = 0.3, min = 0.1, max = 0.9, step = 0.1),
                 selectInput("split_method", "Data Split Method:", choices = c("row-wise", "subject-wise"), selected = "row-wise"),
                 numericInput("replications", "Number of Replications:", value = 1, min = 1, step = 1),
                 
                 actionButton("generate_data", "Generate Data"),
                 actionButton("run_sim", "Run Simulation")
               ),
               
               mainPanel(
                 h4("Generated Features"),
                 tableOutput("generated_features"),
                 
                 h4("Visualization"),
                 plotOutput("simulation_plot"),
                 
                 h4("Simulation Results"),
                 tableOutput("simulation_results")
            
               )
             )
    ),


    # Data Upload Tab
    tabPanel("Upload your data",
             sidebarLayout(
               sidebarPanel(
                 h3("Upload Dataset"),
                 fileInput("data_file", "Choose CSV File", accept = c(".csv")),
                 textInput("outcome_variable", "Outcome Variable Name:", value = "y"),
                 textInput("id_variable", "ID/Subject Variable Name:", value = "subject"),
                 textInput("time_variable", "Time Variable Name:", value = "time"),
                 numericInput("n_features_upload", "Number of Features:", value = 10, min = 1, step = 1),
                 numericInput("test_size_upload", "Test Set Size (Proportion):", value = 0.3, min = 0.1, max = 0.9, step = 0.1),
                 numericInput("reps_upload", "Number of Replications:", value = 1, min = 1, step = 1),
                 actionButton("analyze_data", "Analyze Data"),
                 actionButton("run_sim_upload", "Run Simulation")
               ),
               
               mainPanel(
                 h4("Descriptive Statistics"),
                 tableOutput("descriptives"),
                 
                 h4("ICC Predictor Table"),
                 tableOutput("icc_table"),
                 
                 h4("Visualization"),
                 plotOutput("upload_plot"),
                 
                 h4("Simulation Results"),
                 tableOutput("simulation_results_upload")
                 
               )
             )
    ),
    
    tabPanel("Explore study results",
             sidebarLayout(
               sidebarPanel(
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
                 sliderInput("sd_residual_viz", "SD Residual:", min = 0, max = 0.5, value = c(0.1, 0.5)), # Just add 4 different bins
                 checkboxInput("filter_values", "Filter Data")
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
    ),
    
    
  )
)

server <- function(input, output) {
  # Original Simulation Logic
  generated_features <- eventReactive(input$generate_data, {
    # Simulate some data (you'll need to implement create_data)
    create_data(
      n_features = input$n_features,
      n_samples = input$n_samples,
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
    
    ggplot(features_sample, aes(x = time, y = y)) +
      geom_point(color = "#83AF9B") +
      facet_wrap(~subject) +
      labs(x = "Time", y = "Outcome") +
      theme_minimal()
  })
  
  simulation_results <- eventReactive(input$run_sim, {
    req(generated_features())
    run_simulation(
      features_sample = generated_features(),
      cv = input$split_method,
      n_bootstrap = input$replications,
      testsize = input$test_size
    )
  })
  
  output$simulation_results <- renderTable({
    req(simulation_results())
    simulation_results()
  })
  
  # Data Upload Logic
  uploaded_data <- eventReactive(input$analyze_data, {
    req(input$data_file)
    data <- read.csv(input$data_file$datapath)
    data
  })
  
  output$descriptives <- renderTable({
    req(uploaded_data())
    data <- uploaded_data()
    
    overall_prob_outcome <- mean(data[[input$outcome_variable]])
    grouped <- data %>% group_by(data[[input$id_variable]]) %>% summarise(prob = mean(data[[input$outcome_variable]]))
    sd_outcome <- sd(grouped$prob)
    
    n_subjects <- length(unique(data[[input$id_variable]]))
    n_samples <- length(unique(data[[input$time_variable]]))
    
    data.frame(
      `Overall Outcome Probability` = overall_prob_outcome,
      `Outcome SD` = sd_outcome,
      `Number of Subjects` = n_subjects,
      `Number of Samples per Subject` = n_samples
    )
  })
  
  
  output$icc_table <- renderTable({
    req(uploaded_data())
    data <- uploaded_data()
    
    feature_names <- colnames(data)[1:input$n_features_upload]
    icc_data <- data.frame(variable = feature_names, icc = NA, B = NA, C = NA)
    
    for (i in seq_along(feature_names)) {
      model <- lmer(as.formula(paste0(feature_names[i], "~ 1 + (1|", input$id_variable, ")")), data = data)
      var_random <- as.data.frame(VarCorr(model))$vcov[1]
      var_residual <- attr(VarCorr(model), "sc")^2
      icc_data$icc[i] <- var_random / (var_random + var_residual)
      icc_data$B[i] <- as.data.frame(VarCorr(model))$sdcor[1]
      icc_data$C[i] <- as.data.frame(VarCorr(model))$sdcor[2]
    }
    icc_data
  })
  
  simulation_results_upload <- eventReactive(input$run_sim_upload, {
    req(uploaded_data())
    data <- uploaded_data()
    run_simulation(
      features_sample = data,
      cv = "row-wise",
      n_bootstrap = input$reps_upload,
      testsize = input$test_size_upload
    )
  })
  
  output$simulation_results_upload <- renderTable({
    req(simulation_results_upload())
    simulation_results_upload()
  })
  
  
  output$upload_plot <- renderPlot({
    req(uploaded_data())
    data <- uploaded_data()
    
    ggplot(data, aes(x = !!sym(input$time_variable), y = !!sym(input$outcome_variable))) +
      geom_point(color = "#83AF9B") +
      facet_wrap(as.formula(paste("~", input$id_variable))) +
      labs(x = "Time", y = "Outcome") +
      theme_minimal()
  })
  
  ##### Simulation Results Viz ###
  data <- read.csv("simulation_results.csv")

  
  # Simulation Results
  output$plot1 <- renderPlot({
    
    
    if (input$filter_values) {
        data <- subset(data, 
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
    ggplot(data, aes_string(x = input$x_var, y = input$y_var, color = input$color_var)) +
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
      data <- subset(data, 
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

    data$auc_diff <- data$auc_c_individual - data$auc_individual
    
    # Example visualization
    ggplot(data, aes_string(x = input$x_var, y = "auc_diff", color = input$color_var)) +
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
