library(shiny)
library(bslib)
library(dplyr)
library(lme4)
library(caret)
library(plotly)

# Source external functions
source("Simulation_Functions.R")

# Define UI for application
ui <- fluidPage(
  theme = bs_theme(version = 5, bootswatch = "journal"),
  
  titlePanel("Just in time."),
  
  tabsetPanel(
    # Original Simulation Tab
    tabPanel("Simulation",
             sidebarLayout(
               sidebarPanel(
                 h3("Basic Parameters"),
                 numericInput("n_features", "Number of Features:", value = 10, min = 1, step = 1),
                 numericInput("n_samples", "Number of Samples (per subject):", value = 90, min = 1, step = 1),
                 numericInput("n_subjects", "Number of Subjects (even number):", value = 150, min = 2, step = 2),
                 
                 h3("Outcome Parameters"),
                 numericInput("overall_prob_outcome", "Overall Probability of Outcome:", value = 0.1, min = 0, max = 1, step = 0.01),
                 numericInput("sd_outcome", "Standard Deviation of Outcome (Between Subjects):", value = 0.25, step = 0.01),
                 checkboxInput("time_effect", "Include Time Effect:", value = FALSE),
                 
                 h3("Feature Generation"),
                 numericInput("A", "Relationship Between Features and Outcome (A):", value = 0.05, step = 0.01),
                 numericInput("feature_std", "Population-level Feature Variability (Feature Std):", value = 0.1, step = 0.01),
                 numericInput("B", "Cross-Subject Variability (B):", value = 0.7, step = 0.01),
                 numericInput("C", "Within-Subject Variability (C):", value = 0.1, step = 0.01),
                 
                 h3("Simulation Parameters"),
                 numericInput("test_size", "Test Set Size (Proportion):", value = 0.3, min = 0.1, max = 0.9, step = 0.1),
                 selectInput("split_method", "Data Split Method:", choices = c("row-wise", "column-wise"), selected = "row-wise"),
                 numericInput("replications", "Number of Replications:", value = 1, min = 1, step = 1),
                 
                 actionButton("generate_data", "Generate Data"),
                 actionButton("run_sim", "Run Simulation")
               ),
               
               mainPanel(
                 h4("Generated Features"),
                 tableOutput("generated_features"),
                 
                 h4("Simulation Results"),
                 tableOutput("simulation_results")
               )
             )
    ),
    
    # Data Upload Tab
    tabPanel("Data Upload",
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
                 
                 h4("Simulation Results"),
                 tableOutput("simulation_results_upload")
               )
             )
    )
  )
)

# Define server logic
server <- function(input, output) {
  # Original Simulation Logic
  generated_features <- eventReactive(input$generate_data, {
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
    generated_features()
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
}

# Run the application
shinyApp(ui = ui, server = server)

