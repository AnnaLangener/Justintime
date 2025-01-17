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

# Define UI for application
ui <- fluidPage(
  theme = bs_theme(version = 5, bootswatch = "journal"),
  
  titlePanel("Data Simulation and Analysis"),
  
  sidebarLayout(
    sidebarPanel(
      # Basic Parameters
      h3("Basic Parameters"),
      numericInput("n_features", "Number of Features:", value = 10, min = 1, step = 1),
      numericInput("n_samples", "Number of Samples (per subject):", value = 90, min = 1, step = 1),
      numericInput("n_subjects", "Number of Subjects (even number):", value = 150, min = 2, step = 2),
      
      # Outcome Parameters
      h3("Outcome Parameters"),
      numericInput("overall_prob_outcome", "Overall Probability of Outcome:", value = 0.1, min = 0, max = 1, step = 0.01),
      numericInput("sd_outcome", "Standard Deviation of Outcome (Between Subjects):", value = 0.25, step = 0.01),
      
      checkboxInput("time_effect", "Include Time Effect:", value = FALSE),
      
      # Feature Generation Parameters
      h3("Feature Generation"),
      numericInput("A", "Relationship Between Features and Outcome (A):", value = 0.05, step = 0.01),
      numericInput("feature_std", "Population-level Feature Variability (Feature Std):", value = 0.1, step = 0.01),
      numericInput("B", "Cross-Subject Variability (B):", value = 0.7, step = 0.01),
      numericInput("C", "Within-Subject Variability (C):", value = 0.1, step = 0.01),
      
      # Simulation Parameters
      h3("Simulation Parameters"),
      numericInput("test_size", "Test Set Size (Proportion):", value = 0.3, min = 0.1, max = 0.9, step = 0.1),
      selectInput("split_method", "Data Split Method:", choices = c("row-wise", "column-wise"), selected = "row-wise"),
      numericInput("replications", "Number of Replications:", value = 1, min = 1, step = 1),
      
      # Action Buttons
      actionButton("generate_data", "Generate Data"),
      actionButton("run_sim", "Run Simulation")
    ),
    
    mainPanel(
     
      h4("Simulation Results"),
      tableOutput("simulation_results")
    )
  )
)

# Define server logic
server <- function(input, output) {
  # Generate features when the "Generate Data" button is clicked
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
  
  # Run simulation when the "Run Simulation" button is clicked
  simulation_results <- eventReactive(input$run_sim, {
    req(generated_features())  # Ensure features are generated before running the simulation
    run_simulation(
      generated_features(),
      method = input$split_method,
      replications = input$replications,
      testsize = input$test_size
    )
  })
  

  
  # Display the results of the simulation
  output$simulation_results <- renderTable({
    req(simulation_results())  # Ensure simulation is run before displaying results
    simulation_results()
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
