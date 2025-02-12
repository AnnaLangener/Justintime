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
    )
    
    
  )
)

server <- function(input, output, session) {
 
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
