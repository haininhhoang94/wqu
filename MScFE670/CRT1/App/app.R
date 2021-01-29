#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

# Getting the csv data in

df = read.csv("CountryDataClean.csv")
df <- as.data.frame(df)
library(NbClust)
library(shiny)
library(factoextra) 
library(dplyr)
# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Week 1 Collaboration Assignment"),

    # Sidebar with a slider input for number of bins 

    fluidRow(
        column(4,sliderInput("cluster",
                        "Number of cluster:",
                        min = 2,
                        max = 10,
                        value = 5)),
        column(4,sliderInput("top_n",
                        "Number of countries:",
                        min = 50,
                        max = 192,
                        value = 100)),

        column(2, selectInput("distance_type", "Distance Type",
                    choices = c("euclidean", "maximum", "manhattan","canberra","minkowski")))
            
        ),
    
        hr(),
    
        plotOutput("clusterplot"),
    
        hr(),
    
        plotOutput(outputId="best_n_cluster")

        # Show a plot of the generated distribution
        # mainPanel("main panel",
        #           fluidRow(
        #               column(4,plotOutput(outputId="clusterplot", width="300px",height="300px")),  
        #               column(8,plotOutput(outputId="best_n_cluster", width="300px",height="300px"))
        #           )
        # )
    
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    
   
     filtered <- reactive({
      
         
         df_filter = df %>% sample_n(input$top_n)
         
         # Scaling our data
         scaled_data = scale(df_filter)
         
         
         res<- NbClust(scaled_data, distance = input$distance_type, min.nc=2, max.nc=10, 
                       method = "kmeans", index = "all")  
         
         n_cluster = input$cluster 
         #distance_type = input$distance_type
         
         # K-means clustering
         km.res <- kmeans(scaled_data, n_cluster, nstart = 20)
         # use ?kmeans in console to explore why we have used nstart = 50
         
         # Cluster visualisation
         fviz_cluster(km.res, 
                      data = scaled_data,
                      ellipse.type = "euclid", 
                      star.plot = TRUE, # Add segments from centroids to items
                      repel = TRUE, # Avoid label overplotting (slow)
                      ggtheme = theme_minimal()
         )
         
     })
    
      filtered1 <- reactive({
          
          
          df_filter = df %>% sample_n(input$top_n)
          
          # Scaling our data
          scaled_data = scale(df_filter)
          
          
          res<- NbClust(scaled_data, distance = input$distance_type, min.nc=2, max.nc=10, 
                        method = "kmeans", index = "all")
          
          
      })
      
    
    output$clusterplot <- renderPlot({
        
        filtered()
        
    })
     
     output$best_n_cluster <- renderPlot({
         
         filtered1()
         
     })
     
    
    
}

# Run the application 
shinyApp(ui = ui, server = server)
