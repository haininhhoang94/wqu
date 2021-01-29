library(shiny)

# Getting the csv data after convert from Xlsx original
df = read.csv("CountryData.csv")
df <- as.data.frame(df)

# Removing blank rows generated
#df<- df[c(1:21)]

# Renaming the vairbles to be human readable :)
#names(df)[3:21]<- c("ForeignInvestment", "ElectricityAccess", "RenewableEnergy", "CO2Emission", "Inflation", "MobileSubscriptions", "InternetUse", "Exports", "Imports", "GDP", "MortalityMale", "MortalityFemale", "BirthRate", "DeathRate", "MortalityInfant", "LifeExpectancy", "FertilityRate", "PopulationGrowth", "UrbanPopulation")

library(NbClust)
library(factoextra) 
library(dplyr)
# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Collaborative Review Task 1"),
  
  # Sidebar with a slider input for number of bins 
  
  fluidRow(
    column(4,sliderInput("cluster",
                         "Number of cluster:",
                         min = 2,
                         max = 10,
                         value = 2)),
    column(4,sliderInput("top_n",
                         "Number of countries:",
                         min = 50,
                         max = 192,
                         value = 50)),
    
    column(2, selectInput("distance_type", "Distance Type",
                          choices = c("euclidean", "maximum", "manhattan","canberra","minkowski")))
    
  ),
  
  hr(),
  
  plotOutput("clusterplot"),
  
  hr(),
  
  plotOutput(outputId="best_n_cluster")
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