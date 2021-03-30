library("rjson")
library("forecast")


prepare_paths <- function() {
    keys <- c('hyperparameters', 
              'input', 
              'data',
              'model')

    values <- c('input/config/hyperparameters.json', 
                'input/config/inputdataconfig.json', 
                'input/data/',
                'model/')
    
    paths <- as.list(values)
    names(paths) <- keys
    
    return(paths);
}

PATHS <- prepare_paths()

get_path <- function(key) {
    output <- paste('/opt/ml/', PATHS[[key]], sep="")
    
    return(output);
}


#* @get /ping
function(res) {
  res$body <- "OK"

  return(res)
}


load_model <- function() {
  model <- NULL
  
  filename <- paste0(get_path('model'), 'model')
  print(filename)
  model <- readRDS(filename)
  
  return(model)
}

process.output <- function(output) {
    output.list <- list(
        mean.values=as.numeric(output$mean),
        mean.start=start(output$mean),
        mean.frequency=frequency(output$mean),
        x.values=as.numeric(output$x),
        x.start=start(output$x),
        x.frequency=frequency(output$x)
    )

    return(output.list)
}


#* @post /invocations
function(req, res) {
  print(req$postBody)
  model <- load_model()
  
  payload_value <- as.integer(req$postBody)
  print(payload_value)
  output <- forecast(model, h=payload_value)
  print(output)
  
  res$body <- toString(toJSON(process.output(output)))
  return(res)
}
