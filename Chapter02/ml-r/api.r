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


#* @post /invocations
function(req, res) {
  print(req$postBody)
  model <- load_model()
  
  payload_value <- as.double(req$postBody)
  X_test <- data.frame(payload_value)
  colnames(X_test) <- "X"
  
  print(summary(model))
  y_test <- predict(model, X_test)
  output <- y_test[[1]]
  print(output)
  
  res$body <- toString(output)
  return(res)
}
