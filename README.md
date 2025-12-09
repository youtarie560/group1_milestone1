# IFT6758 Milestone 3

**Before Typing: ```docker-compose up --build```** make sure to create ```.env``` file in the root of this project (meaning same directory as docker-compose.yaml path)
The content of ```.env``` file should look like this:
```API_KEY=...``` 
Use should use the API KEY from wandb

### Setup

Before running the application, you must configure your environment variables:

1. Create a file named .env in the root directory (alongside docker-compose.yaml).
2. Add your Weights & Biases (wandb) API key to the file:
```API_KEY=your_wandb_api_key_here```


Once the file is created, start the application:

```docker-compose up --build```
