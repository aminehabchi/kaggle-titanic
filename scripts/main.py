import pandas as pd
from preprocessing import preprocess_data 
from evaluation import evaluate
from training import train
from save_model import save_model
TRAIN_DATA="../data/train.csv"

def main():
    df = pd.read_csv(TRAIN_DATA)
    
    df_clean = preprocess_data(df)
    
    print(df_clean.head())
    
    model = train(df_clean)
    
    evaluate(model)
    
    save_model(model)
    

if __name__ == "__main__":
    main()
   
