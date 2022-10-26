import gdown
import pandas as pd

df = pd.read_csv(f'https://drive.google.com/uc?id=1ByySSLzx23w6zMum7rF71fmMPcwi8XrQ', sep = ';')
for _, row in df.iterrows():
  gdown.download (row[1], f'{row[0]}.jpg', quiet=False)


  

