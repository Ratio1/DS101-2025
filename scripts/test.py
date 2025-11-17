try:
  import numpy as np
  print("NumPy version:", np.__version__)
except ImportError:
  raise ImportError("NumPy is not installed. Please install it to proceed.")  

try:
  import pandas as pd
  print("Pandas version:", pd.__version__)
except ImportError:
  raise ImportError("Pandas is not installed. Please install it to proceed.")

try:
  import torch as th
  print("PyTorch version:", th.__version__)
except ImportError:
  raise ImportError("PyTorch is not installed. Please install it to proceed.")



if __name__ == "__main__":
  
  dct_data = {
    'col1': [1, 2, 3],
    'col2': [4, 5, 6],
  }
  
  df = pd.DataFrame(dct_data)
  print(f"dummy data dataframe:\n{df}")
  
  th_data = th.tensor(df.values)
  
  gpu_available = th.cuda.is_available()
  print(f"GPU available: {gpu_available}")
  device = th.device("cuda" if gpu_available else "cpu")
  
  th_data = th_data.to(device)
  print(f"dummy data tensor on {device}:\n{th_data}")
  
  input("Press Enter to go forth...")
  
  # 1 GiB tensor
  np_data = np.random.rand(1024, 1024, 1024 // 4)
  
  th_data_large = th.tensor(np_data, dtype=th.float32)
  th_data_large = th_data_large.to(device)
  
  print(f"dummy data tensor on {device}:\n{th_data_large}")
  
  input("Press Enter to exit...")
  
  print("Script done.")
  