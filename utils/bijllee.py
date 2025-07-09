import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
days = 14
dates = [datetime.today() - timedelta(days=i) for i in range(days)][::-1]

data = []
cumulative_saved = 0
for i, date in enumerate(dates):
    total_batches = np.random.randint(80, 120)
    spoilage_before = np.random.uniform(0.18, 0.28)
    spoilage_after = spoilage_before - np.random.uniform(0.04, 0.09)
    spoiled_before = int(total_batches * spoilage_before)
    spoiled_after = int(total_batches * spoilage_after)
    food_saved = spoiled_before - spoiled_after
    cumulative_saved += food_saved
    data.append({
        "Date": date.strftime("%Y-%m-%d"),
        "Total_Batches": total_batches,
        "Spoilage_%_Before": round(spoilage_before * 100, 2),
        "Spoilage_%_After": round(spoilage_after * 100, 2),
        "Spoiled_Batches_Before": spoiled_before,
        "Spoiled_Batches_After": spoiled_after,
        "Food_Saved_Batches": food_saved,
        "Cumulative_Food_Saved": cumulative_saved
    })

df = pd.DataFrame(data)
df.to_csv("data/metrics_data.csv", index=False)
print("metrics_data.csv generated.")