import os
import pandas as pd

# Load CSV
df = pd.read_csv(r"simulated_freshtrack_data_5000.csv")


from sklearn.preprocessing import LabelEncoder

# Create encoders
le_sku = LabelEncoder()
le_cat = LabelEncoder()

# Encode categorical features
df["SKU_Code"] = le_sku.fit_transform(df["SKU"])
df["Category_Code"] = le_cat.fit_transform(df["Category"])

# (Optional) Convert Arrival_Date to datetime if needed
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

# Define feature columns
feature_cols = [
    "Sell_Through_Percent",
    "Days_Since_Arrival",
    "Temperature",
    "Humidity",
    "Shelf_Life",
    "SKU_Code",
    "Category_Code"
]

# Target variable
target_col = "Spoiled"

# Separate features and target
X = df[feature_cols]
y = df[target_col]


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost classifier
ml_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
ml_model.fit(X_train, y_train)

# Predict on test set
y_pred = ml_model.predict(X_test)
y_prob = ml_model.predict_proba(X_test)[:, 1]

def mark_down_sku(batch_id, store_id):
    """
    Mark a batch as markdowned in active_summary.csv by Batch_ID and Store_ID.
    Returns True if successful, False if not found.
    """
    df = pd.read_csv("active_summary.csv")
    # Ensure columns exist
    if "Marked_Down" not in df.columns:
        df["Marked_Down"] = False
    # Find the batch
    mask = (df["Batch_ID"] == batch_id) & (df["Store_ID"] == store_id)
    if mask.any():
        df.loc[mask, "Marked_Down"] = True
        df.to_csv("active_summary.csv", index=False)
        return True
    return False

csv_path = "active_summary.csv"
if os.path.exists(csv_path):
    # Load existing, update Spoilage_Probability, keep Marked_Down
    active_df = pd.read_csv(csv_path)
    # Update encoded columns in case new SKUs/categories
    active_df["SKU_Code"] = le_sku.transform(active_df["SKU"])
    active_df["Category_Code"] = le_cat.transform(active_df["Category"])
    # Update spoilage probability
    active_df["Spoilage_Probability"] = ml_model.predict_proba(active_df[feature_cols])[:, 1]
    # If Marked_Down column missing, add it
    if "Marked_Down" not in active_df.columns:
        active_df["Marked_Down"] = False
        active_df.to_csv(csv_path, index=False)
    df = active_df  # Use this for downstream
    print("✅ Updated Spoilage_Probability in existing active_summary.csv")
else:
    # First run: create new CSV with Marked_Down and Spoilage_Probability
    df["Marked_Down"] = False
    df["Spoilage_Probability"] = ml_model.predict_proba(df[feature_cols])[:, 1]
    df.to_csv(csv_path, index=False)
    print("✅ Created new active_summary.csv with Marked_Down and Spoilage_Probability")


def get_unmarked_high_risk(threshold=0.7):
    df = pd.read_csv("active_summary.csv")
    # Ensure columns exist
    if "Marked_Down" not in df.columns:
        df["Marked_Down"] = False
    if "Spoilage_Probability" not in df.columns:
        raise KeyError("Spoilage_Probability column missing in active_summary.csv")
    # Filter batches that are not marked down and have high spoilage risk
    high_risk_df = df[(df["Spoilage_Probability"] > threshold) & (~df["Marked_Down"])]
    # Sort by highest risk first
    high_risk_df = high_risk_df.sort_values(by="Spoilage_Probability", ascending=False)
    return high_risk_df




print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\n📈 ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")





import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature importance
importances = ml_model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
plt.title("📊 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()





import joblib

# Save model
joblib.dump(ml_model, "freshtrack_spoilage_model.pkl")

# Save label encoders too (needed during prediction)
joblib.dump(le_sku, "le_sku_encoder.pkl")
joblib.dump(le_cat, "le_category_encoder.pkl")

print("✅ Model and encoders saved.")





import pandas as pd

# Load model and encoders
import joblib
ml_model = joblib.load("freshtrack_spoilage_model.pkl")
le_sku = joblib.load("le_sku_encoder.pkl")
le_cat = joblib.load("le_category_encoder.pkl")

# Reload original data
# df = pd.read_csv("simulated_freshtrack_data_5000.csv")
df["SKU_Code"] = le_sku.transform(df["SKU"])
df["Category_Code"] = le_cat.transform(df["Category"])

feature_cols = [
    "Sell_Through_Percent", "Days_Since_Arrival",
    "Temperature", "Humidity", "Shelf_Life",
    "SKU_Code", "Category_Code"
]

# Predict spoilage probabilities
df["Spoilage_Probability"] = ml_model.predict_proba(df[feature_cols])[:, 1]


df.to_csv("active_summary.csv", index=False)

print("✅ Spoilage probabilities calculated and saved to active_summary.csv")

# Filter today's critical batches (say, risk > 0.6)
today_summary = df[df["Spoilage_Probability"] > 0.6]

# Show top 5 risk batches
today_summary = today_summary.sort_values(by="Spoilage_Probability", ascending=False)
today_summary[["Store_ID", "SKU", "Batch_ID", "Sell_Through_Percent", "Days_Since_Arrival",
               "Temperature", "Humidity", "Spoilage_Probability"]].head()




import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into environment

# Set your API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def generate_daily_summary_report(today_summary, date_str):
    # Create the natural language prompt from your DataFrame
    prompt = f"Generate a daily spoilage summary report for store managers for {date_str}:\n\n"

    for _, row in today_summary.iterrows():
        prompt += (
            f"- Store {row['Store_ID']}, *{row['SKU']}* (Batch `{row['Batch_ID']}`) — "
            f"{int(row['Spoilage_Probability'] * 100)}% spoilage risk. "
            f"Days on shelf: {row['Days_Since_Arrival']}, "
            f"Sell-through: {int(row['Sell_Through_Percent'] * 100)}%, "
            f"Temp: {row['Temperature']}°C, Humidity: {row['Humidity']}%\n"
        )

    prompt += (
        "\nSummarize this data as a short report for Telegram, no need for markdown, make it simple and normal text"
        "*Show high-risk batches*, ignore low-risk ones, and give 2-4 quick action suggestions."
        "You can use emojis, dont use markdown, *, /, these things"
    )

    # Gemini API call
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    # Replace any problematic characters that Telegram Markdown V1 dislikes
    final_text = response.text.replace('_', '\\_').replace('*', '\\*')  # Escape Markdown if needed

    return final_text

from datetime import datetime

# Example: top 10 risky batches from earlier
date_today = datetime.today().strftime("%B %d, %Y")
top_batches = today_summary.head(10)  # You can adjust this slice

summary_text = generate_daily_summary_report(top_batches, date_today)
print(summary_text)





def greenshelf_chat(query, df):
    sample_data = df[[
        "Store_ID", "SKU", "Batch_ID", "Spoilage_Probability",
        "Days_Since_Arrival", "Sell_Through_Percent",
        "Temperature", "Humidity"
    ]].head(30)

    rows = ""
    for _, row in sample_data.iterrows():
        rows += (
            f"Store {row['Store_ID']} — {row['SKU']} (Batch {row['Batch_ID']}), "
            f"{int(row['Spoilage_Probability']*100)}% risk, "
            f"{row['Days_Since_Arrival']} days, "
            f"{int(row['Sell_Through_Percent']*100)}% sold, "
            f"🌡 {row['Temperature']}°C, 💧 {row['Humidity']}%\n"
        )

    prompt2 = (
        "You are GreenShelf AI Assistant, helping shop managers minimize food waste.\n"
        f"Here is today’s spoilage data:\n{rows}\n\n"
        f"Answer this user question briefly, use emojis naturally if helpful:\n{query}\n\n"
        "Give answer to users query only if its related to Spoilage risk or prediction, High-risk stock or urgent items, Store-specific spoilage insights, Recommendations for reducing food waste"
        "💡 If the question is unrelated (like finance, HR, entertainment, etc.), politely respond with: I'm sorry, I’m only trained to assist with food spoilage management and related store decisions.\"\n\n"
        "Use short, clear answers. Feel free to use emojis naturally (like 🌡️, 💧, 🚨, ✅), but *never* use markdown symbols like *, _, or /. Keep formatting clean."
    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt2)

    return response.text.strip()



import requests

def send_telegram_alert(report_text, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
    "chat_id": chat_id,
    "text": report_text,
    }

    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("✅ Telegram alert sent!")
    else:
        print("❌ Failed to send alert:", response.text)


telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

send_telegram_alert(summary_text, telegram_bot_token, telegram_chat_id)


import os
import requests
import time

# ✅ Get tokens from environment or set directly
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    TELEGRAM_BOT_TOKEN = "YOUR_ACTUAL_BOT_TOKEN_HERE"

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

OFFSET = None

def get_updates():
    global OFFSET
    params = {"timeout": 100, "offset": OFFSET}
    resp = requests.get(f"{BASE_URL}/getUpdates", params=params)
    if resp.status_code == 200:
        updates = resp.json()["result"]
        if updates:
            OFFSET = updates[-1]["update_id"] + 1
        return updates
    return []

def send_reply(chat_id, text):
    requests.post(f"{BASE_URL}/sendMessage", data={
        "chat_id": chat_id,
        "text": text
    })

def run_bot():
    print("🟢 GreenShelf AI Bot is running... Type 'end' anytime to stop.")

    while True:
        updates = get_updates()
        for update in updates:
            try:
                msg = update["message"]["text"].strip().lower()
                chat_id = update["message"]["chat"]["id"]
                print(f"📩 Incoming query: {msg}")

                # Stop command
                if "end" in msg:
                    send_reply(chat_id, "👋 GreenShelf AI Assistant has been stopped. See you soon!")
                    print("🔴 Bot stopped by user command.")
                    return

                # Mark down command
                if msg.startswith("mark down"):
                    parts = msg.split()
                    if len(parts) >= 3:
                        batch_id = parts[-1].upper()
                        success = mark_down_sku(batch_id)
                        if success:
                            send_reply(chat_id, f"✅ Batch {batch_id} marked down! Showing next urgent batch...")
                        else:
                            send_reply(chat_id, f"⚠️ Batch {batch_id} not found or already marked down.")
                        continue
                    else:
                        send_reply(chat_id, "⚠️ Usage: mark down <Batch_ID>")
                        continue

                # Get current unmarked data
                df = get_unmarked_high_risk()

                # Compose urgent batch message
                if not df.empty:
                    top_batch = df.iloc[0]
                    urgent_msg = (
                        f"🚨 Most urgent batch:\n"
                        f"Store: {top_batch['Store_ID']}\n"
                        f"SKU: {top_batch['SKU']}\n"
                        f"Batch ID: {top_batch['Batch_ID']}\n"
                        f"Risk: {int(top_batch['Spoilage_Probability'] * 100)}%\n"
                        f"Days Since Arrival: {top_batch['Days_Since_Arrival']}\n\n"
                        f"To mark it down, type:\nmark down {top_batch['Batch_ID']}"
                    )
                else:
                    urgent_msg = "🎉 No high-risk unmarked batches left!"

                # Check if question
                if "?" in msg:
                    reply = greenshelf_chat(msg, df)
                    send_reply(chat_id, f"✅ {reply}")
                else:
                    send_reply(chat_id, urgent_msg)

            except Exception as e:
                print("⚠️ Error handling update:", e)

        time.sleep(2)

# ✅ Only run bot if this script is executed directly
if __name__ == "__main__":
    run_bot(today_summary)
