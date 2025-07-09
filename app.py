import streamlit as st
from greenshelf import get_unmarked_high_risk, mark_down_sku, greenshelf_chat
import google.generativeai as genai
from PIL import Image
import pandas as pd

# --- Initialize session state variables ---
if "ignored_batches" not in st.session_state:
    st.session_state.ignored_batches = set()

# ✅ First Streamlit command
st.set_page_config(page_title="GreenShelf AI", page_icon="🌿", layout="centered")

# ✅ Global title
st.markdown("# 🌿 Welcome to GreenShelf AI")
st.markdown("### 🛒 Your Smart Companion for Reducing Food Waste in Retail")

st.markdown("""
👋 Ask anything related to **food spoilage prediction**, **stock freshness**, or **inventory health**.  
📊 Get instant AI-powered insights, sustainability tracking, and even visual spoilage analysis!
""")

tab1, tab2, tab3 = st.tabs(["🤖 Assistant", "🌍 Sustainability Tracker", "📸 Spoilage Scanner"])

# ----------------------------- TAB 1 -----------------------------
with tab1:
    st.header("🔂 Spoilage Report")

    # --- Load unmarked high-risk batches
    today_summary = get_unmarked_high_risk()

# --- Daily Summary Report ---
    st.markdown("#### 📈 Daily Summary Report")

    if "show_report" not in st.session_state:
        st.session_state.show_report = False

    if st.button("📝 Generate Report"):
        st.session_state.show_report = True

    if st.session_state.show_report:
        st.markdown("#### 📋 Today's Summary Data (Filtered)")
        st.dataframe(today_summary, use_container_width=True)
        st.caption("🔍 Excludes stock batches marked as markdowned (by Store ID + Batch ID).")

    st.divider()

    # --- Ask AI Section ---
    st.markdown("#### 🤖 Ask GreenShelf AI")
    query = st.text_input("💬 *What would you like to know about spoilage today?*", key="query_input")

    if query:
        with st.spinner("🧠 Thinking..."):
            answer = greenshelf_chat(query, today_summary)
            st.success(f"✅ **Answer:** {answer}")

    st.divider()

# --- Marked down Section ---
    st.markdown("### 🚫 Mark Down Stock by Batch ID")

    if not today_summary.empty:
        # Unique store IDs
        store_ids = sorted(today_summary['Store_ID'].unique())
        
        # Create "temp" keys to store selections
        if "selected_store" not in st.session_state:
            st.session_state.selected_store = store_ids[0]
        if "selected_batches" not in st.session_state:
            st.session_state.selected_batches = []

        # Select store ID (controlled by session_state)
        st.session_state.selected_store = st.selectbox(
            "🏬 Select Store ID", 
            store_ids, 
            index=store_ids.index(st.session_state.selected_store)
        )

        # Get batches for this store
        batch_ids = today_summary[today_summary['Store_ID'] == st.session_state.selected_store]['Batch_ID'].unique()

        # Select batches (controlled)
        st.session_state.selected_batches = st.multiselect(
            "🔢 Select Batch IDs to Mark as Markdowned", 
            batch_ids,
            default=st.session_state.selected_batches
        )

        # Only update when button clicked
        if st.button("🚫 Mark as Markdowned"):
            marked_count = 0
            for batch_id in st.session_state.selected_batches:
                # Actually mark down in the CSV
                if mark_down_sku(batch_id, st.session_state.selected_store):
                    marked_count += 1
            st.success(f"✅ Marked {marked_count} batch(es) as markdowned for store {st.session_state.selected_store}.")
            # Reset selections after marking
            st.session_state.selected_batches = []
            st.rerun()


# ----------------------------- TAB 2 -----------------------------
with tab2:
    st.header("🌍 Sustainability Tracker (Impact Metrics)")

    # Load latest data
    today_summary = pd.read_csv("active_summary.csv")

    if not today_summary.empty:
        st.markdown("#### ⚙️ Customize Sustainability Factors")
        col_input1, col_input2, col_input3 = st.columns(3)
        AVG_BATCH_WEIGHT_KG = col_input1.number_input("📦 Avg Batch Weight (kg)", min_value=1.0, value=20.0, step=1.0)
        AVG_BATCH_COST_RS = col_input2.number_input("💰 Avg Batch Cost (₹)", min_value=1.0, value=600.0, step=10.0)
        DEFAULT_CO2_PER_KG = col_input3.number_input("🌱 Default CO₂ per kg", min_value=0.1, value=1.9, step=0.1)

        co2_map = {
            "Dairy": 3.0,
            "Fruits": 2.5,
            "Vegetables": 1.8,
            "Bakery": 2.0,
            "Meat": 4.5,
            "Other": DEFAULT_CO2_PER_KG
        }

        spoiled = today_summary[today_summary["Spoiled"] == 1]
        spoiled_batches_count = spoiled.shape[0]

        total_kg_saved = spoiled_batches_count * AVG_BATCH_WEIGHT_KG
        total_cost_saved = spoiled_batches_count * AVG_BATCH_COST_RS

        total_co2_saved = 0
        for _, row in spoiled.iterrows():
            category = row.get("Category", "Other")
            co2_factor = co2_map.get(category, DEFAULT_CO2_PER_KG)
            total_co2_saved += AVG_BATCH_WEIGHT_KG * co2_factor

        st.markdown("#### ✅ Your sustainability impact so far:")
        col1, col2, col3 = st.columns(3)
        col1.metric("🥕 Food Waste Prevented", f"{total_kg_saved:.2f} kg")
        col2.metric("💸 Estimated Cost Savings", f"₹{total_cost_saved:.2f}")
        col3.metric("🌿 CO₂ Emissions Avoided", f"{total_co2_saved:.2f} kg")

        st.caption("📉 Based on your inputs and category-specific emission factors.")

# ----------------------------- TAB 3 -----------------------------
with tab3:
    st.header("📸 AI Spoilage Detector (Experimental)")
    st.markdown("Upload a top view image of a perishable item/stock to detect spoilage.")
    st.markdown("Note: Expected good image quality and lighting condition for accurate results.")

    uploaded_img = st.file_uploader("🖼️ Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="🔍 Image Preview", use_column_width=True)

        if st.button("🔎 Analyze Spoilage"):
            with st.spinner("Analyzing with Vision..."):
                model = genai.GenerativeModel("gemini-2.0-flash")
                prompt = (
                    "You're a food quality inspector AI helping reduce food waste at Walmart.\n"
                    "Analyze the given image of a food item and do the following:\n\n"
                    "1. Tell whether the food item appears **fresh or spoiled** (use only these terms).\n"
                    "2. Give a **mid-length reason** (e.g., color, texture, mold, dryness, bruising, etc).\n"
                    "3. Estimate a **spoilage risk percentage (0–100%)** based on the visible condition.\n\n"
                    "Reply in this format:\n"

                    "**Status:** Fresh or Spoiled\n"
                    "**Reason:** Short visual cue\n"
                    "**Spoilage Probability:** XX%\n\n"
                    
                    "Be accurate and avoid long responses."
                )
                response = model.generate_content([prompt, img])
                st.success("✅ Analysis Complete")
                st.markdown(f"**🧠 AI Result:** {response.text}")

                # Split the response into lines
                lines = response.text.strip().splitlines()
                for line in lines:
                    st.markdown(f"🧠 {line}")


    st.divider()

    st.markdown("### ⚙️ TinyML Edge Simulation (Coming Soon)")
    st.markdown("""
    Our future scope: GreenShelf AI will simulate on-device spoilage detection using **TinyML** for most accurate results.  
    This means spoilage alerts could run **offline on edge devices** like Raspberry Pi, smart fridges, or shelf sensors.

    🧠 Powered by: `TFLite`, `Edge Impulse`, `Arduino Nano BLE Sense` (planned).
    """)

# --- Footer ---
st.markdown("---")
st.markdown("💡 Built with ❤️ for sustainable retail by *God Level Devs*")
