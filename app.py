import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# --- Config & constants ---
st.set_page_config(page_title="Running Shoe Selector", layout="wide")
IMAGE_DIR = Path("images")  # folder for local shoe images

st.markdown("<small style='opacity:0.6'>App loaded ✅</small>", unsafe_allow_html=True)

# --- Load Data with helpful errors & fallback ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("shoes.csv")
        required = [
            "Brand", "Model", "Heel Drop (mm)", "Stack Height (mm)",
            "Weight (g)", "Carbon Plate", "Rocker Type",
            "Support/Stability", "Category",
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ shoes.csv is missing required columns: {missing}")
            st.stop()

        return df

    except FileNotFoundError:
        st.error("❌ Could not find **shoes.csv**. Please place it in the same folder as `app.py`.")
        st.stop()

    except Exception as e:
        st.error(f"❌ Error loading `shoes.csv`: {e}")
        st.stop()

df = load_data()
import numpy as np  # you already import this

def render_shoe_card(row):
    """Render a single shoe card in a compact layout."""
    score = int(row.get("Match Score", 0))

    # Title line
    st.markdown(
        f"<div style='font-size:18px; font-weight:600; margin-bottom:4px;'>"
        f"{row['Brand']} {row['Model']} • Score {score}"
        "</div>",
        unsafe_allow_html=True,
    )

    img_col, info_col = st.columns([1, 1.2])

    # --- Image column ---
    with img_col:
        img_file = str(row.get("Image", "")).strip()
        if img_file:
            if img_file.lower().startswith("http"):
                st.image(img_file, width=300)  # your chosen size
            else:
                img_path = IMAGE_DIR / img_file
                if img_path.is_file():
                    st.image(str(img_path), width=300)
                else:
                    st.write("🖼️ Image not found")
        else:
            st.write("🖼️ No image")

    # --- Info column ---
    with info_col:
        st.write(f"**Heel Drop:** {row['Heel Drop (mm)']} mm")
        st.write(f"**Stack Height:** {row['Stack Height (mm)']} mm")
        st.write(f"**Rocker:** {row['Rocker Type']}")
        st.write(f"**Stability:** {row['Support/Stability']}")
        st.write(f"**Category:** {row['Category']}")
        st.write(f"**Carbon Plate:** {row['Carbon Plate']}")

        # Optional: weight if you’ve added it
        if "Weight (g)" in row.index:
            st.write(f"**Weight:** {row['Weight (g)']} g")

        notes = str(row.get("Notes", "")).strip()
        if notes:
            st.caption(notes)

        # Affiliate link (if present)
        aff = str(row.get("AffiliateLink", "")).strip()
        if aff:
            st.markdown(f"[🔗 View / buy this shoe]({aff})")

    # Small separator
    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)


# Ensure Image column exists even if it's missing in CSV
if "Image" not in df.columns:
    df["Image"] = ""

# Ensure AffiliateLink column exists (for affiliate URLs)
if "AffiliateLink" not in df.columns:
    df["AffiliateLink"] = ""


# Normalize types/strings
df["Carbon Plate"] = df["Carbon Plate"].astype(str).str.strip().str.capitalize()
df["Rocker Type"] = df["Rocker Type"].astype(str).str.strip().str.capitalize()
df["Support/Stability"] = df["Support/Stability"].astype(str).str.strip().str.capitalize()
df["Category"] = df["Category"].astype(str).str.strip()

# --- Session state for search trigger ---
if "run_search" not in st.session_state:
    st.session_state["run_search"] = False

# --- Title ---
st.title("🏃‍♂️ Running Shoe Selector")
st.caption("Find your ideal running shoe based on your profile, use case, and design preferences. " \
"Information is as accurate as possible, but there may be some errors")

with st.expander("About this tool"):
    st.markdown("""
    This tool helps you find running shoes based on:
    - Your experience and weekly mileage  
    - Your primary goal (fitness, racing, speed, recovery)  
    - Design features like heel drop, stack height, rocker and carbon plates

    Data is based on brand informmation as well as  independent measurements, so may differ slightly from brand marketing specs.
    """)
# ---------------- FILTER CONTROLS (MAIN PAGE, COMPACT) ----------------

st.markdown("### Filters")

# --- Row 1: Runner profile (left) + Use case (right) ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Runner profile")
    experience = st.selectbox(
        "Experience level",
        ["Beginner", "Intermediate", "Advanced"],
        index=1,
    )

    primary_goal = st.selectbox(
        "Primary goal",
        [
            "General fitness",
            "Long-distance racing (half/marathon)",
            "Speed-focused (5k/10k)",
            "Recovery / easy miles",
        ],
        index=0,
    )

with col_right:
    st.markdown("#### Use case")
    use_case_options = [
        "Daily / Easy",
        "Tempo / Workout",
        "Racing",
    ]
    selected_use_cases = st.multiselect(
        "Primary use",
        use_case_options,
        default=use_case_options,
        help="Select one or more use cases to filter shoes.",
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Row 2: Geometry (left) + Stability / plate / rocker (right) ---

# helper for bands
def band_to_range(choice, series, low_max, mid_max):
    s_min = float(series.min())
    s_max = float(series.max())
    if choice == "Low":
        return (s_min, min(low_max, s_max))
    elif choice == "Medium":
        return (max(s_min, low_max), min(mid_max, s_max))
    elif choice == "High":
        return (max(s_min, mid_max), s_max)
    else:  # "Any"
        return (s_min, s_max)

drop_series = df["Heel Drop (mm)"]
stack_series = df["Stack Height (mm)"]

col_geo, col_support = st.columns(2)

with col_geo:
    st.markdown("#### Geometry")
    drop_band = st.selectbox(
        "Heel drop",
        ["Any", "Low", "Medium", "High"],
        index=0,
    )
    stack_band = st.selectbox(
        "Stack height",
        ["Any", "Low", "Medium", "High"],
        index=0,
    )

with col_support:
    st.markdown("#### Support & geometry extras")
    stability = st.selectbox(
        "Support / Stability",
        ["Any", "Neutral", "Stable"],
        index=0,
    )
    col_plate, col_rocker = st.columns(2)
    with col_plate:
        carbon = st.selectbox("Carbon plate", ["Any", "Yes", "No", "Nylon"])
    with col_rocker:
        rocker = st.selectbox("Rocker type", ["Any", "Flat", "Moderate", "High"])

# Compute ranges after selections
heel_drop_range = band_to_range(drop_band, drop_series, low_max=4, mid_max=8)
stack_height_range = band_to_range(stack_band, stack_series, low_max=30, mid_max=37)

st.markdown("<br>", unsafe_allow_html=True)

# --- Actions row (small, compact) ---
col_action_1, col_action_2 = st.columns(2)

with col_action_1:
    search_clicked = st.button("🔍 Search shoes")

if search_clicked:
    st.session_state["run_search"] = True

# Alias ranges for scoring function
heel_drop = heel_drop_range
stack_height = stack_height_range


    # ---------------- FILTERING ----------------
cat_lower = df["Category"].astype(str).str.lower()
masks = []

if "Daily / Easy" in selected_use_cases:
    masks.append(cat_lower.str.contains("daily") | cat_lower.str.contains("trainer"))

if "Tempo / Workout" in selected_use_cases:
        masks.append(cat_lower.str.contains("tempo") | cat_lower.str.contains("interval"))

if "Racing" in selected_use_cases:
    masks.append(cat_lower.str.contains("racing") | cat_lower.str.contains("race"))

if masks:
    category_mask = masks[0]
    for m in masks[1:]:
        category_mask |= m
else:
    category_mask = pd.Series(True, index=df.index)

filtered = df.copy()
filtered = filtered[
    (filtered["Heel Drop (mm)"].between(*heel_drop_range)) &
    (filtered["Stack Height (mm)"].between(*stack_height_range)) &
    (category_mask)
]

if carbon != "Any":
    filtered = filtered[filtered["Carbon Plate"].str.capitalize() == carbon]

if rocker != "Any":
    filtered = filtered[filtered["Rocker Type"].str.capitalize() == rocker]

# Stability filter
if stability != "Any":
    filtered = filtered[filtered["Support/Stability"].str.capitalize() == stability]

    # ---------------- SCORING ----------------
def score_row(row):
    s = 0

    # Core matching
    target_drop = sum(heel_drop) / 2
    if abs(row["Heel Drop (mm)"] - target_drop) <= 2:
        s += 2

    target_stack = sum(stack_height) / 2
    if abs(row["Stack Height (mm)"] - target_stack) <= 5:
        s += 2

    if carbon != "Any" and row["Carbon Plate"].capitalize() == carbon:
         s += 2

    if rocker != "Any" and row["Rocker Type"].capitalize() == rocker:
        s += 2

        
    # Runner profile influence
    cat_str = str(row["Category"]).lower()

    if primary_goal == "Long-distance racing (half/marathon)":
        if "racing" in cat_str or "tempo" in cat_str:
            s += 1
    elif primary_goal == "Speed-focused (5k/10k)":
        if "racing" in cat_str or "tempo" in cat_str:
            s += 2
    elif primary_goal == "Recovery / easy miles":
    # Positive bias for appropriate shoes
        if "daily" in cat_str or "trainer" in cat_str or "recovery" in cat_str:
            s += 2
    # Penalty: racing shoes are generally not suitable for recovery
        if "racing" in cat_str:
            s -= 1

    elif primary_goal == "General fitness":
        if "daily" in cat_str or "trainer" in cat_str:
            s += 1

    if experience == "Beginner":
        if "daily" in cat_str and row["Heel Drop (mm)"] >= 6:
            s += 1

    # Penalty: full-on racing shoes are usually not ideal for beginners
        if "racing" in cat_str:
            s -= 1

    elif experience == "Advanced":
        if "racing" in cat_str or "tempo" in cat_str:
            s += 1

    return s

if not filtered.empty:
        filtered = filtered.copy()
        filtered["Match Score"] = filtered.apply(score_row, axis=1)
        filtered = filtered.sort_values(["Match Score", "Stack Height (mm)"],
                                        ascending=[False, True])
        


   # ---------------- COMPARISON ----------------
st.subheader("Compare shoes (optional)")

if filtered.empty:
    st.info("No shoes found for this search. Try broadening your filters.")
else:
    # Labels like "Hoka Clifton 9"
    option_labels = filtered.apply(lambda r: f"{r['Brand']} {r['Model']}", axis=1)
    label_to_index = {label: idx for label, idx in zip(option_labels, filtered.index)}

    selected = st.multiselect(
        "Select 2–4 shoes to compare:",
        option_labels.tolist(),
        help="Pick up to 4 shoes for side-by-side comparison."
    )

    # Validate selection number
    if len(selected) == 0:
        st.caption("Pick at least 2 shoes to compare.")
    elif len(selected) == 1:
        st.info("Select at least 2 shoes to run a comparison.")
    else:
        # Hard cap: first 4 shoes only
        if len(selected) > 4:
            st.warning("Showing the first 4 selected shoes.")
            selected = selected[:4]

        # --- Build numeric comparison rows ---
        comparison_rows = []
        for label in selected:
            r = filtered.loc[label_to_index[label]]
            comparison_rows.append(
                {
                    "Heel Drop (mm)": r["Heel Drop (mm)"],
                    "Stack Height (mm)": r["Stack Height (mm)"],
                    "Weight (g)": r["Weight (g)"] if "Weight (g)" in r.index else None,
                    "Carbon Plate": r["Carbon Plate"],
                    "Rocker": r["Rocker Type"],
                    "Category": r["Category"],
                    "Match Score": r.get("Match Score", 0),
                }
            )

        comp_df = pd.DataFrame(comparison_rows, index=selected)

        st.caption("Side-by-side comparison")
        st.dataframe(comp_df)



    # ---------------- RESULTS LIST ----------------
    st.subheader("Results")

    if filtered.empty:
        st.warning("No shoes found. Try broadening your filters.")
    else:
        # Two columns side by side
        cols = st.columns(2)

        for i, (_, row) in enumerate(filtered.iterrows()):
            with cols[i % 2]:
                render_shoe_card(row)
